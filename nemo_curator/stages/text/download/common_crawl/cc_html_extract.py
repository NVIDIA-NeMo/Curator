# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTML extraction stage for CC pipelines.

``HtmlExtractStage`` accepts one extractor OR a list of extractors.

Single extractor (one actor stage per extractor, classic pipelined approach):

    HtmlExtractStage(TrafilaturaExtractor, "cc_extracted_text_trafilatura")
    HtmlExtractStage(JusTextExtractor,     "cc_extracted_text_justext")
    HtmlExtractStage(ResiliparseExtractor, "cc_extracted_text_resiliparse")

Multi-extractor (all three in one actor, zero intermediate object-store queues):

    HtmlExtractStage(
        [TrafilaturaExtractor, JusTextExtractor, ResiliparseExtractor],
        ["cc_extracted_text_trafilatura", "cc_extracted_text_justext",
         "cc_extracted_text_resiliparse"],
        resources=Resources(cpus=3.0),
    )

The multi-extractor form eliminates the two object-store materialisation
points that exist between three separate stages.  Ray Data cannot fuse
adjacent actor-pool operators automatically; running all extractors in one
actor is the only way to avoid the intermediate queues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd  # noqa: TC002 — used at runtime
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.html_extractors.utils import get_stop_list_dict
from nemo_curator.stages.text.download.utils import decode_html, lang_detect
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.stages.text.download.base.extract import DocumentExtractor


@dataclass
class HtmlExtractStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Actor stage: run one or more HTML extractors over every document in a batch.

    Single-extractor form (one stage per extractor, separate actor pools):

        HtmlExtractStage(TrafilaturaExtractor, "cc_extracted_text_trafilatura")

    Multi-extractor form (all extractors in one actor pool, no intermediate queues):

        HtmlExtractStage(
            [TrafilaturaExtractor, JusTextExtractor, ResiliparseExtractor],
            ["cc_extracted_text_trafilatura", "cc_extracted_text_justext",
             "cc_extracted_text_resiliparse"],
            resources=Resources(cpus=3.0),
        )
    """

    extractor_factory: Callable[[], DocumentExtractor] | list[Callable[[], DocumentExtractor]]
    output_column: str | list[str]
    input_column: str = "content"
    name: str = "html_extract"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _extractors: list[DocumentExtractor] = field(init=False, repr=False, default_factory=list)
    _stop_lists: dict | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if isinstance(self.extractor_factory, list) != isinstance(self.output_column, list):
            msg = "extractor_factory and output_column must both be a list or both be a scalar"
            raise TypeError(msg)
        if isinstance(self.extractor_factory, list) and len(self.extractor_factory) != len(self.output_column):
            msg = (
                f"extractor_factory has {len(self.extractor_factory)} entries "
                f"but output_column has {len(self.output_column)}"
            )
            raise ValueError(msg)

    @property
    def _factories(self) -> list:
        """Normalise extractor_factory to a list — derived from the pickled field."""
        if isinstance(self.extractor_factory, list):
            return self.extractor_factory
        return [self.extractor_factory]

    @property
    def _output_columns(self) -> list[str]:
        """Normalise output_column to a list — derived from the pickled field."""
        if isinstance(self.output_column, list):
            return self.output_column
        return [self.output_column]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        self._extractors = [f() for f in self._factories]
        self._stop_lists = get_stop_list_dict()
        worker_id = worker_metadata.worker_id if worker_metadata is not None else "unknown"
        names = [type(e).__name__ for e in self._extractors]
        logger.info(f"HtmlExtractStage({', '.join(names)}) ready on {worker_id}")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.input_column]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self._output_columns

    def process(self, task: DocumentBatch) -> DocumentBatch:
        df: pd.DataFrame = task.to_pandas()

        # Single pass over rows: decode + lang_detect once per row regardless of
        # how many extractors are configured (was 3x per row with the old outer loop).
        all_results: list[list[str]] = [[] for _ in self._extractors]
        for html_bytes in df[self.input_column]:
            html = decode_html(html_bytes) if html_bytes else None
            if html is None:
                for r in all_results:
                    r.append("")
                continue
            lang = lang_detect(html)
            stop_words = self._stop_lists.get(lang) or self._stop_lists.get("en", frozenset())
            for i, extractor in enumerate(self._extractors):
                try:
                    texts = extractor.extract_text(html, stop_words, lang)
                    all_results[i].append("\n\n".join(texts) if texts else "")
                except Exception:  # noqa: BLE001
                    all_results[i].append("")

        for col, results in zip(self._output_columns, all_results, strict=True):
            df[col] = results

        return DocumentBatch(
            dataset_name=task.dataset_name,
            data=df,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
