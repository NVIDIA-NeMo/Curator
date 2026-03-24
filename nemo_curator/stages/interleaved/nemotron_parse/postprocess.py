# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""CPU postprocess stage: parse model output, align images, build interleaved rows."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pyarrow as pa
from PIL import Image

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.nemotron_parse.utils import (
    DEFAULT_MIN_CROP_PX,
    build_interleaved_rows,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA


@dataclass
class NemotronParsePostprocessStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """CPU stage: parse raw model output and build the final interleaved schema.

    Reads page images from ``binary_content`` and raw Nemotron-Parse output
    from ``text_content``, then constructs one row per element (text, image,
    table, metadata) in the interleaved schema.

    Floater reordering (Pictures/Captions) is applied automatically for
    Nemotron-Parse v1.1 and skipped for v1.2+, based on the ``model_path``
    stored in task metadata by the inference stage.

    Parameters
    ----------
    proc_size
        Default model processor size ``(height, width)``.  Overridden at
        runtime by ``task._metadata["proc_size"]`` when available.
    min_crop_px
        Minimum pixel dimension for image crops.  Smaller crops (typically
        degenerate bboxes) are filtered out.
    """

    proc_size: tuple[int, int] = (2048, 1664)
    min_crop_px: int = DEFAULT_MIN_CROP_PX
    name: str = "nemotron_parse_postprocess"
    resources: Resources = field(default_factory=lambda: Resources(cpus=2.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: InterleavedBatch) -> InterleavedBatch | None:
        pages = task.to_pandas()
        proc_size = tuple(task._metadata.get("proc_size", self.proc_size))
        model_path = task._metadata.get("model_path", "")
        reorder = "v1.1" in model_path or "v1.2" not in model_path

        all_rows: list[dict[str, Any]] = []
        for sample_id, sample_group in pages.groupby("sample_id", sort=False):
            sorted_group = sample_group.sort_values("position")
            url = str(sorted_group["url"].iloc[0])
            pdf_name = str(sorted_group["pdf_name"].iloc[0])

            page_images = [Image.open(io.BytesIO(b)) for b in sorted_group["binary_content"]]
            page_outputs = [str(t) if t else "" for t in sorted_group["text_content"].tolist()]

            all_rows.extend(
                build_interleaved_rows(
                    str(sample_id),
                    url,
                    pdf_name,
                    page_images,
                    page_outputs,
                    proc_size,
                    reorder_floaters=reorder,
                    min_crop_px=self.min_crop_px,
                )
            )

        if not all_rows:
            return None

        final_df = pd.DataFrame(all_rows)
        for col in INTERLEAVED_SCHEMA.names:
            if col not in final_df.columns:
                final_df[col] = None

        return InterleavedBatch(
            task_id=f"{task.task_id}_postprocessed",
            dataset_name=task.dataset_name,
            data=pa.Table.from_pandas(final_df, preserve_index=False),
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
