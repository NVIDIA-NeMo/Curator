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

"""Stage for capturing per-document segment translation pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch


@dataclass
class CaptureSegmentPairsStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Capture source/target segment pairs before reassembly."""

    name: str = "CaptureSegmentPairsStage"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["_seg_segments", "_translated", "_seg_doc_id"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["_seg_translation_pairs"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Group segments by document and write per-document pair JSON."""
        df = batch.to_pandas().copy()

        if df.empty:
            df["_seg_translation_pairs"] = pd.Series(dtype="object")
            return DocumentBatch(
                task_id=batch.task_id,
                dataset_name=batch.dataset_name,
                data=df,
                _metadata=batch._metadata,
                _stage_perf=batch._stage_perf,
            )

        doc_pairs: dict[Any, list[dict[str, str]]] = {}
        for _, row in df.iterrows():
            doc_id = row["_seg_doc_id"]
            segment = str(row.get("_seg_segments", ""))
            translation = str(row.get("_translated", ""))
            doc_pairs.setdefault(doc_id, []).append({"src": segment, "tgt": translation})

        pairs_col: list[str] = []
        seen_docs: set[Any] = set()
        for _, row in df.iterrows():
            doc_id = row["_seg_doc_id"]
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                pairs_col.append(json.dumps(doc_pairs[doc_id], ensure_ascii=False))
            else:
                pairs_col.append("[]")

        df["_seg_translation_pairs"] = pairs_col
        logger.info(
            "CaptureSegmentPairsStage: captured {} segment pairs across {} documents",
            sum(len(v) for v in doc_pairs.values()),
            len(doc_pairs),
        )

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


__all__ = ["CaptureSegmentPairsStage"]
