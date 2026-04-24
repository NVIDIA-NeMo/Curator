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

"""Stage for merging FAITH scores into translation metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.translation.utils.metadata import (
    merge_faith_scores_into_metadata,
)
from nemo_curator.tasks import DocumentBatch


@dataclass
class MergeFaithScoresStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Merge FAITH scores into ``translation_metadata``."""

    name: str = "MergeFaithScoresStage"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["translation_metadata", "faith_avg"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["translation_metadata"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Merge FAITH scores into the translation_metadata column."""
        df = batch.to_pandas().copy()
        if df.empty or "translation_metadata" not in df.columns:
            return batch

        available_faith_cols = [
            col
            for col in (
                "faith_fluency",
                "faith_accuracy",
                "faith_idiomaticity",
                "faith_terminology",
                "faith_handling_of_format",
                "faith_avg",
            )
            if col in df.columns
        ]
        if not available_faith_cols:
            logger.info("MergeFaithScoresStage: no FAITH score columns found, skipping merge")
            return batch

        df["translation_metadata"] = [
            merge_faith_scores_into_metadata(
                str(df.iloc[idx].get("translation_metadata", "{}")),
                self._extract_faith_scores(df.iloc[idx], available_faith_cols),
            )
            for idx in range(len(df))
        ]
        logger.info(
            "MergeFaithScoresStage: merged FAITH scores into metadata for {} rows",
            len(df),
        )

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    @staticmethod
    def _extract_faith_scores(row: pd.Series, available_faith_cols: list[str]) -> dict[str, Any]:
        """Build the FAITH score payload expected by metadata merging."""
        scores: dict[str, Any] = {}
        for col in available_faith_cols:
            val = row.get(col)
            if pd.notna(val):
                key = col.replace("faith_", "").title()
                if key == "Avg":
                    key = "average"
                elif key == "Handling_Of_Format":
                    key = "Handling_of_Format"
                scores[key] = float(val)
        return scores


__all__ = ["MergeFaithScoresStage"]
