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

"""Stages for skipping and later restoring already-translated rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

_SKIPPED_ROWS_METADATA_KEY = "_skipped_rows_state"


@dataclass
class SkipExistingTranslationsStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Split a batch into already-translated and needs-translation rows."""

    name: str = "SkipExistingTranslationsStage"
    translation_column: str = "translated_text"
    original_order_col: str = "_skip_original_idx"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Remove already-translated rows and stash them for later merge."""
        df = batch.to_pandas().copy()
        metadata = dict(batch._metadata)
        metadata.pop(_SKIPPED_ROWS_METADATA_KEY, None)

        if self.translation_column not in df.columns:
            logger.info(
                "SkipExistingTranslationsStage: column '{}' not found, processing all {} rows",
                self.translation_column,
                len(df),
            )
            return DocumentBatch(
                task_id=batch.task_id,
                dataset_name=batch.dataset_name,
                data=df,
                _metadata=metadata,
                _stage_perf=batch._stage_perf,
            )

        df[self.original_order_col] = range(len(df))
        has_translation = df[self.translation_column].notna() & (
            df[self.translation_column].astype(str).str.strip() != ""
        )

        skipped_rows = df[has_translation].to_dict(orient="records")
        remaining_df = df[~has_translation].reset_index(drop=True)

        if skipped_rows:
            metadata[_SKIPPED_ROWS_METADATA_KEY] = {
                "rows": skipped_rows,
                "order_col": self.original_order_col,
                "translation_column": self.translation_column,
            }

        logger.info(
            "SkipExistingTranslationsStage: skipping {} already-translated rows, processing {} rows",
            len(skipped_rows),
            len(remaining_df),
        )

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=remaining_df,
            _metadata=metadata,
            _stage_perf=batch._stage_perf,
        )


@dataclass
class RestoreSkippedRowsStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Re-merge previously skipped rows back into the translated batch."""

    name: str = "RestoreSkippedRowsStage"

    _COLUMN_DEFAULTS: ClassVar[dict[str, object]] = {
        "faith_fluency": 0.0,
        "faith_accuracy": 0.0,
        "faith_idiomaticity": 0.0,
        "faith_terminology": 0.0,
        "faith_handling_of_format": 0.0,
        "faith_avg": 0.0,
        "faith_parse_failed": False,
        "faith_segment_scores": "[]",
        "_translation_time": 0.0,
        "_translation_error": "",
        "translation_time": 0.0,
        "translation_errors": "",
        "translation_metadata": "{}",
    }

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Merge stashed rows back and restore original order."""
        df = batch.to_pandas()
        metadata = dict(batch._metadata)
        skip_state = metadata.pop(_SKIPPED_ROWS_METADATA_KEY, None)

        if not skip_state or not skip_state.get("rows"):
            order_col = "_skip_original_idx"
            if order_col in df.columns:
                df = df.drop(columns=[order_col])
            return DocumentBatch(
                task_id=batch.task_id,
                dataset_name=batch.dataset_name,
                data=df,
                _metadata=metadata,
                _stage_perf=batch._stage_perf,
            )

        skipped_df = pd.DataFrame(skip_state["rows"])
        order_col = str(skip_state["order_col"])
        translation_column = str(skip_state["translation_column"])

        for col in df.columns:
            if col in skipped_df.columns:
                continue
            if col == translation_column:
                skipped_df[col] = ""
                continue
            skipped_df[col] = self._COLUMN_DEFAULTS.get(col, "")

        merged = pd.concat([df, skipped_df], ignore_index=True)
        if order_col in merged.columns:
            merged = merged.sort_values(order_col).reset_index(drop=True)
            merged = merged.drop(columns=[order_col])

        logger.info(
            "RestoreSkippedRowsStage: merged {} translated + {} skipped = {} total rows",
            len(df),
            len(skipped_df),
            len(merged),
        )

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=merged,
            _metadata=metadata,
            _stage_perf=batch._stage_perf,
        )


__all__ = [
    "RestoreSkippedRowsStage",
    "SkipExistingTranslationsStage",
    "_SKIPPED_ROWS_METADATA_KEY",
]
