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

"""Helper stages used by the translation pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.translation.utils.metadata import (
    build_translation_metadata,
    merge_faith_scores_into_metadata,
    reconstruct_messages_with_translation,
)
from nemo_curator.tasks import DocumentBatch


@dataclass
class SkipTranslatedStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Split a batch into already-translated and needs-translation rows."""

    name: str = "SkipTranslatedStage"
    translation_column: str = "translated_text"

    _skipped_rows: list[dict[str, Any]] = field(init=False, repr=False, default_factory=list)
    _original_order_col: str = field(init=False, repr=False, default="_skip_original_idx")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Remove already-translated rows and stash them for later merge."""
        df = batch.to_pandas().copy()

        if self.translation_column not in df.columns:
            logger.info(
                "SkipTranslatedStage: column '{}' not found, processing all {} rows",
                self.translation_column,
                len(df),
            )
            self._skipped_rows = []
            return batch

        df[self._original_order_col] = range(len(df))
        has_translation = df[self.translation_column].notna() & (
            df[self.translation_column].astype(str).str.strip() != ""
        )

        skipped_df = df[has_translation]
        remaining_df = df[~has_translation].reset_index(drop=True)
        self._skipped_rows = skipped_df.to_dict(orient="records")

        logger.info(
            "SkipTranslatedStage: skipping {} already-translated rows, processing {} rows",
            len(self._skipped_rows),
            len(remaining_df),
        )

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=remaining_df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


@dataclass
class MergeSkippedStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Re-merge previously skipped rows back into the translated batch."""

    name: str = "MergeSkippedStage"
    skip_stage: SkipTranslatedStage | None = None

    _COLUMN_DEFAULTS: ClassVar[dict[str, object]] = {
        "faith_fluency": 0.0,
        "faith_accuracy": 0.0,
        "faith_idiomaticity": 0.0,
        "faith_terminology": 0.0,
        "faith_handling_of_format": 0.0,
        "faith_avg": 0.0,
        "faith_parse_failed": False,
        "_seg_translation_pairs": "[]",
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
        if self.skip_stage is None or not self.skip_stage._skipped_rows:
            df = batch.to_pandas()
            order_col = "_skip_original_idx"
            if order_col in df.columns:
                df = df.drop(columns=[order_col])
            return DocumentBatch(
                task_id=batch.task_id,
                dataset_name=batch.dataset_name,
                data=df,
                _metadata=batch._metadata,
                _stage_perf=batch._stage_perf,
            )

        df = batch.to_pandas()
        skipped_df = pd.DataFrame(self.skip_stage._skipped_rows)
        order_col = self.skip_stage._original_order_col
        translation_column = self.skip_stage.translation_column

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
            "MergeSkippedStage: merged {} translated + {} skipped = {} total rows",
            len(df),
            len(skipped_df),
            len(merged),
        )

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=merged,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


@dataclass
class SegmentPairCaptureStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Capture source/target segment pairs before reassembly."""

    name: str = "SegmentPairCaptureStage"

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
            "SegmentPairCaptureStage: captured {} segment pairs across {} documents",
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


@dataclass
class OutputFormattingStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Apply the requested translation output format."""

    name: str = "OutputFormattingStage"
    output_mode: str = "replaced"
    target_lang: str = "zh"
    output_field: str = "translated_text"
    preserve_segment_pairs: bool = False
    reconstruct_messages: bool = False
    messages_field: str = "messages"
    messages_content_field: str = "content"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.output_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        out_cols: list[str] = []
        if self.output_mode in ("raw", "both"):
            out_cols.append("translation_metadata")
        if self.output_mode in ("replaced", "both"):
            out_cols.append(self.output_field)
        if self.reconstruct_messages:
            out_cols.append("translated_messages")
        return ["data"], out_cols

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Apply output formatting to the batch."""
        df = batch.to_pandas().copy()
        if df.empty:
            return batch

        if self.output_mode in ("raw", "both"):
            self._build_metadata_column(df)

        if self.output_mode == "raw" and self.output_field in df.columns:
            df = df.drop(columns=[self.output_field])

        if self.reconstruct_messages and self.messages_field in df.columns:
            self._build_translated_messages(df)

        if "_seg_translation_pairs" in df.columns:
            df = df.drop(columns=["_seg_translation_pairs"])
        helper_cols = [
            col
            for col in (
                "_translation_map",
                "_segmented_translation_map",
                "_faith_source_text",
                "_faith_translated_text",
            )
            if col in df.columns
        ]
        if helper_cols:
            df = df.drop(columns=helper_cols)

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _build_metadata_column(self, df: pd.DataFrame) -> None:
        """Construct the ``translation_metadata`` JSON column."""
        metadata_values: list[str] = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            translated_text = row.get(self.output_field, "")
            segment_pairs_json: str | None = None
            if self.preserve_segment_pairs and "_seg_translation_pairs" in df.columns:
                segment_pairs_json = str(row["_seg_translation_pairs"])

            translation_map = self._parse_optional_json_object(row.get("_translation_map"))
            segmented_translation_map = self._parse_optional_json_object(
                row.get("_segmented_translation_map")
            )

            metadata_values.append(
                build_translation_metadata(
                    target_lang=self.target_lang,
                    translated_text=translated_text,
                    segment_pairs_json=segment_pairs_json,
                    translation_map=translation_map,
                    segmented_translation_map=segmented_translation_map,
                )
            )

        df["translation_metadata"] = metadata_values

    def _build_translated_messages(self, df: pd.DataFrame) -> None:
        """Construct the ``translated_messages`` column from original messages."""
        translated_msgs: list[str] = []
        for idx in range(len(df)):
            raw_messages = df.iloc[idx].get(self.messages_field)
            translated_text = df.iloc[idx].get(self.output_field, "")

            if raw_messages is None:
                translated_msgs.append("[]")
                continue

            if isinstance(raw_messages, str):
                try:
                    messages_list = json.loads(raw_messages)
                except (json.JSONDecodeError, TypeError):
                    translated_msgs.append("[]")
                    continue
            elif isinstance(raw_messages, list):
                messages_list = raw_messages
            else:
                translated_msgs.append("[]")
                continue

            reconstructed = reconstruct_messages_with_translation(
                original_messages=messages_list,
                translated_text=translated_text,
                field_path=self.messages_content_field,
            )
            translated_msgs.append(json.dumps(reconstructed, ensure_ascii=False))

        df["translated_messages"] = translated_msgs

    @staticmethod
    def _parse_optional_json_object(value: Any) -> dict[str, Any] | None:
        """Parse helper JSON emitted by ReassemblyStage when present."""
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, TypeError):
                return None
            if isinstance(parsed, dict):
                return parsed
        return None


@dataclass
class ScoreMergeStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Merge FAITH scores into ``translation_metadata``."""

    name: str = "ScoreMergeStage"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["translation_metadata", "faith_avg"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["translation_metadata"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Merge FAITH scores into the translation_metadata column."""
        df = batch.to_pandas().copy()
        if df.empty or "translation_metadata" not in df.columns:
            return batch

        faith_cols = [
            "faith_fluency",
            "faith_accuracy",
            "faith_idiomaticity",
            "faith_terminology",
            "faith_handling_of_format",
            "faith_avg",
        ]
        available_faith_cols = [col for col in faith_cols if col in df.columns]
        if not available_faith_cols:
            logger.info("ScoreMergeStage: no FAITH score columns found, skipping merge")
            return batch

        updated_metadata: list[str] = []
        for idx in range(len(df)):
            row = df.iloc[idx]
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

            metadata_json = str(row.get("translation_metadata", "{}"))
            updated_metadata.append(
                merge_faith_scores_into_metadata(metadata_json, scores)
            )

        df["translation_metadata"] = updated_metadata
        logger.info("ScoreMergeStage: merged FAITH scores into metadata for {} rows", len(df))

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
