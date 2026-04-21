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

"""Translation pipeline stages and the composite pipeline wrapper."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd
from loguru import logger

from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.translation.faith_eval import FaithEvalFilter
from nemo_curator.stages.text.translation.output_utils import (
    build_translation_metadata,
    merge_faith_scores_into_metadata,
    reconstruct_messages_with_translation,
)
from nemo_curator.stages.text.translation.reassembly import ReassemblyStage
from nemo_curator.stages.text.translation.segmentation import SegmentationStage
from nemo_curator.stages.text.translation.translate import TranslateStage
from nemo_curator.tasks import DocumentBatch

# Valid output modes for the pipeline
_VALID_OUTPUT_MODES = {"replaced", "raw", "both"}


def _needs_structured_faith_helpers(text_field: str | list[str]) -> bool:
    """Return whether FAITH needs flattened helper columns."""
    if isinstance(text_field, list):
        return True
    return "*" in text_field or "." in text_field


@dataclass
class SkipTranslatedStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Split a batch into already-translated and needs-translation rows.

    Rows where ``translation_column`` is non-empty and non-null are stashed
    in ``_skipped_rows`` (accessible on this stage instance) and removed from
    the batch so that downstream stages only process untranslated rows.

    After the rest of the pipeline runs, :class:`MergeSkippedStage` re-merges
    the stashed rows back into the result.

    Attributes:
        translation_column: Column name to check for existing translations.
    """

    name: str = "SkipTranslatedStage"
    translation_column: str = "translated_text"

    _skipped_rows: list[dict[str, Any]] = field(init=False, repr=False, default_factory=list)
    _original_order_col: str = field(init=False, repr=False, default="_skip_original_idx")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Remove already-translated rows, stash them for later merge."""
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
        remaining_df = df[~has_translation]

        self._skipped_rows = skipped_df.to_dict(orient="records")

        logger.info(
            "SkipTranslatedStage: skipping {} already-translated rows, "
            "processing {} rows",
            len(self._skipped_rows),
            len(remaining_df),
        )

        remaining_df = remaining_df.reset_index(drop=True)

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=remaining_df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


@dataclass
class MergeSkippedStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Re-merge previously skipped rows back into the translated batch.

    Restores original row order using the ``_skip_original_idx`` column
    written by :class:`SkipTranslatedStage`.

    Attributes:
        skip_stage: Reference to the corresponding ``SkipTranslatedStage``
            from which to read stashed rows.
    """

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
        """Group segments by doc_id and build per-document pair JSON."""
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

        total_pairs = sum(len(v) for v in doc_pairs.values())
        logger.info(
            "SegmentPairCaptureStage: captured {} segment pairs across {} documents",
            total_pairs,
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

        if self.output_mode == "raw":
            if self.output_field in df.columns:
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

        available_faith_cols = [c for c in faith_cols if c in df.columns]
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


@dataclass(kw_only=True)
class TranslationPipeline(CompositeStage[DocumentBatch, DocumentBatch]):
    """Compose segmentation, translation, reassembly, and optional scoring."""

    name: str = "TranslationPipeline"

    source_lang: str = "en"
    target_lang: str = "zh"
    text_field: str = "text"
    output_field: str = "translated_text"
    segmentation_mode: str = "coarse"
    min_segment_chars: int = 0

    client: AsyncLLMClient | None = None
    model_name: str = ""
    generation_config: GenerationConfig | None = None

    backend_type: str = "llm"
    backend_config: dict = field(default_factory=dict)

    enable_faith_eval: bool = False
    faith_threshold: float = 2.5
    faith_model_name: str = ""
    segment_level: bool = False
    filter_enabled: bool = True

    preserve_segment_pairs: bool = False
    output_mode: str = "replaced"
    merge_scores: bool = False
    reconstruct_messages: bool = False
    messages_field: str = "messages"
    messages_content_field: str = "content"
    skip_translated: bool = False
    translation_column: str = "translated_text"

    def __post_init__(self) -> None:
        if self.output_mode not in _VALID_OUTPUT_MODES:
            raise ValueError(
                f"Invalid output_mode '{self.output_mode}'. "
                f"Must be one of: {sorted(_VALID_OUTPUT_MODES)}"
            )

        if self.merge_scores and self.output_mode == "replaced":
            raise ValueError(
                "merge_scores=True requires output_mode in {'raw','both'}. "
                "Got output_mode='replaced'. Set output_mode='both' explicitly."
            )

        if self.merge_scores and not self.enable_faith_eval:
            logger.warning(
                "merge_scores=True but enable_faith_eval=False; "
                "score merging will be skipped"
            )

        if self.segment_level and not self.preserve_segment_pairs:
            raise ValueError(
                "segment_level=True requires preserve_segment_pairs=True "
                "so that SegmentPairCaptureStage writes the "
                "'_seg_translation_pairs' column consumed by FaithEvalFilter."
            )

        super().__init__()
        self.stages = self._build_stages()

    def _build_stages(self) -> list[ProcessingStage]:
        """Construct the ordered list of sub-stages."""
        stages: list[ProcessingStage] = []
        faith_helper_needed = self.enable_faith_eval and _needs_structured_faith_helpers(
            self.text_field
        )

        skip_stage: SkipTranslatedStage | None = None
        if self.skip_translated:
            skip_stage = SkipTranslatedStage(
                translation_column=self.translation_column,
            )
            stages.append(skip_stage)

        stages.append(
            SegmentationStage(
                text_field=self.text_field,
                source_lang=self.source_lang,
                mode=self.segmentation_mode,
                min_segment_chars=self.min_segment_chars,
            )
        )
        stages.append(
            TranslateStage(
                client=self.client,
                model_name=self.model_name,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                backend_type=self.backend_type,
                backend_config=self.backend_config,
                generation_config=self.generation_config,
            )
        )

        if self.preserve_segment_pairs:
            stages.append(SegmentPairCaptureStage())

        stages.append(
            ReassemblyStage(
                text_field=self.text_field,
                output_field=self.output_field,
                replace_source_fields=self.output_mode in ("replaced", "both"),
                emit_metadata_helpers=self.output_mode in ("raw", "both"),
                emit_faith_helpers=faith_helper_needed,
            )
        )

        if self.skip_translated and skip_stage is not None:
            stages.append(
                MergeSkippedStage(skip_stage=skip_stage)
            )

        if self.enable_faith_eval:
            faith_model = self.faith_model_name or self.model_name
            faith_source_field = (
                "_faith_source_text" if faith_helper_needed else self.text_field
            )
            faith_translated_field = (
                "_faith_translated_text" if faith_helper_needed else self.output_field
            )

            stages.append(
                FaithEvalFilter(
                    client=self.client,
                    model_name=faith_model,
                    source_lang=self.source_lang,
                    target_lang=self.target_lang,
                    source_text_field=faith_source_field,
                    translated_text_field=faith_translated_field,
                    threshold=self.faith_threshold,
                    filter_enabled=self.filter_enabled,
                    segment_level=self.segment_level,
                ),
            )

        needs_formatting = (
            self.output_mode != "replaced"
            or self.reconstruct_messages
            or self.preserve_segment_pairs
            or faith_helper_needed
        )
        if needs_formatting:
            stages.append(
                OutputFormattingStage(
                    output_mode=self.output_mode,
                    target_lang=self.target_lang,
                    output_field=self.output_field,
                    preserve_segment_pairs=self.preserve_segment_pairs,
                    reconstruct_messages=self.reconstruct_messages,
                    messages_field=self.messages_field,
                    messages_content_field=self.messages_content_field,
                )
            )

        if self.enable_faith_eval and self.merge_scores and self.output_mode in ("raw", "both"):
            stages.append(ScoreMergeStage())

        return stages

    def decompose(self) -> list[ProcessingStage]:
        """Return the ordered sub-stages for pipeline execution."""
        return self.stages
