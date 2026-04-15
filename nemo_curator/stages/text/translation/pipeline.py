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

"""
TranslationPipeline -- CompositeStage that wires together the full
translation workflow: Segmentation -> Translation -> Reassembly -> (optional) FAITH evaluation.

Follows the ``FineMathClassifier`` pattern: ``@dataclass(kw_only=True)``,
``__post_init__`` calls ``super().__init__()``, builds the stage list,
and ``decompose()`` returns it.

Additional capabilities (gap-fill features):
- **Gap 3.2**: Segment pair capture (``preserve_segment_pairs``)
- **Gap 4.1/4.2**: Dual output mode (``output_mode``)
- **Gap 5.2**: Score merging into metadata (``merge_scores``)
- **Gap 5.4**: Message reconstruction (``reconstruct_messages``)
- **Gap 9.1**: Skip-already-translated rows (``skip_translated``)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from loguru import logger

from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.translation.faith_eval import FaithEvalFilter
from nemo_curator.stages.text.translation.output_utils import (
    build_segment_pairs,
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


# ---------------------------------------------------------------------------
# Lightweight helper stages
# ---------------------------------------------------------------------------


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

    # Internal: stashed rows that already have translations
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

        # Add original index for deterministic re-merge
        df[self._original_order_col] = range(len(df))

        # Identify rows that already have a non-empty, non-null translation
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

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Merge stashed rows back and restore original order."""
        if self.skip_stage is None or not self.skip_stage._skipped_rows:
            # Nothing was skipped; just drop the helper column if present
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

        # Fill columns that downstream stages added to processed rows but
        # are absent from skipped rows.  This prevents NaN values from
        # confusing downstream consumers.
        _COLUMN_DEFAULTS: dict[str, object] = {
            "translated_text": None,  # sentinel; handled specially below
            "faith_fluency": 0.0,
            "faith_accuracy": 0.0,
            "faith_idiomaticity": 0.0,
            "faith_terminology": 0.0,
            "faith_handling_of_format": 0.0,
            "faith_avg": 0.0,
            "translation_metadata": "{}",
            "translation_time": 0.0,
            "translation_errors": "",
        }

        for col in df.columns:
            if col not in skipped_df.columns:
                if col == self.skip_stage.translation_column:
                    # Skipped rows already had translations; use existing value
                    # (already present in skipped_df under translation_col)
                    pass
                elif col in _COLUMN_DEFAULTS:
                    skipped_df[col] = _COLUMN_DEFAULTS[col]
                else:
                    # Unknown new column -- fill with empty string
                    skipped_df[col] = ""

        # If translated_text column was added by the pipeline but skipped rows
        # already have existing translations under the same column name, the
        # column is already present and no action is needed.

        # Merge
        merged = pd.concat([df, skipped_df], ignore_index=True)

        # Restore original order if the column exists
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
    """Capture source/target segment pairs before reassembly.

    Reads ``_seg_segments``, ``_translated``, and ``_seg_doc_id`` columns
    (written by SegmentationStage / TranslateStage), groups by doc_id,
    and writes a ``_seg_translation_pairs`` column containing a JSON-serialized
    list of ``{"src": ..., "tgt": ...}`` dicts per document row.

    This stage runs BETWEEN TranslateStage and ReassemblyStage.  It attaches
    the pairs column to each segment row so that ReassemblyStage (which groups
    by doc_id and takes the first row) carries the pairs forward.
    """

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

        # Build pairs grouped by document
        doc_pairs: dict[Any, list[dict[str, str]]] = {}
        for _, row in df.iterrows():
            doc_id = row["_seg_doc_id"]
            segment = str(row.get("_seg_segments", ""))
            translation = str(row.get("_translated", ""))
            doc_pairs.setdefault(doc_id, []).append({"src": segment, "tgt": translation})

        # Serialize all pairs for each document.  Write the full document-level
        # JSON string on the *first* row of each group (the one ReassemblyStage
        # will pick up) and an empty list on the rest.
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
    """Format the pipeline output according to the chosen ``output_mode``.

    Modes:
    - ``"replaced"`` (default): only ``translated_text`` column (current behaviour).
    - ``"raw"``: add a ``translation_metadata`` column with JSON structure and
      remove the ``translated_text`` column.
    - ``"both"``: keep both ``translated_text`` and ``translation_metadata``.

    When ``reconstruct_messages`` is ``True`` and a ``messages_field`` column
    exists, an additional ``translated_messages`` column is written containing
    the original messages structure with content replaced by translated text.

    Attributes:
        output_mode: One of ``"replaced"``, ``"raw"``, ``"both"``.
        target_lang: ISO 639-1 target language code.
        output_field: Name of the reassembled translation column.
        preserve_segment_pairs: Whether segment pairs were captured upstream.
        reconstruct_messages: Whether to build a ``translated_messages`` column.
        messages_field: Column containing original OpenAI-format messages.
        messages_content_field: Field path within each message to replace.
    """

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
            # Remove the flat translated_text column; metadata is the output
            if self.output_field in df.columns:
                df = df.drop(columns=[self.output_field])

        if self.reconstruct_messages and self.messages_field in df.columns:
            self._build_translated_messages(df)

        # Clean up internal pairs column if it leaked through
        if "_seg_translation_pairs" in df.columns and self.output_mode == "replaced":
            df = df.drop(columns=["_seg_translation_pairs"])

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
            translated_text = str(df.iloc[idx].get(self.output_field, ""))

            segment_pairs_json: str | None = None
            if self.preserve_segment_pairs and "_seg_translation_pairs" in df.columns:
                segment_pairs_json = str(df.iloc[idx]["_seg_translation_pairs"])

            metadata_values.append(
                build_translation_metadata(
                    target_lang=self.target_lang,
                    translated_text=translated_text,
                    segment_pairs_json=segment_pairs_json,
                )
            )

        df["translation_metadata"] = metadata_values

    def _build_translated_messages(self, df: pd.DataFrame) -> None:
        """Construct the ``translated_messages`` column from original messages."""
        translated_msgs: list[str] = []
        for idx in range(len(df)):
            raw_messages = df.iloc[idx].get(self.messages_field)
            translated_text = str(df.iloc[idx].get(self.output_field, ""))

            if raw_messages is None:
                translated_msgs.append("[]")
                continue

            # Parse messages if stored as JSON string
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


@dataclass
class ScoreMergeStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Merge FAITH evaluation scores into the ``translation_metadata`` column.

    Reads per-row FAITH score columns (``faith_fluency``, ``faith_accuracy``,
    etc.) and folds them into the ``translation_metadata`` JSON under a
    ``"faith_scores"`` key.

    This stage only runs when both ``merge_scores=True`` and FAITH eval is
    enabled.
    """

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

        # Check which faith columns actually exist
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
                    # Use clean key names (strip "faith_" prefix)
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TranslationPipeline(CompositeStage[DocumentBatch, DocumentBatch]):
    """End-to-end translation pipeline composed of sub-stages.

    Decomposes into:
        1. (optional) ``SkipTranslatedStage`` -- skip already-translated rows.
        2. ``SegmentationStage`` -- split documents into translatable segments.
        3. ``TranslateStage`` -- translate each segment via LLM or external backend.
        4. (optional) ``SegmentPairCaptureStage`` -- capture src/tgt pairs.
        5. ``ReassemblyStage`` -- stitch translated segments back into documents.
        6. (optional) ``MergeSkippedStage`` -- re-merge skipped rows.
        7. (optional) ``FaithEvalFilter`` -- score and filter translations.
        8. (optional) ``OutputFormattingStage`` -- build metadata / dual output.
        9. (optional) ``ScoreMergeStage`` -- fold FAITH scores into metadata.

    Parameters
    ----------
    source_lang : str
        ISO 639-1 source language code (default ``"en"``).
    target_lang : str
        ISO 639-1 target language code (default ``"zh"``).
    text_field : str
        Input column containing the source text.
    output_field : str
        Output column for the reassembled translated text.
    segmentation_mode : str
        ``"coarse"`` (line-level) or ``"fine"`` (sentence-level).
    client : AsyncLLMClient | None
        Async LLM client used by ``TranslateStage`` and ``FaithEvalFilter``.
    model_name : str
        LLM model identifier for translation.
    generation_config : GenerationConfig | None
        LLM generation parameters for translation.
    backend_type : str
        Translation backend: ``"llm"``, ``"google"``, ``"aws"``, or ``"nmt"``.
    backend_config : dict
        Backend-specific configuration (passed to the backend factory).
    enable_faith_eval : bool
        Whether to append a ``FaithEvalFilter`` stage.
    faith_threshold : float
        Minimum ``faith_avg`` score to keep a row (only used when FAITH is enabled).
    faith_model_name : str
        LLM model for FAITH scoring.  Falls back to ``model_name`` when empty.
    preserve_segment_pairs : bool
        When ``True``, capture per-segment source/target pairs before reassembly
        in a ``_seg_translation_pairs`` column (Gap 3.2).
    output_mode : str
        Output format: ``"replaced"`` (default, flat translated_text column),
        ``"raw"`` (translation_metadata JSON column only), or ``"both"``
        (both columns).  See Gap 4.1.
    merge_scores : bool
        When ``True`` **and** FAITH eval is enabled, fold FAITH scores into
        the ``translation_metadata`` JSON (Gap 5.2).  Requires
        ``output_mode`` to be ``"raw"`` or ``"both"``.
    reconstruct_messages : bool
        When ``True``, build a ``translated_messages`` column from the original
        OpenAI-format messages with content replaced (Gap 5.4).
    messages_field : str
        Column name containing original messages (for ``reconstruct_messages``).
    messages_content_field : str
        Dot-path to the content field within each message dict.
    skip_translated : bool
        When ``True``, rows where ``translation_column`` already has a
        non-empty value are passed through without re-translation (Gap 9.1).
    translation_column : str
        Column to check for existing translations (used with ``skip_translated``).
    """

    name: str = "TranslationPipeline"

    # Translation config
    source_lang: str = "en"
    target_lang: str = "zh"
    text_field: str = "text"
    output_field: str = "translated_text"
    segmentation_mode: str = "coarse"

    # LLM config (used for translate + optional faith eval)
    client: AsyncLLMClient | None = None
    model_name: str = ""
    generation_config: GenerationConfig | None = None

    # Backend config (for non-LLM backends)
    backend_type: str = "llm"
    backend_config: dict = field(default_factory=dict)

    # Faith eval config (optional)
    enable_faith_eval: bool = False
    faith_threshold: float = 2.5
    faith_model_name: str = ""

    # Gap 3.2: Segmented translation tracking
    preserve_segment_pairs: bool = False

    # Gap 4.1: Dual output
    output_mode: str = "replaced"

    # Gap 5.2: Score merging
    merge_scores: bool = False

    # Gap 5.4: Message reconstruction
    reconstruct_messages: bool = False
    messages_field: str = "messages"
    messages_content_field: str = "content"

    # Gap 9.1: Skip already-translated
    skip_translated: bool = False
    translation_column: str = "translated_text"

    def __post_init__(self) -> None:
        # Validate output_mode
        if self.output_mode not in _VALID_OUTPUT_MODES:
            raise ValueError(
                f"Invalid output_mode '{self.output_mode}'. "
                f"Must be one of: {sorted(_VALID_OUTPUT_MODES)}"
            )

        # merge_scores requires metadata output
        if self.merge_scores and self.output_mode == "replaced":
            logger.warning(
                "merge_scores=True requires output_mode='raw' or 'both'; "
                "auto-upgrading output_mode to 'both'"
            )
            self.output_mode = "both"

        # merge_scores requires faith eval
        if self.merge_scores and not self.enable_faith_eval:
            logger.warning(
                "merge_scores=True but enable_faith_eval=False; "
                "score merging will be skipped"
            )

        super().__init__()
        self.stages = self._build_stages()

    def _build_stages(self) -> list[ProcessingStage]:
        """Construct the ordered list of sub-stages."""
        stages: list[ProcessingStage] = []

        # --- Gap 9.1: Skip-already-translated rows ---
        skip_stage: SkipTranslatedStage | None = None
        if self.skip_translated:
            skip_stage = SkipTranslatedStage(
                translation_column=self.translation_column,
            )
            stages.append(skip_stage)

        # --- Core pipeline: Segmentation -> Translate ---
        stages.append(
            SegmentationStage(
                text_field=self.text_field,
                source_lang=self.source_lang,
                mode=self.segmentation_mode,
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

        # --- Gap 3.2: Segment pair capture (before reassembly) ---
        if self.preserve_segment_pairs:
            stages.append(SegmentPairCaptureStage())

        # --- Core pipeline: Reassembly ---
        stages.append(
            ReassemblyStage(
                text_field=self.text_field,
                output_field=self.output_field,
            )
        )

        # --- Gap 9.1: Merge skipped rows back ---
        if self.skip_translated and skip_stage is not None:
            stages.append(
                MergeSkippedStage(skip_stage=skip_stage)
            )

        # --- FAITH evaluation ---
        # IMPORTANT: FaithEvalFilter runs BEFORE OutputFormattingStage because
        # OutputFormattingStage in "raw" mode drops the translated_text column,
        # which FaithEvalFilter needs as input.
        if self.enable_faith_eval:
            faith_model = self.faith_model_name or self.model_name

            stages.append(
                FaithEvalFilter(
                    client=self.client,
                    model_name=faith_model,
                    source_lang=self.source_lang,
                    target_lang=self.target_lang,
                    source_text_field=self.text_field,
                    translated_text_field=self.output_field,
                    threshold=self.faith_threshold,
                ),
            )

        # --- Gap 4.1/4.2/5.4: Output formatting ---
        needs_formatting = (
            self.output_mode != "replaced"
            or self.reconstruct_messages
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

        # --- Gap 5.2: Score merging (after both FAITH eval and output formatting) ---
        if self.enable_faith_eval and self.merge_scores and self.output_mode in ("raw", "both"):
            stages.append(ScoreMergeStage())

        return stages

    def decompose(self) -> list[ProcessingStage]:
        """Return the ordered sub-stages for pipeline execution."""
        return self.stages
