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

"""FAITH-based translation quality scoring and optional filtering."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

from nemo_curator.stages.text.translation.utils.async_utils import run_async_safe
from nemo_curator.stages.text.translation.utils.prompt_loader import (
    load_prompt_template,
)
from nemo_curator.stages.text.utils.text_utils import get_language_name

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


FAITH_KEYS = ["Fluency", "Accuracy", "Idiomaticity", "Terminology", "Handling_of_Format"]

# Column names written to the output DataFrame
_SCORE_COLUMNS = [
    "faith_fluency",
    "faith_accuracy",
    "faith_idiomaticity",
    "faith_terminology",
    "faith_handling_of_format",
    "faith_avg",
]


@dataclass
class FaithEvalFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """LLM-based translation quality filter using the FAITH metric.

    For each row in the incoming ``DocumentBatch``, this stage:
    1. Formats a FAITH evaluation prompt with source and translated text.
    2. Calls the LLM via ``AsyncLLMClient`` to obtain a JSON score response.
    3. Parses the response for 5 FAITH dimension scores.
    4. Computes ``faith_avg`` (mean of the 5 scores).
    5. Optionally drops rows where ``faith_avg < threshold`` (when ``filter_enabled=True``).

    When ``segment_level=True`` and a ``_seg_translation_pairs`` column is present,
    each document's segment pairs are scored independently and per-document averages
    are computed via ``_average_scores()``.  Both ``faith_avg`` and (optionally)
    ``faith_segment_scores`` columns are produced.

    Parameters
    ----------
    client : AsyncLLMClient | None
        Async LLM client for scoring. Must not be None.
    model_name : str
        LLM model identifier to use for scoring.
    source_lang : str
        ISO 639-1 code of the source language (e.g. ``"en"``).
    target_lang : str
        ISO 639-1 code of the target language (e.g. ``"zh"``).
    source_text_field : str
        Column name containing the original source text.
    translated_text_field : str
        Column name containing the translated text.
    threshold : float
        Minimum ``faith_avg`` score to keep a row. Rows below this are dropped
        (only when ``filter_enabled=True``).
    filter_enabled : bool
        When ``True`` (default), rows with ``faith_avg < threshold`` are dropped.
        When ``False``, all rows are kept with their scores attached, enabling
        downstream score analysis before committing to a threshold.
    segment_level : bool
        When ``True``, look for a ``_seg_translation_pairs`` column containing
        JSON-serialized ``[{"src": ..., "tgt": ...}, ...]`` pairs per document.
        Each segment is scored independently, and document-level scores are
        computed as the average across segments.  Falls back to whole-document
        scoring if the column is absent.
    generation_config : GenerationConfig | None
        LLM generation parameters. Defaults to ``temperature=0.0, max_tokens=256``.
    """

    name: str = "FaithEvalFilter"
    client: AsyncLLMClient | None = None
    model_name: str = ""
    source_lang: str = "en"
    target_lang: str = "zh"
    source_text_field: str = "text"
    translated_text_field: str = "translated_text"
    threshold: float = 2.5
    filter_enabled: bool = True
    segment_level: bool = False
    generation_config: GenerationConfig | None = None
    max_concurrent_requests: int = 64

    # -- internal state (not constructor args) ---------------------------------
    _system_prompt: str = field(init=False, repr=False, default="")
    _user_template: str = field(init=False, repr=False, default="")
    _initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        if self.client is None:
            raise ValueError("FaithEvalFilter requires a non-None 'client' (AsyncLLMClient)")

    # ------------------------------------------------------------------
    # ProcessingStage interface
    # ------------------------------------------------------------------

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.source_text_field, self.translated_text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        # process() always writes faith_parse_failed, and additionally writes
        # faith_segment_scores when segment_level=True.  Declare both so the
        # stage contract matches what downstream stages can rely on.
        out_cols = list(_SCORE_COLUMNS) + ["faith_parse_failed"]
        if self.segment_level:
            out_cols.append("faith_segment_scores")
        return ["data"], out_cols

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        """Initialize the LLM client and load prompt templates.

        Prompt YAML loading and default generation config are deferred here
        (instead of ``__post_init__``) for Ray compatibility: ``__post_init__``
        runs on the driver, while ``setup()`` runs on the worker.
        """
        if not self._initialized:
            self._system_prompt, self._user_template = load_prompt_template("faith_eval.yaml")

            if self.generation_config is None:
                self.generation_config = GenerationConfig(
                    temperature=0.0,
                    max_tokens=256,
                )

            if self.client is not None:
                self.client.setup()

            self._initialized = True

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Score each translation and filter rows below threshold.

        When ``segment_level=True`` and a ``_seg_translation_pairs`` column exists,
        each document's segment pairs are individually scored by the LLM and the
        per-document result is the average across segments.  Otherwise, whole-document
        scoring is used.

        When ``filter_enabled=False``, all rows are kept regardless of their
        ``faith_avg`` score.
        """
        df = batch.to_pandas().copy()

        if df.empty:
            for col in _SCORE_COLUMNS:
                df[col] = pd.Series(dtype="float64")
            return DocumentBatch(
                task_id=batch.task_id,
                dataset_name=batch.dataset_name,
                data=df,
                _metadata=batch._metadata,
                _stage_perf=batch._stage_perf,
            )

        num_docs = len(df)
        logger.info("FaithEvalFilter: evaluating {} documents", num_docs)

        all_scores, segment_scores_json, parse_failed_flags = self._score_batch(df)
        self._attach_score_columns(df, all_scores, segment_scores_json, parse_failed_flags)
        self._log_batch_scores(df)
        df = self._filter_rows(df)

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _score_batch(
        self,
        df: pd.DataFrame,
    ) -> tuple[list[dict], list[str] | None, list[bool]]:
        """Run either document-level or segment-level FAITH scoring."""
        if self.segment_level and "_seg_translation_pairs" in df.columns:
            return self._score_segments(df)

        if self.segment_level:
            logger.info(
                "segment_level=True but '_seg_translation_pairs' column not found; "
                "falling back to whole-document scoring"
            )

        responses = self._score_all(df)
        parsed = [self._extract_scores_from_json(response) for response in responses]
        return [scores for scores, _ in parsed], None, [failed for _, failed in parsed]

    def _attach_score_columns(
        self,
        df: pd.DataFrame,
        all_scores: list[dict],
        segment_scores_json: list[str] | None,
        parse_failed_flags: list[bool],
    ) -> None:
        """Write parsed FAITH scores back onto the DataFrame."""
        if segment_scores_json is not None:
            df["faith_segment_scores"] = segment_scores_json

        df["faith_fluency"] = [scores["Fluency"] for scores in all_scores]
        df["faith_accuracy"] = [scores["Accuracy"] for scores in all_scores]
        df["faith_idiomaticity"] = [scores["Idiomaticity"] for scores in all_scores]
        df["faith_terminology"] = [scores["Terminology"] for scores in all_scores]
        df["faith_handling_of_format"] = [scores["Handling_of_Format"] for scores in all_scores]
        df["faith_avg"] = [self._compute_faith_avg(scores) for scores in all_scores]
        df["faith_parse_failed"] = parse_failed_flags

    def _log_batch_scores(self, df: pd.DataFrame) -> None:
        """Log aggregate FAITH scores and parse-failure counts."""
        avg_batch_scores = {col: round(df[col].mean(), 3) for col in _SCORE_COLUMNS}
        logger.info("FaithEvalFilter: average batch scores: {}", avg_batch_scores)

        parse_failure_count = int(df["faith_parse_failed"].sum())
        if parse_failure_count:
            logger.warning(
                "FaithEvalFilter: {} of {} responses failed JSON parsing; "
                "these rows are preserved (not filtered as 'low quality')",
                parse_failure_count,
                len(df),
            )

    def _filter_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply threshold filtering while preserving parse-failed rows."""
        if not self.filter_enabled:
            logger.info("FaithEvalFilter: filter_enabled=False, keeping all {} documents", len(df))
            return df

        pre_filter_count = len(df)
        keep_mask = (df["faith_avg"] >= self.threshold) | df["faith_parse_failed"]
        filtered_df = df[keep_mask].reset_index(drop=True)
        num_filtered = pre_filter_count - len(filtered_df)
        logger.info(
            "FaithEvalFilter: filtered {}/{} documents below threshold {}",
            num_filtered,
            pre_filter_count,
            self.threshold,
        )
        return filtered_df

    # ------------------------------------------------------------------
    # Per-segment scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_faith_avg(scores: dict) -> float:
        """Compute ``faith_avg`` as the mean of non-zero per-dimension scores.

        Follows the "zero means not applicable" convention: dimensions
        scored as ``0.0`` are excluded from the average.  If every
        dimension is zero, returns ``0.0``.

        Single source of truth for both doc-level and segment-level paths
        so the semantics stay in lockstep.

        Parameters
        ----------
        scores : dict
            Dict keyed by :data:`FAITH_KEYS` (missing keys treated as 0).
        """
        values = [float(scores.get(k, 0.0)) for k in FAITH_KEYS]
        non_zero = [v for v in values if v > 0]
        if not non_zero:
            return 0.0
        return float(sum(non_zero) / len(non_zero))

    @staticmethod
    def _average_scores(segment_scores: list[dict]) -> dict:
        """Average FAITH scores across segments.

        Only positive scores (> 0) contribute to the average so that
        segments where a dimension was not applicable (scored 0) do not
        drag down the mean.

        Parameters
        ----------
        segment_scores : list[dict]
            Per-segment score dictionaries with FAITH_KEYS.

        Returns
        -------
        dict
            Averaged score dictionary with the same FAITH_KEYS, rounded to
            2 decimal places.
        """
        if not segment_scores:
            return {k: 0.0 for k in FAITH_KEYS}
        averaged: dict[str, float] = {}
        for key in FAITH_KEYS:
            values = [s.get(key, 0.0) for s in segment_scores if s.get(key, 0.0) > 0]
            averaged[key] = round(sum(values) / len(values), 2) if values else 0.0
        return averaged

    def _score_segments(self, df: pd.DataFrame) -> tuple[list[dict], list[str], list[bool]]:
        """Score each document by evaluating its individual segment pairs.

        Reads ``_seg_translation_pairs`` (JSON-serialized list of
        ``{"src": ..., "tgt": ...}`` dicts) from each row, issues one LLM
        request per segment, and averages the scores per document.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a ``_seg_translation_pairs`` column.

        Returns
        -------
        tuple[list[dict], list[str], list[bool]]
            ``(doc_scores, segment_scores_json, parse_failed_flags)`` where
            *doc_scores* is a list of averaged score dicts (one per row),
            *segment_scores_json* is a list of JSON-serialized per-segment
            score lists (one per row), and *parse_failed_flags* marks rows
            where at least one segment response failed to parse as JSON.
        """
        all_pairs, doc_segment_counts = self._collect_segment_pairs(df)

        total_segments = len(all_pairs)
        logger.info(
            "FaithEvalFilter: segment_level scoring -- {} total segments across {} documents",
            total_segments,
            len(df),
        )

        # Score all segments in bulk
        all_segment_scores: list[dict] = []
        all_segment_parse_failed: list[bool] = []
        if total_segments > 0:
            segment_df = pd.DataFrame(
                {self.source_text_field: [p[0] for p in all_pairs],
                 self.translated_text_field: [p[1] for p in all_pairs]},
            )
            responses = self._score_all(segment_df)
            for r in responses:
                scores, failed = self._extract_scores_from_json(r)
                all_segment_scores.append(scores)
                all_segment_parse_failed.append(failed)

        return self._rebuild_document_scores(
            all_segment_scores,
            all_segment_parse_failed,
            doc_segment_counts,
        )

    @staticmethod
    def _collect_segment_pairs(df: pd.DataFrame) -> tuple[list[tuple[str, str]], list[int]]:
        """Flatten ``_seg_translation_pairs`` into a batch-scoring worklist."""
        all_pairs: list[tuple[str, str]] = []
        doc_segment_counts: list[int] = []

        for _, row in df.iterrows():
            raw = row["_seg_translation_pairs"]
            if isinstance(raw, str):
                raw_stripped = raw.strip()
                pairs: list[dict] = json.loads(raw_stripped) if raw_stripped else []
            else:
                pairs = raw or []

            doc_segment_counts.append(len(pairs))
            all_pairs.extend((pair["src"], pair["tgt"]) for pair in pairs)

        return all_pairs, doc_segment_counts

    def _rebuild_document_scores(
        self,
        all_segment_scores: list[dict],
        all_segment_parse_failed: list[bool],
        doc_segment_counts: list[int],
    ) -> tuple[list[dict], list[str], list[bool]]:
        """Fold bulk segment scores back into per-document outputs."""
        doc_scores: list[dict] = []
        segment_scores_json: list[str] = []
        parse_failed_flags: list[bool] = []
        offset = 0

        for count in doc_segment_counts:
            seg_scores = all_segment_scores[offset : offset + count]
            seg_failed = all_segment_parse_failed[offset : offset + count]
            offset += count

            doc_scores.append(self._average_scores(seg_scores))
            segment_scores_json.append(json.dumps(seg_scores, ensure_ascii=False))
            parse_failed_flags.append(any(seg_failed))

        return doc_scores, segment_scores_json, parse_failed_flags

    # ------------------------------------------------------------------
    # LLM interaction helpers
    # ------------------------------------------------------------------

    def _build_messages(self, source_text: str, translated_text: str) -> list[dict]:
        """Build the chat messages for a single FAITH evaluation request."""
        source_language = get_language_name(self.source_lang)
        target_language = get_language_name(self.target_lang)
        return [
            {
                "role": "system",
                "content": self._system_prompt.format(
                    source_language=source_language,
                    target_language=target_language,
                ),
            },
            {
                "role": "user",
                "content": self._user_template.format(
                    source_language=source_language,
                    target_language=target_language,
                    source_text=source_text,
                    translated_text=translated_text,
                ),
            },
        ]

    def _score_all(self, df: pd.DataFrame) -> list[str]:
        """Score all rows using the async LLM client.

        Handles event-loop edge cases (e.g. being called from within an
        existing async context such as a Ray async actor).
        """
        return run_async_safe(lambda: self._score_all_async(df))

    async def _score_all_async(self, df: pd.DataFrame) -> list[str]:
        """Issue concurrent LLM requests for every row.

        Uses ``return_exceptions=True`` so that individual scoring failures
        do not abort the entire batch.  Failed rows receive an empty string
        response, and the error is logged.
        """
        sem = asyncio.Semaphore(self.max_concurrent_requests)

        async def _score_one(source_text: str, translated_text: str) -> str:
            messages = self._build_messages(source_text, translated_text)
            response = await self.client.query_model(
                model=self.model_name,
                messages=messages,
                generation_config=self.generation_config,
            )
            return response[0] if response else ""

        async def _score_one_throttled(source_text: str, translated_text: str) -> str:
            async with sem:
                return await _score_one(source_text, translated_text)

        tasks = [
            _score_one_throttled(row[self.source_text_field], row[self.translated_text_field])
            for _, row in df.iterrows()
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[str] = []
        for idx, result in enumerate(raw_results):
            if isinstance(result, BaseException):
                logger.error(
                    "FAITH scoring failed for row index {}: {}",
                    idx,
                    result,
                )
                results.append("")
            else:
                results.append(result)
        return results

    # ------------------------------------------------------------------
    # Score parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        """Find and return the first balanced ``{...}`` JSON object in *text*.

        Walks the string counting ``{``/``}`` pairs, respecting string
        literals so that braces inside quoted strings do not affect the
        balance *and* do not anchor the scan.  For example, in
        ``'message: "{pre}" scores: {"Fluency": 4}'`` the first ``{`` lives
        inside a string literal and must be ignored; the real object starts
        at the second ``{``.

        Supports nested objects (e.g. ``{"scores": {"Fluency": 4, ...}}``).

        Returns:
            Substring from the first real ``{`` to its matching ``}``
            inclusive, or ``None`` if no balanced object can be found.
        """
        # First pass: find the first ``{`` that lies *outside* any string
        # literal.  Tracking ``in_string``/``escape`` from index 0 ensures
        # braces inside quoted strings do not anchor the scan.
        start = -1
        in_string = False
        escape = False
        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    start = i
                    break
        if start == -1:
            return None

        # Second pass: balanced brace walk from ``start``.
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        return None

    @classmethod
    def _extract_scores_from_json(cls, text: str) -> tuple[dict, bool]:
        """Extract FAITH scores from an LLM JSON response.

        Finds the first balanced ``{...}`` block in *text* (with support for
        nested objects), parses it as JSON, and normalises the keys to the
        five FAITH dimensions. Missing keys default to ``0.0``.

        A score of ``0.0`` follows the "zero means not applicable" convention
        (see :meth:`_average_scores`).

        Returns:
            Tuple of ``(scores, parse_failed)`` where ``scores`` is a dict
            keyed by :data:`FAITH_KEYS` (values float) and ``parse_failed``
            is ``True`` iff no JSON object could be located or it failed to
            parse / validate.
        """
        zero_scores = {k: 0.0 for k in FAITH_KEYS}
        candidate = cls._extract_json_object(text)
        if candidate is None:
            return zero_scores, True
        try:
            scores_dict = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return zero_scores, True
        if not isinstance(scores_dict, dict):
            return zero_scores, True
        normalized: dict[str, float] = {}
        for key in FAITH_KEYS:
            if key in scores_dict:
                try:
                    normalized[key] = float(scores_dict[key])
                except (TypeError, ValueError):
                    normalized[key] = 0.0
            else:
                normalized[key] = 0.0
        return normalized, False
