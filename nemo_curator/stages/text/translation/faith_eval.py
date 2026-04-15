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
FaithEvalFilter -- LLM-based translation quality scoring using the FAITH metric.

Scores translations on 5 dimensions (Fluency, Accuracy, Idiomaticity,
Terminology, Handling_of_Format) and filters rows below a configurable
average threshold.

Ported from Speaker's faith_eval.py with Curator-native interfaces.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml
from loguru import logger

from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

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


def _load_prompt_template() -> tuple[str, str]:
    """Load the FAITH evaluation prompt template from the bundled YAML file.

    Returns:
        Tuple of (system_prompt, user_template) strings.
    """
    prompt_path = Path(__file__).parent / "prompts" / "faith_eval.yaml"
    with open(prompt_path, encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    return prompts["system"], prompts["user"]


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
    prompt_template : str
        Unused reserved field (prompts are loaded from YAML).
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
    prompt_template: str = ""
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
        return ["data"], list(_SCORE_COLUMNS)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        """Initialize the LLM client and load prompt templates.

        Prompt YAML loading and default generation config are deferred here
        (instead of ``__post_init__``) for Ray compatibility: ``__post_init__``
        runs on the driver, while ``setup()`` runs on the worker.
        """
        if not self._initialized:
            # Load prompt templates from YAML on the worker
            self._system_prompt, self._user_template = _load_prompt_template()

            # Default generation config for deterministic scoring
            if self.generation_config is None:
                self.generation_config = GenerationConfig(temperature=0.0, max_tokens=256)

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

        use_segment_scoring = (
            self.segment_level and "_seg_translation_pairs" in df.columns
        )

        if use_segment_scoring:
            all_scores, segment_scores_json = self._score_segments(df)
            df["faith_segment_scores"] = segment_scores_json
        else:
            if self.segment_level:
                logger.info(
                    "segment_level=True but '_seg_translation_pairs' column not found; "
                    "falling back to whole-document scoring"
                )
            # Whole-document scoring (original behaviour)
            responses = self._score_all(df)
            all_scores = [self._extract_scores_from_json(r) for r in responses]

        # Attach per-dimension score columns
        df["faith_fluency"] = [s["Fluency"] for s in all_scores]
        df["faith_accuracy"] = [s["Accuracy"] for s in all_scores]
        df["faith_idiomaticity"] = [s["Idiomaticity"] for s in all_scores]
        df["faith_terminology"] = [s["Terminology"] for s in all_scores]
        df["faith_handling_of_format"] = [s["Handling_of_Format"] for s in all_scores]
        df["faith_avg"] = df[
            ["faith_fluency", "faith_accuracy", "faith_idiomaticity", "faith_terminology", "faith_handling_of_format"]
        ].mean(axis=1)

        # Log average scores across the batch
        avg_batch_scores = {col: round(df[col].mean(), 3) for col in _SCORE_COLUMNS}
        logger.info("FaithEvalFilter: average batch scores: {}", avg_batch_scores)

        # Filter rows below threshold (when enabled)
        if self.filter_enabled:
            pre_filter_count = len(df)
            df = df[df["faith_avg"] >= self.threshold].reset_index(drop=True)
            num_filtered = pre_filter_count - len(df)
            logger.info(
                "FaithEvalFilter: filtered {}/{} documents below threshold {}",
                num_filtered,
                pre_filter_count,
                self.threshold,
            )
        else:
            logger.info("FaithEvalFilter: filter_enabled=False, keeping all {} documents", len(df))

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    # ------------------------------------------------------------------
    # Per-segment scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _average_scores(segment_scores: list[dict]) -> dict:
        """Average FAITH scores across segments.

        Only positive scores (> 0) contribute to the average so that
        segments where a dimension was not applicable (scored 0) do not
        drag down the mean.

        Ported from ``speaker/src/speaker/core/translate/faith_eval.py``
        ``FaithEvaluationTask.average_scores``.

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

    def _score_segments(self, df: pd.DataFrame) -> tuple[list[dict], list[str]]:
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
        tuple[list[dict], list[str]]
            ``(doc_scores, segment_scores_json)`` where *doc_scores* is a
            list of averaged score dicts (one per row) and
            *segment_scores_json* is a list of JSON-serialized per-segment
            score lists (one per row).
        """
        # Flatten all segment pairs across all documents for bulk LLM scoring
        all_pairs: list[tuple[str, str]] = []      # (src, tgt)
        doc_segment_counts: list[int] = []          # number of segments per doc

        for _, row in df.iterrows():
            raw = row["_seg_translation_pairs"]
            pairs: list[dict] = json.loads(raw) if isinstance(raw, str) else raw
            doc_segment_counts.append(len(pairs))
            for pair in pairs:
                all_pairs.append((pair["src"], pair["tgt"]))

        total_segments = len(all_pairs)
        logger.info(
            "FaithEvalFilter: segment_level scoring -- {} total segments across {} documents",
            total_segments,
            len(df),
        )

        # Score all segments in bulk
        if total_segments > 0:
            segment_df = pd.DataFrame(
                {self.source_text_field: [p[0] for p in all_pairs],
                 self.translated_text_field: [p[1] for p in all_pairs]},
            )
            responses = self._score_all(segment_df)
            all_segment_scores = [self._extract_scores_from_json(r) for r in responses]
        else:
            all_segment_scores = []

        # Split back per-document and average
        doc_scores: list[dict] = []
        segment_scores_json: list[str] = []
        offset = 0
        for count in doc_segment_counts:
            seg_scores = all_segment_scores[offset: offset + count]
            offset += count
            averaged = self._average_scores(seg_scores)
            doc_scores.append(averaged)
            segment_scores_json.append(json.dumps(seg_scores, ensure_ascii=False))

        return doc_scores, segment_scores_json

    # ------------------------------------------------------------------
    # LLM interaction helpers
    # ------------------------------------------------------------------

    def _get_language_name(self, lang_code: str) -> str:
        """Convert ISO 639-1 code to full language name.

        Falls back to the raw code if iso639 is not available.
        """
        try:
            import iso639

            return iso639.Lang(lang_code).name
        except Exception:
            return lang_code

    def _build_messages(self, source_text: str, translated_text: str) -> list[dict]:
        """Build the chat messages for a single FAITH evaluation request."""
        source_language = self._get_language_name(self.source_lang)
        target_language = self._get_language_name(self.target_lang)
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
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop running -- normal path
            return asyncio.run(self._score_all_async(df))

        # Already inside an event loop -- offload to a thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, self._score_all_async(df))
            return future.result()

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
                    "FAITH scoring failed for row index %d: %s",
                    idx,
                    result,
                )
                results.append("")
            else:
                results.append(result)
        return results

    # ------------------------------------------------------------------
    # Score parsing (ported from Speaker faith_eval.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_scores_from_json(text: str) -> dict:
        """Extract FAITH scores from an LLM JSON response.

        Searches for the first ``{...}`` block in *text*, parses it as JSON,
        and normalises the keys to the five FAITH dimensions. Missing keys
        default to ``0.0``.

        Ported from ``speaker/src/speaker/core/translate/faith_eval.py``
        ``FaithEvaluationTask.extract_scores_from_json``.
        """
        json_match = re.search(r"\{[^}]*\}", text, re.DOTALL)
        if json_match:
            try:
                scores_dict = json.loads(json_match.group(0))
                normalized: dict[str, float] = {}
                for key in FAITH_KEYS:
                    if key in scores_dict:
                        normalized[key] = float(scores_dict[key])
                    else:
                        normalized[key] = 0.0
                return normalized
            except (json.JSONDecodeError, ValueError):
                return {k: 0.0 for k in FAITH_KEYS}
        return {k: 0.0 for k in FAITH_KEYS}
