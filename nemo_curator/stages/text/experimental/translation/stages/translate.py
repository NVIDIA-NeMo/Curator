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

"""Translate segmented text with an LLM or external backend."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.experimental.translation.utils.async_utils import run_async_safe
from nemo_curator.stages.text.experimental.translation.utils.prompt_loader import (
    load_prompt_template,
)
from nemo_curator.stages.text.utils.text_utils import get_language_name
from nemo_curator.tasks import DocumentBatch

from .segmentation import is_line_translatable_content

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig

# ---------------------------------------------------------------------------
# SegmentTranslationStage
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class SegmentTranslationStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Translate segments emitted by :class:`SegmentationStage`.

    Reads ``_seg_segments`` and writes ``_translated``.
    """

    name: str = "SegmentTranslationStage"
    source_lang: str
    target_lang: str
    client: AsyncLLMClient | None = None
    model_name: str = ""
    backend_type: str = "llm"
    backend_config: dict = field(default_factory=dict)
    generation_config: GenerationConfig | None = None
    prompt_path: str | None = None
    """Absolute local YAML prompt path, or ``None`` for the packaged prompt."""
    max_concurrent_requests: int = 64
    health_check: bool = True
    """If True, verify the translation backend is reachable during ``setup()``."""
    dry_run: bool = False
    """If True, skip actual translation and return empty strings."""
    dry_run_log_count: int = 5
    """Number of example prompts to log when *dry_run* is enabled."""

    _system_prompt: str = field(init=False, repr=False, default="")
    _user_template: str = field(init=False, repr=False, default="")
    _backend: object = field(init=False, repr=False, default=None)  # TranslationBackend or None
    _initialized: bool = field(init=False, repr=False, default=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self.source_lang = self.source_lang.strip()
        self.target_lang = self.target_lang.strip()
        self.model_name = self.model_name.strip()
        if not self.source_lang:
            msg = "SegmentTranslationStage requires a non-empty 'source_lang'"
            raise ValueError(msg)
        if not self.target_lang:
            msg = "SegmentTranslationStage requires a non-empty 'target_lang'"
            raise ValueError(msg)
        if self.backend_type == "llm":
            if self.client is None:
                msg = "SegmentTranslationStage requires a non-None 'client' (AsyncLLMClient) when backend_type='llm'"
                raise ValueError(msg)
            if not self.model_name:
                msg = "SegmentTranslationStage requires a non-empty 'model_name' when backend_type='llm'"
                raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["_seg_segments"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["_translated", "_translation_time", "_translation_error"]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        """Initialize the client or backend on the worker."""
        if not self._initialized:
            prompt_file = self.prompt_path or "translate.yaml"
            self._system_prompt, self._user_template = load_prompt_template(prompt_file)

            if self.backend_type != "llm":
                from nemo_curator.stages.text.experimental.translation.backends import get_backend

                self._backend = get_backend(self.backend_type, self.backend_config)

            if self.backend_type == "llm":
                if self.client is not None:
                    self.client.setup()
            elif self._backend is not None:
                self._backend.setup()

            if self.health_check:
                self._run_health_check()

            self._initialized = True

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def _run_health_check(self) -> None:
        """Verify the translation backend is reachable."""
        if self.backend_type == "llm":
            try:
                result = run_async_safe(self._query_llm_health_check)
            except RuntimeError:
                raise
            except Exception as exc:
                msg = f"LLM health check failed: {exc}. Ensure the LLM server is running and reachable."
                raise RuntimeError(msg) from exc
            if not result:
                msg = "LLM health check returned empty response"
                raise RuntimeError(msg)
            logger.info("LLM health check passed")
        elif self._backend is not None and hasattr(self._backend, "check_server"):
            ok = self._backend.check_server()
            if not ok:
                msg = (
                    f"{self.backend_type!r} backend health check failed. "
                    "Ensure the translation service is running and reachable."
                )
                raise RuntimeError(msg)
        else:
            logger.debug(
                "Backend {!r} does not implement check_server(); skipping health check",
                self.backend_type,
            )

    async def _query_llm_health_check(self) -> str:
        """Run the lightweight LLM health-check request."""
        messages = self._build_messages("Hello")
        resp = await self.client.query_model(  # type: ignore[union-attr]
            model=self.model_name,
            messages=messages,
            generation_config=self.generation_config,
        )
        return resp[0] if resp else ""

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Translate every segment in the batch."""
        df = batch.to_pandas().copy()
        segments: list[str] = df["_seg_segments"].tolist()

        if self.dry_run:
            n = min(self.dry_run_log_count, len(segments))
            for i in range(n):
                if self.backend_type == "llm":
                    messages = self._build_messages(segments[i])
                    logger.info("Dry-run prompt [{}]:\n{}", i, messages)
                else:
                    logger.info("Dry-run segment [{}]: {}", i, segments[i][:200])
            logger.info(
                "Dry run: skipping translation for {} segments",
                len(segments),
            )
            df["_translated"] = [""] * len(segments)
            df["_translation_time"] = [0.0] * len(segments)
            df["_translation_error"] = [""] * len(segments)
            return DocumentBatch(
                dataset_name=batch.dataset_name,
                data=df,
                _metadata=batch._metadata,
                _stage_perf=batch._stage_perf,
            )

        if self.backend_type == "llm":
            translated, timings, errors = self._translate_llm_async(segments)
        else:
            translated, timings, errors = self._translate_backend(segments)

        df["_translated"] = translated
        df["_translation_time"] = timings
        df["_translation_error"] = errors

        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    # ------------------------------------------------------------------
    # LLM translation helpers
    # ------------------------------------------------------------------

    def _translate_llm_async(self, segments: list[str]) -> tuple[list[str], list[float], list[str]]:
        """Translate segments with the async LLM client."""
        return run_async_safe(lambda: self._translate_all_async(segments))

    async def _translate_all_async(self, segments: list[str]) -> tuple[list[str], list[float], list[str]]:
        """Translate all segments concurrently."""
        sem = asyncio.Semaphore(self.max_concurrent_requests)

        async def _translate_one(seg: str) -> tuple[str, float]:
            if not seg or not seg.strip():
                return ("", 0.0)
            if not is_line_translatable_content(seg):
                return (seg, 0.0)
            messages = self._build_messages(seg)
            start = time.time()
            response = await self.client.query_model(  # type: ignore[union-attr]
                model=self.model_name,
                messages=messages,
                generation_config=self.generation_config,
            )
            elapsed = time.time() - start
            return (self._unwrap_translation(response[0] if response else ""), elapsed)

        async def _translate_one_throttled(seg: str) -> tuple[str, float]:
            async with sem:
                return await _translate_one(seg)

        tasks = [_translate_one_throttled(seg) for seg in segments]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        translated: list[str] = []
        timings: list[float] = []
        errors: list[str] = []

        for idx, result in enumerate(raw_results):
            if isinstance(result, BaseException):
                logger.error(
                    "Translation failed for segment index {}: {}",
                    idx,
                    result,
                )
                translated.append("")
                timings.append(0.0)
                errors.append(str(result))
            else:
                text, elapsed = result
                translated.append(text)
                timings.append(elapsed)
                errors.append("")

        return translated, timings, errors

    # ------------------------------------------------------------------
    # Non-LLM backend delegation
    # ------------------------------------------------------------------

    def _translate_backend(self, segments: list[str]) -> tuple[list[str], list[float], list[str]]:
        """Delegate translation to a non-LLM backend."""
        if self._backend is None:
            msg = f"Backend '{self.backend_type}' was not initialized"
            raise RuntimeError(msg)

        translated = [""] * len(segments)
        timings = [0.0] * len(segments)
        errors = [""] * len(segments)

        translate_indices, translate_segments = self._collect_backend_segments(segments, translated)

        if not translate_segments:
            return translated, timings, errors

        try:
            start = time.time()
            result = self._call_backend_batch(translate_segments)
            elapsed = time.time() - start

            self._validate_backend_result_count(result, translate_segments)
        except self._backend_failure_exceptions() as exc:
            logger.warning(
                "Bulk backend translation failed for {} segments: {}. Falling back to per-segment requests.",
                len(translate_segments),
                exc,
            )
        else:
            self._write_bulk_backend_results(
                result=result,
                elapsed=elapsed,
                translate_indices=translate_indices,
                translated=translated,
                timings=timings,
            )
            return translated, timings, errors

        self._translate_backend_one_by_one(segments, translated, timings, errors)
        return translated, timings, errors

    @staticmethod
    def _collect_backend_segments(segments: list[str], translated: list[str]) -> tuple[list[int], list[str]]:
        """Collect translatable segments and preserve passthrough segments."""
        translate_indices: list[int] = []
        translate_segments: list[str] = []
        for idx, seg in enumerate(segments):
            if not seg or not seg.strip():
                continue
            if not is_line_translatable_content(seg):
                translated[idx] = seg
                continue
            translate_indices.append(idx)
            translate_segments.append(seg)
        return translate_indices, translate_segments

    @staticmethod
    def _validate_backend_result_count(result: list[str], translate_segments: list[str]) -> None:
        """Raise if the backend returned a different number of translations."""
        if len(result) != len(translate_segments):
            msg = f"Backend returned {len(result)} translations for {len(translate_segments)} segments"
            raise RuntimeError(msg)

    @staticmethod
    def _write_bulk_backend_results(
        *,
        result: list[str],
        elapsed: float,
        translate_indices: list[int],
        translated: list[str],
        timings: list[float],
    ) -> None:
        """Write successful bulk backend outputs into result arrays."""
        per_segment_time = elapsed / len(translate_indices)
        for out_idx, translated_text in zip(translate_indices, result, strict=False):
            translated[out_idx] = translated_text
            timings[out_idx] = per_segment_time

    def _translate_backend_one_by_one(
        self,
        segments: list[str],
        translated: list[str],
        timings: list[float],
        errors: list[str],
    ) -> None:
        """Fallback path that retries backend translation one segment at a time."""
        for idx, seg in enumerate(segments):
            if not seg or not seg.strip():
                continue
            if not is_line_translatable_content(seg):
                translated[idx] = seg
                continue
            start = time.time()
            try:
                result = self._call_backend_batch([seg])
                elapsed = time.time() - start
                translated[idx] = result[0] if result else ""
                timings[idx] = elapsed
                errors[idx] = ""
            except self._backend_failure_exceptions() as exc:
                elapsed = time.time() - start
                logger.error(
                    "Backend translation failed for segment index {}: {}",
                    idx,
                    exc,
                )
                translated[idx] = ""
                timings[idx] = elapsed
                errors[idx] = str(exc)

    def _backend_failure_exceptions(self) -> tuple[type[BaseException], ...]:
        """Return exception types handled at the backend boundary."""
        return (Exception,)

    def _call_backend_batch(self, segments: list[str]) -> list[str]:
        """Invoke the configured non-LLM backend for one batch of segments."""
        if hasattr(self._backend, "translate_batch_async"):
            backend = self._backend
            return run_async_safe(
                lambda: backend.translate_batch_async(
                    segments,
                    self.source_lang,
                    self.target_lang,
                )
            )
        return self._backend.translate_batch(  # type: ignore[union-attr]
            segments,
            self.source_lang,
            self.target_lang,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_messages(self, segment: str) -> list[dict]:
        """Build the prompt for one segment."""
        source_lang_name = get_language_name(self.source_lang)
        target_lang_name = get_language_name(self.target_lang)
        return [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": self._user_template.format(
                    source_lang=source_lang_name,
                    target_lang=target_lang_name,
                    src=segment,
                ),
            },
        ]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_translation(text: str) -> str:
        """Extract translated text from the expected ``〘...〙`` wrapper."""
        left_loc = text.rfind("\u3018")  # 〘
        right_loc = text.rfind("\u3019")  # 〙
        if left_loc != -1 and right_loc != -1 and left_loc < right_loc:
            return text[left_loc + 1 : right_loc]
        elif left_loc != -1:
            return text[left_loc + 1 :]
        return text
