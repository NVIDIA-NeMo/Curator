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

"""
TranslateStage -- translates text segments using an LLM backend (AsyncLLMClient)
or a pluggable non-LLM backend (Google, AWS, NMT).

The stage reads the ``_seg_segments`` column produced by :class:`SegmentationStage`
and writes a ``_translated`` column consumed by :class:`ReassemblyStage`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from dataclasses import dataclass, field
from pathlib import Path

import iso639
import pandas as pd
import yaml
from loguru import logger

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

# ---------------------------------------------------------------------------
# Prompt template loading
# ---------------------------------------------------------------------------

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt_template(filename: str) -> tuple[str, str]:
    """Load a YAML prompt file and return (system_prompt, user_template).

    Args:
        filename: Name of the YAML file inside the ``prompts/`` directory.

    Returns:
        Tuple of (system prompt string, user template string).
    """
    path = _PROMPT_DIR / filename
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data["system"], data["user"]


# ---------------------------------------------------------------------------
# TranslateStage
# ---------------------------------------------------------------------------


@dataclass
class TranslateStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Translate text segments produced by :class:`SegmentationStage`.

    For the **LLM backend** (default), the stage formats a translation prompt
    and sends it to an :class:`AsyncLLMClient`.  The LLM response is expected
    to wrap the translation in ``〘...〙`` brackets, which are stripped by
    :meth:`_unwrap_translation`.

    For **non-LLM backends** (``google``, ``aws``, ``nmt``), the stage
    delegates to a :class:`TranslationBackend` obtained via
    ``backends.get_backend()``.

    Note:
        The ``client`` parameter accepts ``AsyncLLMClient``. When used in a
        ``TranslationPipeline`` that includes ``FaithEvalFilter``, an
        ``AsyncLLMClient`` is required since FaithEvalFilter only supports
        async clients.
    """

    name: str = "TranslateStage"
    client: AsyncLLMClient | None = None
    model_name: str = ""
    source_lang: str = "en"
    target_lang: str = "zh"
    backend_type: str = "llm"
    backend_config: dict = field(default_factory=dict)
    generation_config: GenerationConfig | None = None
    prompt_template: str = ""  # Reserved for user override; loaded from YAML when empty.
    max_concurrent_requests: int = 64
    health_check: bool = True
    """If True, verify the translation backend is reachable during ``setup()``."""
    dry_run: bool = False
    """If True, skip actual translation and return empty strings."""
    dry_run_log_count: int = 5
    """Number of example prompts to log when *dry_run* is enabled."""
    show_progress: bool = False
    """If True, display a ``tqdm`` progress bar during async translation."""

    # -- internal state (not constructor args) ---------------------------------
    _system_prompt: str = field(init=False, repr=False, default="")
    _user_template: str = field(init=False, repr=False, default="")
    _backend: object = field(init=False, repr=False, default=None)  # TranslationBackend or None
    _initialized: bool = field(init=False, repr=False, default=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Validate basic config on the driver side (before Ray serialization).
        if self.backend_type == "llm":
            if self.client is None:
                msg = "TranslateStage requires a non-None 'client' (AsyncLLMClient) when backend_type='llm'"
                raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["_seg_segments"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["_translated", "_translation_time", "_translation_error"]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        """Initialize the LLM client or translation backend.

        Prompt YAML loading and backend instantiation are deferred here
        (instead of ``__post_init__``) for Ray compatibility: ``__post_init__``
        runs on the driver, while ``setup()`` runs on the worker.

        When ``health_check`` is enabled, the backend is verified before any
        translation work begins:
        - LLM backend: a small test translation is attempted.
        - Non-LLM backends: the backend's ``check_server()`` method is called.
        """
        if not self._initialized:
            # Load prompt template on the worker
            self._system_prompt, self._user_template = _load_prompt_template("translate.yaml")

            if self.backend_type != "llm":
                # Non-LLM backend -- import factory lazily so that backends/
                # can be developed independently.
                from nemo_curator.stages.text.translation.backends import get_backend

                self._backend = get_backend(self.backend_type, self.backend_config)

            # Initialize the client or backend once.
            if self.backend_type == "llm":
                if self.client is not None:
                    self.client.setup()
            elif self._backend is not None:
                self._backend.setup()

            # --- Gap 11.2: Example prompt logging ---
            if self.backend_type == "llm":
                example_messages = self._build_messages("Sample text to translate")
                logger.info("Example translation prompt:\n{}", example_messages)

            # --- Gap 10.1: Server health check ---
            if self.health_check:
                self._run_health_check()

            self._initialized = True

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def _run_health_check(self) -> None:
        """Verify the translation backend is reachable.

        For the LLM backend, attempts a small test translation to confirm
        the server is responding.  For non-LLM backends, calls the
        backend's ``check_server()`` method.

        Raises:
            RuntimeError: If the health check fails.
        """
        if self.backend_type == "llm":
            try:
                messages = self._build_messages("Hello")
                # Fire a lightweight test request.
                loop_running = True
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    loop_running = False

                async def _test_query() -> str:
                    resp = await self.client.query_model(  # type: ignore[union-attr]
                        model=self.model_name,
                        messages=messages,
                        generation_config=self.generation_config,
                    )
                    return resp[0] if resp else ""

                if loop_running:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        result = pool.submit(asyncio.run, _test_query()).result()
                else:
                    result = asyncio.run(_test_query())

                if result:
                    logger.info("LLM health check passed")
                else:
                    raise RuntimeError("LLM health check returned empty response")
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"LLM health check failed: {exc}. "
                    "Ensure the LLM server is running and reachable."
                ) from exc
        else:
            if self._backend is not None and hasattr(self._backend, "check_server"):
                ok = self._backend.check_server()
                if not ok:
                    raise RuntimeError(
                        f"{self.backend_type!r} backend health check failed. "
                        "Ensure the translation service is running and reachable."
                    )
            else:
                logger.debug(
                    "Backend %r does not implement check_server(); "
                    "skipping health check",
                    self.backend_type,
                )

    # ------------------------------------------------------------------
    # process()
    # ------------------------------------------------------------------

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Translate every segment in the batch.

        Reads the ``_seg_segments`` column, translates each value, and writes
        the results into a new ``_translated`` column.  Additionally populates
        ``_translation_time`` (per-segment elapsed seconds) and
        ``_translation_error`` (empty string on success, error message on
        failure).

        When ``dry_run`` is enabled, logs the first N prompts and returns
        immediately with empty translations.
        """
        df = batch.to_pandas().copy()
        segments: list[str] = df["_seg_segments"].tolist()

        # --- Gap 10.3: Dry run mode ---
        if self.dry_run:
            n = min(self.dry_run_log_count, len(segments))
            for i in range(n):
                if self.backend_type == "llm":
                    messages = self._build_messages(segments[i])
                    logger.info("Dry-run prompt [{}]:\n{}", i, messages)
                else:
                    logger.info("Dry-run segment [{}]: {}", i, segments[i][:200])
            logger.info(
                "Dry run: skipping translation for %d segments",
                len(segments),
            )
            df["_translated"] = [""] * len(segments)
            df["_translation_time"] = [0.0] * len(segments)
            df["_translation_error"] = [""] * len(segments)
            return DocumentBatch(
                task_id=batch.task_id,
                dataset_name=batch.dataset_name,
                data=df,
                _metadata=batch._metadata,
                _stage_perf=batch._stage_perf,
            )

        if self.backend_type == "llm":
            translated, timings, errors = self._translate_llm(segments)
        else:
            translated, timings, errors = self._translate_backend(segments)

        df["_translated"] = translated
        df["_translation_time"] = timings
        df["_translation_error"] = errors

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    # ------------------------------------------------------------------
    # LLM translation helpers
    # ------------------------------------------------------------------

    def _translate_llm(
        self, segments: list[str]
    ) -> tuple[list[str], list[float], list[str]]:
        """Translate segments via the AsyncLLMClient.

        Since ``client`` is always ``AsyncLLMClient``, this delegates
        directly to the async path.

        Returns:
            Tuple of (translated_texts, timing_seconds, error_messages).
        """
        return self._translate_llm_async(segments)

    def _translate_llm_async(
        self, segments: list[str]
    ) -> tuple[list[str], list[float], list[str]]:
        """Concurrent translation using an :class:`AsyncLLMClient`.

        Handles both cases:
        - Normal case: no event loop running -- use ``asyncio.run()``.
        - Edge case: called from within an async context (e.g. Ray async actors)
          -- run in a separate thread with its own loop.

        Returns:
            Tuple of (translated_texts, timing_seconds, error_messages).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop running -- the expected / normal case.
            return asyncio.run(self._translate_all_async(segments))

        # Already inside an event loop -- offload to a new thread.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, self._translate_all_async(segments))
            return future.result()

    async def _translate_all_async(
        self, segments: list[str]
    ) -> tuple[list[str], list[float], list[str]]:
        """Fire off concurrent translation requests for all segments.

        Uses ``return_exceptions=True`` so that individual segment failures
        do not abort the entire batch (Gap 10.2).  Failed segments receive
        an empty translation and the error message is recorded in the
        ``_translation_error`` column.

        When ``show_progress`` is enabled, a ``tqdm`` progress bar is updated
        after each segment completes via an asyncio callback.

        Returns:
            Tuple of (translated_texts, timing_seconds, error_messages).
        """
        sem = asyncio.Semaphore(self.max_concurrent_requests)

        # --- Gap 11.1: Progress bar with real-time updates ---
        pbar = None
        if self.show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(segments), desc="Translating segments")
            except ImportError:
                logger.warning("tqdm not installed; progress bar disabled")

        async def _translate_one(seg: str) -> tuple[str, float]:
            """Translate a single segment, returning (result, elapsed_seconds)."""
            # Skip empty/whitespace-only segments to avoid wasting LLM calls
            if not seg or not seg.strip():
                return ("", 0.0)
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
                result = await _translate_one(seg)
                if pbar is not None:
                    pbar.update(1)
                return result

        tasks = [_translate_one_throttled(seg) for seg in segments]

        # --- Gap 10.2: Partial translation recovery ---
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        if pbar is not None:
            pbar.close()

        translated: list[str] = []
        timings: list[float] = []
        errors: list[str] = []

        for idx, result in enumerate(raw_results):
            if isinstance(result, BaseException):
                logger.error(
                    "Translation failed for segment index %d: %s",
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

    def _translate_backend(
        self, segments: list[str]
    ) -> tuple[list[str], list[float], list[str]]:
        """Delegate translation to a non-LLM backend (Google, AWS, NMT).

        Wraps each segment with timing and error handling so the output
        format is consistent with the LLM path.

        Returns:
            Tuple of (translated_texts, timing_seconds, error_messages).
        """
        if self._backend is None:
            msg = f"Backend '{self.backend_type}' was not initialized"
            raise RuntimeError(msg)

        translated: list[str] = []
        timings: list[float] = []
        errors: list[str] = []

        # --- Gap 11.1: Progress bar for backend translation ---
        pbar = None
        if self.show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(segments), desc="Translating segments")
            except ImportError:
                logger.warning("tqdm not installed; progress bar disabled")

        for idx, seg in enumerate(segments):
            start = time.time()
            try:
                result = self._backend.translate_batch([seg], self.source_lang, self.target_lang)
                elapsed = time.time() - start
                translated.append(result[0] if result else "")
                timings.append(elapsed)
                errors.append("")
            except Exception as exc:
                elapsed = time.time() - start
                logger.error(
                    "Backend translation failed for segment index %d: %s",
                    idx,
                    exc,
                )
                translated.append("")
                timings.append(elapsed)
                errors.append(str(exc))
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        return translated, timings, errors

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_messages(self, segment: str) -> list[dict]:
        """Build the chat-style messages list for the LLM.

        Uses ``iso639`` to resolve language codes to human-readable names so
        the prompt reads naturally (e.g. "English" instead of "en").
        """
        source_lang_name = iso639.Lang(self.source_lang).name
        target_lang_name = iso639.Lang(self.target_lang).name
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
    # Response parsing (ported from Speaker)
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_translation(text: str) -> str:
        """Extract the translated text from ``〘...〙`` bracket delimiters.

        Ported from ``speaker.core.translate.translate_jsonl.unwrap_translation_results()``.

        Rules:
        - If both ``〘`` and ``〙`` are found (with left before right), return
          the text between them.
        - If only ``〘`` is found, return everything after it.
        - Otherwise return the original text unchanged.
        """
        left_loc = text.rfind("\u3018")  # 〘
        right_loc = text.rfind("\u3019")  # 〙
        if left_loc != -1 and right_loc != -1 and left_loc < right_loc:
            return text[left_loc + 1 : right_loc]
        elif left_loc != -1:
            return text[left_loc + 1 :]
        return text
