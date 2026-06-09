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

"""Remote-server variant of :class:`ContextualASRExtractionStage`.

Sends OpenAI-compatible chat requests to a shared inference server
instead of loading an in-process vLLM engine. Subclasses the in-process
stage to reuse its field set, prompt resolution (``_resolve_prompts``),
and the module-level ``_parse_json_response`` / ``_normalize_extraction``
helpers. It NEVER imports vLLM or loads a tokenizer.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.audio.text_filtering.contextual_asr_extraction import (
    ContextualASRExtractionStage,
    _normalize_extraction,
    _parse_json_response,
)
from nemo_curator.stages.resources import Resources

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.tasks import AudioTask


@dataclass
class RemoteContextualASRExtractionStage(ContextualASRExtractionStage):
    """:class:`ContextualASRExtractionStage` against a remote server.

    Adds the same remote-client fields as
    :class:`~nemo_curator.stages.audio.text_filtering.remote_text_llm_stage.RemoteTextLLMStage`.
    Inherited GPU-engine fields (``max_model_len`` etc.) are ignored — the
    server owns the engine.
    """

    inference_base_url: str = ""
    inference_api_key: str = "EMPTY"
    served_model_name: str | None = None
    max_concurrent_requests: int = 64
    request_timeout: int = 120

    _client: Any = field(default=None, init=False, repr=False)
    _gen_config: Any = field(default=None, init=False, repr=False)
    _loop_runner: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # CPU-only: the GPUs belong to the server (see RemoteTextLLMStage).
        self.resources = Resources(cpus=8.0)

    # ── Lifecycle ────────────────────────────────────────────────────

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._client is not None:
            return
        if not self.inference_base_url:
            msg = "RemoteContextualASRExtractionStage requires inference_base_url"
            raise ValueError(msg)

        from nemo_curator.models.client.async_runner import PersistentEventLoop
        from nemo_curator.models.client.openai_client import AsyncOpenAIClient

        self._system_prompt, self._user_prompt_template = self._resolve_prompts()  # inherited
        self._client = AsyncOpenAIClient(
            max_concurrent_requests=self.max_concurrent_requests,
            base_url=self.inference_base_url,
            api_key=self.inference_api_key,
            timeout=self.request_timeout,
        )
        self._client.setup()
        # One event loop for this actor's lifetime so the async client (and its
        # connection pool) stays bound to a single, always-running loop. Driving
        # it via a fresh asyncio.run() per batch can wedge on a primitive bound
        # to a since-closed loop — a silent, timeout-immune hang.
        self._loop_runner = PersistentEventLoop(name=self.name)
        self._loop_runner.start()
        # Matches the local SamplingParams(temperature=0.1, top_p=0.95).
        self._gen_config = GenerationConfig(
            temperature=0.1,
            top_p=0.95,
            max_tokens=self.max_output_tokens,
            extra_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        )
        logger.info(
            "{}: ready (remote={}, model={}, output_key={})",
            self.name,
            self.inference_base_url,
            self.served_model_name or self.model_id,
            self.output_key,
        )

    def teardown(self) -> None:
        super().teardown()
        if self._loop_runner is not None:
            if self._client is not None:
                with contextlib.suppress(Exception):
                    self._loop_runner.run(self._client.client.close(), timeout=30)
            self._loop_runner.close()
            self._loop_runner = None
        self._client = None

    # ── Prompt / inference ───────────────────────────────────────────

    def _build_messages(self, transcript: str, source_lang: str) -> list[dict]:
        """Build chat messages; mirrors the base ``_format_prompt`` fill."""
        fmt: dict[str, str] = {}
        if "{transcript}" in self._user_prompt_template:
            fmt["transcript"] = transcript
        if "{source_lang}" in self._user_prompt_template:
            fmt["source_lang"] = source_lang
        user_content = self._user_prompt_template.format(**fmt) if fmt else self._user_prompt_template
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _batch_timeout(self, n_requests: int) -> float:
        """Generous wall-clock cap for a whole batch, so a real hang surfaces.

        Per request worst case is ``request_timeout * (max_retries + 1)``
        (retries live in :class:`AsyncLLMClient`); the semaphore lets
        ``max_concurrent_requests`` run at once, so the batch takes
        ``ceil(n / max_concurrent_requests)`` waves. A fixed buffer covers retry
        backoff and scheduling slack.
        """
        max_retries = getattr(self._client, "max_retries", 3)
        per_request = self.request_timeout * (max_retries + 1)
        concurrency = max(1, self.max_concurrent_requests)
        waves = max(1, math.ceil(n_requests / concurrency))
        return per_request * waves + 120

    def _generate_remote(self, messages_list: list[list[dict]]) -> list[str]:
        """Fan out one chat request per message list, preserving order."""
        model = self.served_model_name or self.model_id

        async def _one(messages: list[dict]) -> str:
            resp = await self._client.query_model(
                messages=messages,
                model=model,
                generation_config=self._gen_config,
            )
            return (resp[0] if resp else "").strip()

        async def _all() -> list[str]:
            return await asyncio.gather(*[_one(m) for m in messages_list])

        # Drive every batch on the actor's single persistent event loop (bound
        # in setup()), with a wall-clock cap so a stuck request raises a loud
        # TimeoutError instead of wedging the actor silently.
        return self._loop_runner.run(_all(), timeout=self._batch_timeout(len(messages_list)))

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        if self._client is None:
            msg = "Client not initialised — setup() was not called"
            raise RuntimeError(msg)

        valid_indices: list[int] = []
        messages_list: list[list[dict]] = []

        for i, task in enumerate(tasks):
            text = task.data.get(self.text_key, "")
            skip = task.data.get(self.skip_me_key, "")
            if skip:
                task.data[self.output_key] = None
                set_note(task.data, self.name, "skipped (flagged)", self.notes_key)
                continue
            if not text or not text.strip():
                task.data[self.output_key] = None
                set_note(task.data, self.name, "skipped (empty)", self.notes_key)
                continue
            source_lang = task.data.get(self.source_lang_key) or self.default_source_lang
            valid_indices.append(i)
            messages_list.append(self._build_messages(text.strip(), str(source_lang)))

        if messages_list:
            results = self._generate_remote(messages_list)

            for seq_idx, task_idx in enumerate(valid_indices):
                task = tasks[task_idx]
                parsed = _parse_json_response(results[seq_idx])

                self._n_processed += 1
                if parsed is None:
                    task.data[self.output_key] = None
                    set_note(task.data, self.name, "json_parse_failed", self.notes_key)
                    self._n_failed += 1
                else:
                    task.data[self.output_key] = _normalize_extraction(parsed)
                    set_note(task.data, self.name, "extracted", self.notes_key)

        logger.debug(
            "{}: batch of {} tasks ({} inferred, {} failed total)",
            self.name,
            len(tasks),
            len(messages_list),
            self._n_failed,
        )
        return tasks
