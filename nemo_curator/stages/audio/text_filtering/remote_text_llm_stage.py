# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Remote-server variant of :class:`TextLLMStage`.

Instead of loading an in-process vLLM engine, this stage sends
OpenAI-compatible chat requests to a shared inference server (e.g. a
``nemo_curator.core.serve.InferenceServer`` running Ray Serve + vLLM, or
any external OpenAI-compatible endpoint). Many such CPU-only client
stages can run concurrently against one GPU-backed server, avoiding the
per-stage engine loads and GPU time-sharing churn of the in-process path.

It subclasses :class:`TextLLMStage` purely to reuse the field set and the
prompt-resolution / validation helpers (``_resolve_prompt``, ``_validate``,
``_check_novel_words``, ``inputs``, ``outputs``). It NEVER imports vLLM or
loads a tokenizer — the server applies the chat template.
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
from nemo_curator.stages.audio.text_filtering.text_llm_stage import TextLLMStage
from nemo_curator.stages.resources import Resources

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.tasks import AudioTask


@dataclass
class RemoteTextLLMStage(TextLLMStage):
    """:class:`TextLLMStage` that calls a remote OpenAI-compatible server.

    Args (in addition to the inherited :class:`TextLLMStage` fields):
        inference_base_url: OpenAI-compatible base URL of the server,
            e.g. ``http://localhost:8000/v1``. Required.
        inference_api_key: API key sent with each request (servers that
            do not authenticate accept any non-empty value).
        served_model_name: Value passed as ``model=`` in each request.
            Defaults to ``model_id`` when unset.
        max_concurrent_requests: Max in-flight requests per stage actor
            (bounds the async client's semaphore).
        request_timeout: Per-request timeout in seconds.

    GPU-engine fields inherited from :class:`TextLLMStage`
    (``max_model_len``, ``kv_cache_dtype``, ``gpu_memory_utilization``,
    ``tensor_parallel_size``) are ignored here — those live on the server.
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
        # Remote mode holds no in-process engine. Force CPU-only so the
        # executor schedules these client actors OFF the GPUs the server
        # occupies (asking for a GPU here would deadlock waiting for one
        # the server never frees). 1 CPU is enough — actors only make HTTP
        # requests, no local GPU work.
        self.resources = Resources(cpus=1.0)

    # ── Lifecycle ────────────────────────────────────────────────────

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._client is not None:
            return
        if not self.inference_base_url:
            msg = "RemoteTextLLMStage requires inference_base_url"
            raise ValueError(msg)

        from nemo_curator.models.client.async_runner import PersistentEventLoop
        from nemo_curator.models.client.openai_client import AsyncOpenAIClient

        self._system_prompt = self._resolve_prompt()  # inherited
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
        # temperature 0 with top_p 1 gives greedy decoding, matching the local
        # in-process path. The chat-template flag is forwarded to vLLM through
        # the OpenAI SDK extra_body so the server disables thinking exactly as
        # the local apply_chat_template call does.
        self._gen_config = GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_output_tokens,
            seed=0,
            extra_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        )
        logger.info(
            "%s: ready (remote=%s, model=%s, output_key=%s)",
            self.name,
            self.inference_base_url,
            self.served_model_name or self.model_id,
            self.output_text_key,
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

    def _build_messages(self, user_text: str, task_data: dict | None = None) -> list[dict]:
        """Build the chat messages (server applies the template).

        Mirrors :meth:`TextLLMStage._format_prompt` message construction
        but returns the raw message list instead of a templated string.
        """
        prompt_template = self._system_prompt
        if "{language}" in prompt_template:
            lang = task_data.get("source_lang", "English") if task_data else "English"
            prompt_template = prompt_template.replace("{language}", lang)
        if "{text}" in prompt_template:
            prompt_template = prompt_template.replace("{text}", user_text)
            return [{"role": "user", "content": prompt_template}]
        return [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": user_text},
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
                task.data[self.output_text_key] = ""
                set_note(task.data, self.name, "skipped (flagged)", self.notes_key)
                continue
            if not text or not text.strip():
                task.data[self.output_text_key] = text
                set_note(task.data, self.name, "skipped (empty)", self.notes_key)
                continue
            valid_indices.append(i)
            messages_list.append(self._build_messages(text, task.data))

        if messages_list:
            results = self._generate_remote(messages_list)

            for seq_idx, task_idx in enumerate(valid_indices):
                task = tasks[task_idx]
                input_text = task.data[self.text_key]
                result_text = results[seq_idx]

                if self.enable_validation:
                    ok, reason = self._validate(input_text, result_text)
                    if ok:
                        task.data[self.output_text_key] = result_text
                        note = "applied (modified)" if result_text != input_text else "applied (unchanged)"
                    else:
                        task.data[self.output_text_key] = input_text
                        self._n_filtered += 1
                        note = f"fallback ({reason})"
                else:
                    task.data[self.output_text_key] = result_text
                    note = "applied (modified)" if result_text != input_text else "applied (unchanged)"

                set_note(task.data, self.name, note, self.notes_key)
                self._n_processed += 1

        logger.debug("%s: batch of %d tasks (%d inferred)", self.name, len(tasks), len(messages_list))
        return tasks
