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

"""Fused remote LLM stage: fan out multiple independent prompts in parallel.

Running N independent :class:`RemoteTextLLMStage` actors sequentially leaves the
Dynamo server underutilised — only one stage sends requests at a time.
:class:`FusedRemoteTextLLMStage` replaces N sequential ``map_batches`` actors with
a single actor that fires all enabled sub-stage prompts concurrently via
``asyncio.gather``, keeping the server saturated across the full batch.

Typical usage in ``run_text_pipeline.py`` with ``--fuse_stages``:

    sub = [LanguageID_stage, ITN_stage, Captioning_stage, ...]
    stages.append(FusedRemoteTextLLMStage(sub_stages=sub, ...))
    # LLMLanguageVerification and DisfluencyRemoval still follow as separate stages.
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
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.stages.audio.text_filtering.remote_text_llm_stage import RemoteTextLLMStage
    from nemo_curator.tasks import AudioTask


@dataclass
class FusedRemoteTextLLMStage(ProcessingStage["AudioTask", "AudioTask"]):
    """Single Ray actor that runs multiple independent LLM stages in parallel.

    Each sub-stage is a :class:`RemoteTextLLMStage` instance whose prompt,
    text key, output key, and validation settings are respected.  A single
    shared :class:`AsyncOpenAIClient` + :class:`PersistentEventLoop` services
    all sub-stages, and every ``process_batch`` call fans out all their
    messages simultaneously via ``asyncio.gather``.

    Args:
        sub_stages: Enabled :class:`RemoteTextLLMStage` instances to run in
            parallel.  Only stages present here send requests; disabled stages
            are simply absent.
        inference_base_url: OpenAI-compatible base URL of the shared server.
        inference_api_key: API key forwarded with each request.
        served_model_name: ``model=`` value in requests.  Defaults to the
            first sub-stage's ``model_id`` when unset.
        max_concurrent_requests: Semaphore bound *across all sub-stages
            combined*.  E.g. 64 means at most 64 HTTP requests in-flight at
            once, regardless of how many sub-stages are present.
        request_timeout: Per-request timeout in seconds.
    """

    sub_stages: list[RemoteTextLLMStage] = field(default_factory=list)

    inference_base_url: str = ""
    inference_api_key: str = "EMPTY"
    served_model_name: str | None = None
    max_concurrent_requests: int = 64
    request_timeout: int = 120

    name: str = "FusedRemoteTextLLM"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 128

    _client: Any = field(default=None, init=False, repr=False)
    _loop_runner: Any = field(default=None, init=False, repr=False)
    _gen_configs: dict[str, GenerationConfig] = field(default_factory=dict, init=False, repr=False)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._client is not None:
            return
        if not self.inference_base_url:
            msg = "FusedRemoteTextLLMStage requires inference_base_url"
            raise ValueError(msg)
        if not self.sub_stages:
            msg = "FusedRemoteTextLLMStage requires at least one sub-stage"
            raise ValueError(msg)

        from nemo_curator.models.client.async_runner import PersistentEventLoop
        from nemo_curator.models.client.openai_client import AsyncOpenAIClient

        # Resolve each sub-stage's prompt (sets sub_stage._system_prompt).
        # We do NOT call sub_stage.setup() — that would create per-sub-stage
        # clients and event loops which we deliberately share here.
        for sub in self.sub_stages:
            sub._system_prompt = sub._resolve_prompt()

        self._client = AsyncOpenAIClient(
            max_concurrent_requests=self.max_concurrent_requests,
            base_url=self.inference_base_url,
            api_key=self.inference_api_key,
            timeout=self.request_timeout,
        )
        self._client.setup()

        self._loop_runner = PersistentEventLoop(name=self.name)
        self._loop_runner.start()

        for sub in self.sub_stages:
            self._gen_configs[sub.name] = GenerationConfig(
                temperature=0.0,
                top_p=1.0,
                max_tokens=sub.max_output_tokens,
                seed=0,
                extra_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            )

        sub_names = [s.name for s in self.sub_stages]
        logger.info(
            "%s: ready (remote=%s, model=%s, sub_stages=%s)",
            self.name,
            self.inference_base_url,
            self.served_model_name or self.sub_stages[0].model_id,
            sub_names,
        )

    def teardown(self) -> None:
        if self._loop_runner is not None:
            if self._client is not None:
                with contextlib.suppress(Exception):
                    self._loop_runner.run(self._client.client.close(), timeout=30)
            self._loop_runner.close()
            self._loop_runner = None
        self._client = None

    # ── Timeout ──────────────────────────────────────────────────────────────

    def _batch_timeout(self, n_total_requests: int) -> float:
        max_retries = getattr(self._client, "max_retries", 3)
        per_request = self.request_timeout * (max_retries + 1)
        concurrency = max(1, self.max_concurrent_requests)
        waves = max(1, math.ceil(n_total_requests / concurrency))
        return per_request * waves + 120

    # ── Processing ───────────────────────────────────────────────────────────

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        if self._client is None:
            msg = "FusedRemoteTextLLMStage.setup() was not called"
            raise RuntimeError(msg)

        model = self.served_model_name or self.sub_stages[0].model_id

        # Per sub-stage: collect (task_index, messages) for valid tasks only.
        # Structure: {sub_name: (valid_indices, messages_list)}
        per_sub: dict[str, tuple[list[int], list[list[dict]]]] = {}

        for sub in self.sub_stages:
            valid_indices: list[int] = []
            messages_list: list[list[dict]] = []
            for i, task in enumerate(tasks):
                text = task.data.get(sub.text_key, "")
                skip = task.data.get(sub.skip_me_key, "")
                if skip:
                    task.data[sub.output_text_key] = ""
                    set_note(task.data, sub.name, "skipped (flagged)", sub.notes_key)
                    continue
                if not text or not text.strip():
                    task.data[sub.output_text_key] = text
                    set_note(task.data, sub.name, "skipped (empty)", sub.notes_key)
                    continue
                valid_indices.append(i)
                messages_list.append(sub._build_messages(text, task.data))
            per_sub[sub.name] = (valid_indices, messages_list)

        total_requests = sum(len(msgs) for _, msgs in per_sub.values())
        if total_requests == 0:
            return tasks

        # Fan out ALL messages from ALL sub-stages in one asyncio.gather call.
        # Each coroutine carries (sub_name, seq_idx) so we can route results back.
        async def _one(sub_name: str, seq_idx: int, messages: list[dict]) -> tuple[str, int, str]:
            resp = await self._client.query_model(
                messages=messages,
                model=model,
                generation_config=self._gen_configs[sub_name],
            )
            return sub_name, seq_idx, (resp[0] if resp else "").strip()

        async def _all() -> list[tuple[str, int, str]]:
            coros = []
            for sub in self.sub_stages:
                _, messages_list = per_sub[sub.name]
                for seq_idx, msgs in enumerate(messages_list):
                    coros.append(_one(sub.name, seq_idx, msgs))
            return await asyncio.gather(*coros)

        raw_results = self._loop_runner.run(_all(), timeout=self._batch_timeout(total_requests))

        # Route results back to tasks, applying per-sub-stage validation.
        # Group results by sub_name first for O(1) lookup.
        results_by_sub: dict[str, dict[int, str]] = {}
        for sub_name, seq_idx, text in raw_results:
            results_by_sub.setdefault(sub_name, {})[seq_idx] = text

        for sub in self.sub_stages:
            valid_indices, _ = per_sub[sub.name]
            sub_results = results_by_sub.get(sub.name, {})
            for seq_idx, task_idx in enumerate(valid_indices):
                task = tasks[task_idx]
                input_text = task.data[sub.text_key]
                result_text = sub_results.get(seq_idx, "")

                if sub.enable_validation:
                    ok, reason = sub._validate(input_text, result_text)
                    if ok:
                        task.data[sub.output_text_key] = result_text
                        note = "applied (modified)" if result_text != input_text else "applied (unchanged)"
                    else:
                        task.data[sub.output_text_key] = input_text
                        sub._n_filtered += 1
                        note = f"fallback ({reason})"
                else:
                    task.data[sub.output_text_key] = result_text
                    note = "applied (modified)" if result_text != input_text else "applied (unchanged)"

                set_note(task.data, sub.name, note, sub.notes_key)
                sub._n_processed += 1

        logger.debug(
            "%s: batch=%d, total_requests=%d, sub_stages=%s",
            self.name,
            len(tasks),
            total_requests,
            [s.name for s in self.sub_stages],
        )
        return tasks

    # ── ProcessingStage interface ─────────────────────────────────────────────

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def num_workers(self) -> int | None:
        if self.sub_stages:
            return getattr(self.sub_stages[0], "num_workers_override", None)
        return None
