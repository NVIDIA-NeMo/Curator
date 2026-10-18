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

"""Prompt-based transcript processing through an OpenAI-compatible endpoint."""

import asyncio
import math
import re
import threading
import unicodedata
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import ClassVar, Literal

from loguru import logger
from openai.types.chat import ChatCompletion

from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.models.client.openai_client import AsyncOpenAIClient
from nemo_curator.stages.audio._agent._agent_ready import (
    AgentReady,
    ConditionalWrite,
    Gates,
    IOSpec,
    Role,
    StageContract,
    StaticHints,
)
from nemo_curator.stages.audio.text_filtering.language_utils import verify_language
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class AudioTextLLMStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Transform an audio transcript using a shared inference server, such as Dynamo.

    The server owns the model and GPUs; this stage only sends text requests.
    It returns the original tasks in order, preserving audio and task metadata.
    ``prompt`` is a system message followed by the transcript, or a single user
    message when it contains ``{text}``. ``{language}`` and ``{language_rules}``
    resolve per row from ``language_key`` and the optional rules mapping.

    ``validation_mode='itn'`` rejects added words and novel Latin tokens;
    ``'tn'`` detects English leakage and conversion from a non-Latin script.
    Both reject outputs retaining less than ``min_word_retention_ratio`` of
    the input words when the input has at least ``min_words_for_deletion_check``
    words. ``'none'`` disables these rewrite heuristics, for example for captions.
    Responses use the reference rewrite guards; unvalidated operations retain
    their response text, including empty or truncated text.
    Request failures propagate after the shared client's retries.

    ``mode='recover_entities'`` renders a system/user prompt from the normalized
    transcript and ``ground_truth_key``, then removes reasoning/fence wrappers.
    ``mode='language_id'`` verifies the prediction against ``language_key`` and
    flags single-language mismatches for downstream stages. ``operations`` groups
    independent text operations into one concurrent batch with a shared client.
    Dependent operations belong in separate pipeline stages.

    Each stage instance reuses one client and event loop across sequential batches.
    Teardown or a failed batch closes both; pending requests are cancelled and
    drained before cleanup. Garbage collection also attempts cleanup. A batch
    deadline bounds request retries and semaphore waits. Concurrent calls on
    the same stage instance are not supported.

    Args:
        text_key: Transcript for this audio row; no audio samples are sent.
        output_text_key: Model response, or the input transcript after a rewrite
            fallback. In language-ID mode this contains language labels, not a
            transcript. Generated text is not an independently verified reference.
        ground_truth_key: Existing comparison transcript for entity recovery;
            this stage does not verify its accuracy.
        language_key: Source-language label used in prompts and verification.
        notes_key: Shared audit mapping, augmented with this operation's outcome.
        skip_me_key: Existing skip reason; language verification may set it for
            an unparseable or mismatched single-language prediction. Rows are
            retained, so downstream stages must honor this flag to exclude them.
    """

    endpoint: str
    model_name: str
    prompt: str = ""
    mode: Literal["text", "recover_entities", "language_id"] = "text"
    user_prompt: str | None = None
    ground_truth_key: str = "granary_v1_prediction"
    operations: list[dict] | None = None
    num_workers_override: int | None = None
    _operations: list["AudioTextLLMStage"] = field(default_factory=list, init=False, repr=False)
    text_key: str = "text"
    output_text_key: str = "processed_text"
    language_key: str = "source_lang"
    language_rules: dict[str, str] | None = None
    generation_config: GenerationConfig | dict | None = None
    api_key: str = field(default="unused", repr=False)  # pragma: allowlist secret
    batch_size: int = 64
    max_concurrent_requests: int = 16
    request_timeout_s: float = 120.0
    max_retries: int = 3
    validation_mode: Literal["none", "itn", "tn"] = "none"
    min_word_retention_ratio: float = 0.3
    min_words_for_deletion_check: int = 8
    english_leak_threshold: int = 2
    script_drift_min_chars: int = 3
    skip_me_key: str = "_skipme"
    notes_key: str = "additional_notes"
    skip_if_output_exists: bool = False
    name: str = "AudioTextLLM"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    _runner: asyncio.Runner | None = field(default=None, init=False, repr=False, compare=False)
    _client: AsyncOpenAIClient | None = field(default=None, init=False, repr=False, compare=False)
    _cleanup: weakref.finalize | None = field(default=None, init=False, repr=False, compare=False)
    _runtime_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    KEY_ROLE_OVERRIDES: ClassVar[dict[str, Role]] = {
        "output_text_key": "text",
        "ground_truth_key": "reference_text",
        "language_key": "unknown",
        "notes_key": "unknown",
        "skip_me_key": "unknown",
    }
    INTERNAL_KEY_FIELDS: ClassVar[frozenset[str]] = frozenset({"api_key"})
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(
            lifecycle_side_effects=True,
            runtime_secrets=["api_key"],
            # Requests are independent, but the remote generator may be stochastic.
            per_row_independent=False,
        )
    )

    def describe(self) -> StageContract:
        reads, optional_reads, writes = [], [], []
        conditional_writes = []
        key_roles: dict[str, Role] = {}
        for operation in self.operation_stages():
            optional_reads.extend([operation.notes_key, operation.skip_me_key])
            if operation.mode == "recover_entities":
                optional_reads.extend([operation.text_key, operation.ground_truth_key])
                key_roles[operation.ground_truth_key] = "reference_text"
            else:
                reads.append(operation.text_key)
                if "{language_rules}" in operation.prompt:
                    reads.append(operation.language_key)
                elif "{language}" in operation.prompt or operation.mode == "language_id":
                    optional_reads.append(operation.language_key)
            if operation.skip_if_output_exists or operation.mode != "recover_entities":
                optional_reads.append(operation.output_text_key)
            writes.extend([operation.output_text_key, operation.notes_key])
            key_roles[operation.text_key] = "text"
            key_roles[operation.output_text_key] = "prediction" if operation.mode == "language_id" else "text"
            if operation.mode == "language_id":
                conditional_writes.append(
                    ConditionalWrite(
                        writes=IOSpec(data_keys=[operation.skip_me_key]),
                        condition=(
                            f"'{operation.skip_me_key}' is unset, '{operation.language_key}' is recognized, "
                            f"and nonempty '{operation.output_text_key}' is unparseable "
                            "or identifies one wrong language"
                        ),
                    )
                )
        return StageContract(
            reads=IOSpec(data_keys=list(dict.fromkeys(reads))),
            optional_reads=IOSpec(data_keys=list(dict.fromkeys(optional_reads))),
            writes=IOSpec(data_keys=list(dict.fromkeys(writes))),
            conditional_writes=conditional_writes,
            key_roles=key_roles,
            cardinality="1:1",
            gates=replace(self.AGENT_STATIC.gates, requires_gpu=self.resources.requires_gpu),
        )

    def __getstate__(self) -> dict:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in {"_runner", "_client", "_cleanup", "_runtime_lock"}
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._runtime_lock = threading.Lock()

    @staticmethod
    def _close_runtime(runner: asyncio.Runner, client: AsyncOpenAIClient, name: str) -> None:
        async def close_client() -> None:
            if hasattr(client, "client"):
                await asyncio.wait_for(client.client.close(), timeout=30.0)

        close = close_client()
        try:
            with logger.catch(message=f"{name}: client cleanup failed"):
                runner.run(close)
        finally:
            close.close()
            with logger.catch(message=f"{name}: event loop cleanup failed"):
                runner.close()

    def _reset_runtime(self) -> None:
        try:
            if self._cleanup is not None:
                self._cleanup()
        finally:
            self._runner = self._client = self._cleanup = None

    def teardown(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            msg = "AudioTextLLMStage requires a synchronous worker thread"
            raise RuntimeError(msg)
        if not self._runtime_lock.acquire(blocking=False):
            msg = "Cannot tear down AudioTextLLMStage while a batch is running"
            raise RuntimeError(msg)
        try:
            self._reset_runtime()
        finally:
            self._runtime_lock.release()

    def __post_init__(self) -> None:  # noqa: C901, PLR0912, PLR0915
        for key in (
            "endpoint",
            "model_name",
            "text_key",
            "output_text_key",
            "language_key",
            "skip_me_key",
            "notes_key",
        ):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                msg = f"{key} must be a non-empty string"
                raise ValueError(msg)
        if self.mode not in {"text", "recover_entities", "language_id"}:
            msg = "mode must be text, recover_entities, or language_id"
            raise ValueError(msg)
        if self.num_workers_override is not None and (
            isinstance(self.num_workers_override, bool)
            or not isinstance(self.num_workers_override, int)
            or self.num_workers_override < 1
        ):
            msg = "num_workers_override must be a positive integer"
            raise ValueError(msg)
        if not isinstance(self.prompt, str) or (not self.operations and not self.prompt.strip()):
            msg = "prompt must be a non-empty string unless operations are configured"
            raise ValueError(msg)
        if self.mode == "recover_entities":
            if not isinstance(self.user_prompt, str) or not self.user_prompt.strip():
                msg = "recover_entities requires a user_prompt template"
                raise ValueError(msg)
            if not isinstance(self.ground_truth_key, str) or not self.ground_truth_key.strip():
                msg = "ground_truth_key must be a non-empty string"
                raise ValueError(msg)
            if self.ground_truth_key in {self.output_text_key, self.notes_key, self.skip_me_key}:
                msg = "ground_truth_key must not be overwritten by outputs"
                raise ValueError(msg)
        if self.output_text_key in {self.skip_me_key, self.notes_key, self.language_key}:
            msg = "output_text_key must not overwrite skip, notes, or language fields"
            raise ValueError(msg)
        if self.notes_key in {self.text_key, self.skip_me_key, self.language_key}:
            msg = "notes_key must not overwrite text, skip, or language fields"
            raise ValueError(msg)
        for key in (
            "batch_size",
            "max_concurrent_requests",
            "min_words_for_deletion_check",
            "english_leak_threshold",
            "script_drift_min_chars",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                msg = f"{key} must be a positive integer"
                raise ValueError(msg)
        if (
            not 0 < self.request_timeout_s < float("inf")
            or isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 0
        ):
            msg = "request_timeout_s must be positive and max_retries must be non-negative"
            raise ValueError(msg)
        if self.validation_mode not in {"none", "itn", "tn"}:
            msg = "validation_mode must be none, itn, or tn"
            raise ValueError(msg)
        if not 0 <= self.min_word_retention_ratio <= 1:
            msg = "min_word_retention_ratio must be between zero and one"
            raise ValueError(msg)
        if self.generation_config is None:
            self.generation_config = GenerationConfig()
        elif isinstance(self.generation_config, Mapping):
            self.generation_config = GenerationConfig(**self.generation_config)
        if not isinstance(self.generation_config, GenerationConfig):
            msg = "generation_config must be a GenerationConfig or mapping"
            raise TypeError(msg)
        extra = self.generation_config.extra_kwargs or {}
        body = extra.get("extra_body") or {}
        reserved = {"stream", "n", "messages", "model"}
        if (
            self.generation_config.stream
            or self.generation_config.n != 1
            or reserved.intersection(extra)
            or reserved.intersection(body)
        ):
            msg = "AudioTextLLMStage requires stream=False, n=1, and no request-shape overrides"
            raise ValueError(msg)
        if self.language_rules is not None and (
            not isinstance(self.language_rules, Mapping)
            or any(
                not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
                for k, v in self.language_rules.items()
            )
        ):
            msg = "language_rules must map non-empty language codes to non-empty strings"
            raise ValueError(msg)
        if "{language_rules}" in self.prompt and not self.language_rules:
            msg = "prompt requires language_rules"
            raise ValueError(msg)

        if self.operations is not None:
            self._configure_operations()

    def _configure_operations(self) -> None:
        if not isinstance(self.operations, list) or not self.operations or self.prompt or self.mode != "text":
            msg = "operations must be a nonempty list without a parent prompt or special mode"
            raise ValueError(msg)
        shared = {
            key: getattr(self, key)
            for key in (
                "endpoint",
                "model_name",
                "api_key",
                "batch_size",
                "max_concurrent_requests",
                "request_timeout_s",
                "max_retries",
                "language_key",
                "notes_key",
                "skip_me_key",
            )
        }
        for operation in self.operations:
            if not isinstance(operation, Mapping):
                msg = "Each parallel operation must be a mapping"
                raise TypeError(msg)
            if (
                "operations" in operation
                or set(shared).intersection(operation)
                or operation.get("mode", "text") != "text"
            ):
                msg = (
                    "Parallel operations must be independent text prompts using the parent's client and metadata keys"
                )
                raise ValueError(msg)
            self._operations.append(type(self)(**shared, **{"generation_config": self.generation_config, **operation}))
        names = [operation.name for operation in self._operations]
        outputs = [operation.output_text_key for operation in self._operations]
        inputs = {key for operation in self._operations for key in operation.inputs()[1]}
        if len(set(names)) != len(names) or len(set(outputs)) != len(outputs) or set(outputs).intersection(inputs):
            msg = "Parallel operations need unique names/outputs and must not consume each other's outputs"
            raise ValueError(msg)

    def operation_stages(self) -> list["AudioTextLLMStage"]:
        """Return the independent operations executed by this worker."""
        return self._operations or [self]

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], self.describe().reads.data_keys

    def outputs(self) -> tuple[list[str], list[str]]:
        if self._operations:
            return [], list(dict.fromkeys(key for operation in self._operations for key in operation.outputs()[1]))
        return [], [self.output_text_key, self.notes_key, *([self.skip_me_key] if self.mode == "language_id" else [])]

    def _build_messages(self, text: str, data: dict) -> list[dict[str, str]]:
        if self.mode == "recover_entities":
            return [
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": self.user_prompt.format(
                        ground_truth=data[self.ground_truth_key].strip(), normalized=text.strip()
                    ),
                },
            ]
        prompt = self.prompt
        if "{language}" in prompt or "{language_rules}" in prompt:
            language = data.get(self.language_key)
            if "{language_rules}" in prompt:
                if language is None or not str(language).strip():
                    msg = f"{self.language_key} must be non-empty for language_rules"
                    raise ValueError(msg)
                code = str(language).strip().lower()
                if code not in self.language_rules:
                    msg = f"No language_rules configured for {code!r}"
                    raise ValueError(msg)
                prompt = prompt.replace("{language_rules}", self.language_rules[code])
            language = str(language).strip() if language is not None and str(language).strip() else "English"
            prompt = prompt.replace("{language}", language)
        if "{text}" in self.prompt:
            return [{"role": "user", "content": prompt.replace("{text}", text)}]
        return [{"role": "system", "content": prompt}, {"role": "user", "content": text}]

    def _batch_timeout(self, num_requests: int, client: AsyncOpenAIClient) -> float:
        waves = max(1, math.ceil(num_requests / self.max_concurrent_requests))
        retry_wait = client.base_delay * (2**self.max_retries - 1) + self.max_retries
        sdk_retries = client.client.max_retries
        # The SDK accepts Retry-After delays of up to 60 seconds per retry.
        request_budget = self.request_timeout_s * (sdk_retries + 1) + 60.0 * sdk_retries
        return waves * (request_budget * (self.max_retries + 1) + retry_wait) + 120.0

    async def _generate(
        self, messages: list[list[dict[str, str]]], configs: list[GenerationConfig]
    ) -> list[ChatCompletion]:
        client = self._client
        assert client is not None  # noqa: S101
        requests = [
            asyncio.create_task(
                client.query_model_response(model=self.model_name, messages=message, generation_config=config)
            )
            for message, config in zip(messages, configs, strict=True)
        ]
        try:
            return list(
                await asyncio.wait_for(asyncio.gather(*requests), timeout=self._batch_timeout(len(messages), client))
            )
        finally:
            for request in requests:
                request.cancel()
            await asyncio.gather(*requests, return_exceptions=True)

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def _prepare_batch(  # noqa: C901
        self, tasks: list[AudioTask]
    ) -> tuple[list[tuple[int, str, str]], list[int], list[list[dict[str, str]]]]:
        pending: list[tuple[int, str, str]] = []
        indices: list[int] = []
        messages: list[list[dict[str, str]]] = []
        for index, task in enumerate(tasks):
            if not isinstance(task, AudioTask):
                msg = "AudioTextLLMStage only accepts AudioTask inputs"
                raise TypeError(msg)
            if self.skip_if_output_exists and self.output_text_key in task.data:
                pending.append((index, task.data[self.output_text_key], "skipped (output exists)"))
                continue
            if self.mode == "recover_entities":
                normalized = task.data.get(self.text_key, "")
                normalized = normalized if isinstance(normalized, str) else ""
                ground_truth = task.data.get(self.ground_truth_key, "")
                if task.data.get(self.skip_me_key):
                    pending.append((index, normalized, "skipped (flagged)"))
                elif not normalized.strip() or not isinstance(ground_truth, str) or not ground_truth.strip():
                    pending.append((index, normalized, "skipped (empty)"))
                else:
                    messages.append(self._build_messages(normalized, task.data))
                    indices.append(index)
                continue
            if task.data.get(self.skip_me_key):
                pending.append((index, task.data.get(self.output_text_key, ""), "skipped (flagged)"))
                continue
            if self.text_key not in task.data:
                msg = f"Missing input field {self.text_key!r}"
                raise ValueError(msg)
            text = task.data[self.text_key]
            if not isinstance(text, str):
                msg = f"{self.text_key} must be a string"
                raise TypeError(msg)
            if not text.strip():
                pending.append((index, text, "skipped (empty)"))
                continue
            messages.append(self._build_messages(text, task.data))
            indices.append(index)
        return pending, indices, messages

    def _read_response(self, original: str, response: ChatCompletion) -> tuple[str, str]:
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice is not None else None
        result = content.strip() if isinstance(content, str) else ""
        if self.mode == "recover_entities":
            result = self._clean_entity_response(result)
            if not result:
                return original, "fallback (empty output)"
            return result, "unchanged" if result == original else "recovered"
        if self.validation_mode != "none":
            ok, reason = self._validate(original, result)
            if not ok:
                return original, f"fallback ({reason})"
        return result, "applied (unchanged)" if result == original else "applied (modified)"

    @staticmethod
    def _clean_entity_response(raw: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":  # noqa: PLR2004
            text = text[1:-1].strip()
        return text

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            msg = "AudioTextLLMStage requires a synchronous worker thread"
            raise RuntimeError(msg)
        if not self._runtime_lock.acquire(blocking=False):
            msg = "Concurrent batches on the same AudioTextLLMStage instance are not supported"
            raise RuntimeError(msg)
        try:
            return self._process_batch(tasks)
        except BaseException:
            self._reset_runtime()
            raise
        finally:
            self._runtime_lock.release()

    def _process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        pending = []
        requests = []
        messages = []
        for operation in self.operation_stages():
            skipped, indices, operation_messages = operation._prepare_batch(tasks)
            pending.extend((operation, *item) for item in skipped)
            requests.extend((operation, index) for index in indices)
            messages.extend(operation_messages)
        metrics = {
            "num_requests": float(len(messages)),
            "num_usage_reports": 0.0,
            "num_fallbacks": 0.0,
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
        }
        if messages:
            if self._runner is None:
                self._runner = asyncio.Runner()
                self._client = AsyncOpenAIClient(
                    base_url=self.endpoint,
                    api_key=self.api_key,
                    max_concurrent_requests=self.max_concurrent_requests,
                    max_retries=self.max_retries,
                    timeout=self.request_timeout_s,
                )
                self._cleanup = weakref.finalize(
                    self, type(self)._close_runtime, self._runner, self._client, self.name
                )
                self._client.setup()
            responses = self._runner.run(
                self._generate(messages, [operation.generation_config for operation, _ in requests])
            )
            if len(responses) != len(requests):
                msg = "Inference response count does not match request count"
                raise RuntimeError(msg)
            for (operation, index), response in zip(requests, responses, strict=True):
                original = tasks[index].data[operation.text_key]
                result, note = operation._read_response(original, response)
                metrics["num_fallbacks"] += note.startswith("fallback (")
                pending.append((operation, index, result, note))
                if response.usage is not None:
                    metrics["num_usage_reports"] += 1
                    metrics["prompt_tokens"] += response.usage.prompt_tokens or 0
                    metrics["completion_tokens"] += response.usage.completion_tokens or 0

        # Commit outputs only after every request and response has been handled.
        for operation, index, result, note in pending:
            data = tasks[index].data
            data[operation.output_text_key] = result
            if not isinstance(data.get(operation.notes_key), dict):
                data[operation.notes_key] = {}
            data[operation.notes_key][operation.name] = note
            if operation.mode == "language_id":
                verify_language(
                    data, operation.output_text_key, operation.language_key, operation.skip_me_key, operation.notes_key
                )
        self._log_metrics(metrics)
        return tasks

    _ALPHA_RE = re.compile(r"[a-zA-Z]+")
    _ROMAN_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
    _VALID_ITN_ALPHA = frozenset(
        {
            "st",
            "nd",
            "rd",
            "th",
            "dr",
            "mr",
            "mrs",
            "ms",
            "prof",
            "sr",
            "jr",
            "ave",
            "blvd",
            "ln",
            "ct",
            "pl",
            "n",
            "s",
            "e",
            "w",
            "ne",
            "nw",
            "se",
            "sw",
            "kg",
            "km",
            "cm",
            "mm",
            "mg",
            "lb",
            "lbs",
            "oz",
            "ft",
            "mi",
            "ml",
            "mph",
            "h",
            "g",
            "m",
            "l",
            "am",
            "pm",
            "vs",
            "dept",
            "inc",
            "corp",
            "ltd",
            "co",
            "no",
        }
    )

    def _check_novel_words(self, in_words: list[str], out_words: list[str]) -> list[str]:
        in_alpha: set[str] = set()
        for w in in_words:
            for part in self._ALPHA_RE.findall(w):
                in_alpha.add(part.lower())
        novel: list[str] = []
        for w in out_words:
            for part in self._ALPHA_RE.findall(w):
                p = part.lower()
                if p in in_alpha or p in self._VALID_ITN_ALPHA:
                    continue
                if self._ROMAN_RE.match(p):
                    continue
                novel.append(p)
        return novel

    # English marker words whose appearance in TN output signals the LLM
    # translated to English instead of normalizing.
    _WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
    _EN_MARKERS = frozenset(
        """the and for you are was were will would should could have has had from which what when
        where who why how out about over under between because their there they them your not but than
        then into within during through after before above below against without
        hundred thousand million billion point percent half quarter dozen clock means people""".split()  # noqa: SIM905
    )
    # Number magnitudes / point / percent: a single new occurrence flags a leak.
    _EN_STRONG_MARKERS = frozenset({"hundred", "thousand", "million", "billion", "point", "percent"})

    def _english_leak(self, in_words: list[str], out_words: list[str]) -> list[str]:
        in_en = {w for w in in_words if w in self._EN_MARKERS}
        return sorted({w for w in out_words if w in self._EN_MARKERS} - in_en)

    @staticmethod
    def _script_counts(text: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ch in text:
            if ch.isalpha():
                try:
                    family = unicodedata.name(ch).split()[0]  # LATIN / CYRILLIC / GREEK / ...
                except ValueError:
                    continue
                counts[family] = counts.get(family, 0) + 1
        return counts

    def _script_drift(self, input_text: str, output_text: str) -> bool:
        """True if a non-Latin-script input produced a substantial Latin run
        (transliteration / romanization the model wasn't asked for)."""
        in_sc = self._script_counts(input_text)
        if not in_sc or in_sc.get("LATIN", 0) > 0:
            return False  # input is empty-of-letters or already contains Latin
        non_latin = sum(v for k, v in in_sc.items() if k != "LATIN")
        if non_latin < self.script_drift_min_chars:
            return False
        return self._script_counts(output_text).get("LATIN", 0) >= self.script_drift_min_chars

    def _validate(self, input_text: str, output_text: str) -> tuple[bool, str]:  # noqa: PLR0911
        in_words = input_text.lower().split()
        out_words = output_text.lower().split()
        # Non-empty input -> empty output always fails.
        if input_text.strip() and not output_text.strip():
            return False, "empty_output"
        if self.validation_mode == "none" or not in_words or not out_words:
            return True, "ok"

        if self.validation_mode == "tn":
            # TN expands digits to words; word-count increase is expected.
            in_tokens = self._WORD_RE.findall(input_text.lower())
            out_tokens = self._WORD_RE.findall(output_text.lower())
            # Only NEW English counts; an already-English input is a langID issue.
            in_en = {w for w in in_tokens if w in self._EN_MARKERS}
            if len(in_en) < self.english_leak_threshold:
                leak = self._english_leak(in_tokens, out_tokens)
                strong = [w for w in leak if w in self._EN_STRONG_MARKERS]
                if strong or len(leak) >= self.english_leak_threshold:
                    return False, f"english_leak: {leak}"
            if self._script_drift(input_text, output_text):
                return False, "script_drift_to_latin"
        else:
            # ITN: word-count increase or novel Latin token is suspicious.
            if len(out_words) > len(in_words):
                return False, f"word_count_increase ({len(in_words)}->{len(out_words)})"
            novel = self._check_novel_words(in_words, out_words)
            if novel:
                return False, f"novel_words: {novel}"

        if (
            len(in_words) >= self.min_words_for_deletion_check
            and len(out_words) < len(in_words) * self.min_word_retention_ratio
        ):
            return False, f"excessive_deletion ({len(in_words)}->{len(out_words)})"
        return True, "ok"
