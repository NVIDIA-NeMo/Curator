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
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from openai.types.chat import ChatCompletion

from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.models.client.openai_client import AsyncOpenAIClient
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class AudioTextLLMStage(ProcessingStage[AudioTask, AudioTask]):
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
    Empty or truncated completions always fall back to the input with a note.
    Request failures propagate after the shared client's retries.

    Connections are reused within a batch and closed on its event loop before
    returning. No client or event loop is serialized between workers.
    """

    endpoint: str
    model_name: str
    prompt: str
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

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        for key in (
            "endpoint",
            "model_name",
            "prompt",
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

    def inputs(self) -> tuple[list[str], list[str]]:
        fields = [self.text_key]
        if "{language}" in self.prompt or "{language_rules}" in self.prompt:
            fields.append(self.language_key)
        return [], fields

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_text_key, self.notes_key]

    def _build_messages(self, text: str, data: dict) -> list[dict[str, str]]:
        prompt = self.prompt
        if "{language}" in prompt or "{language_rules}" in prompt:
            language = data.get(self.language_key)
            if not isinstance(language, str) or not language.strip():
                msg = f"{self.language_key} must be a non-empty string for this prompt"
                raise ValueError(msg)
            if "{language_rules}" in prompt:
                code = language.strip().lower()
                if code not in self.language_rules:
                    msg = f"No language_rules configured for {code!r}"
                    raise ValueError(msg)
                prompt = prompt.replace("{language_rules}", self.language_rules[code])
            prompt = prompt.replace("{language}", language.strip())
        if "{text}" in self.prompt:
            return [{"role": "user", "content": prompt.replace("{text}", text)}]
        return [{"role": "system", "content": prompt}, {"role": "user", "content": text}]

    async def _generate(self, messages: list[list[dict[str, str]]]) -> list[ChatCompletion]:
        client = AsyncOpenAIClient(
            base_url=self.endpoint,
            api_key=self.api_key,
            max_concurrent_requests=self.max_concurrent_requests,
            max_retries=self.max_retries,
            timeout=self.request_timeout_s,
        )
        try:
            return list(
                await asyncio.gather(
                    *(
                        client.query_model_response(
                            model=self.model_name, messages=message, generation_config=self.generation_config
                        )
                        for message in messages
                    )
                )
            )
        finally:
            if hasattr(client, "client"):
                await client.client.close()

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def _prepare_batch(
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
        if choice is None:
            ok, reason = False, "missing_choice"
        elif choice.finish_reason != "stop":
            ok, reason = False, f"finish_reason={choice.finish_reason}"
        else:
            ok, reason = self._validate(original, result)
        if not ok:
            return original, f"fallback ({reason})"
        return result, "applied (unchanged)" if result == original else "applied (modified)"

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        pending, indices, messages = self._prepare_batch(tasks)
        metrics = {
            "num_requests": float(len(messages)),
            "num_fallbacks": 0.0,
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
        }
        if messages:
            responses = asyncio.run(self._generate(messages))
            if len(responses) != len(indices):
                msg = "Inference response count does not match request count"
                raise RuntimeError(msg)
            for index, response in zip(indices, responses, strict=True):
                original = tasks[index].data[self.text_key]
                result, note = self._read_response(original, response)
                metrics["num_fallbacks"] += note.startswith("fallback (")
                pending.append((index, result, note))
                if response.usage is not None:
                    metrics["prompt_tokens"] += response.usage.prompt_tokens or 0
                    metrics["completion_tokens"] += response.usage.completion_tokens or 0

        # Commit outputs only after every request and response has been handled.
        for index, result, note in pending:
            data = tasks[index].data
            data[self.output_text_key] = result
            if not isinstance(data.get(self.notes_key), dict):
                data[self.notes_key] = {}
            data[self.notes_key][self.name] = note
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
