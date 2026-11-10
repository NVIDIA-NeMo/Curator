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

"""vLLM-based conversation generation using VLLMModel from nemo_curator.models."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import unicodedata
import uuid
from collections.abc import Mapping
from copy import deepcopy
from string import Template
from typing import Any, ClassVar

import yaml
from jsonschema import Draft202012Validator
from loguru import logger

from nemo_curator.models.vllm_model import VLLMModel
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_MIN_CLEAN_TEXT_LEN = 2
_MIN_TURNS = 2
_FIRST_PRINTABLE_CHARACTER = 32


class vLLMInference(ProcessingStage[AudioTask, AudioTask]):  # noqa: N801 -- established public stage name
    """Generate multi-speaker conversations via VLLMModel.

    Delegates to :class:`nemo_curator.models.vllm_model.VLLMModel` for engine
    management and inference instead of creating ``vllm.LLM`` directly.

    Args:
        prompt: Static ``"role:content"`` prompt string.
        prompt_field: Key in each entry containing a ``"role:content"`` string.
        prompt_file: Path to a YAML file with ``role: content`` mappings.
            Supports ``$topic`` substitution from each entry's ``topic`` field.
        model: Either a :class:`VLLMModel` instance or a dict whose keys
            match ``VLLMModel.__init__`` parameters (``model``,
            ``max_model_len``, ``tensor_parallel_size``, etc.).
        apply_chat_template: Kwargs forwarded to
            ``tokenizer.apply_chat_template()``.
        max_retry_rounds: Maximum number of regeneration rounds for entries
            whose output fails JSON/turn validation.
        speakers: Exact permitted speaker labels; defaults to prompt YAML metadata
            or speaker_0 and speaker_1. Every label must appear in a conversation.
        min_overlap: Minimum permitted overlap in seconds.
        max_overlap: Maximum permitted overlap in seconds.
    """

    name = "vLLMInference"
    is_resumable = False
    resources = Resources(gpus=1)

    _VLLM_MODEL_KEYS: ClassVar[frozenset[str]] = frozenset(
        name for name in inspect.signature(VLLMModel.__init__).parameters if name != "self"
    )

    def __init__(  # noqa: PLR0913
        self,
        prompt: str | None = None,
        prompt_field: str | None = None,
        prompt_file: str | None = None,
        model: VLLMModel | dict | None = None,
        apply_chat_template: dict | None = None,
        max_retry_rounds: int = 5,
        speakers: list[str] | None = None,
        min_overlap: float = -2.0,
        max_overlap: float = 2.5,
    ):
        super().__init__()

        if isinstance(max_retry_rounds, bool) or not isinstance(max_retry_rounds, int) or max_retry_rounds < 1:
            msg = f"max_retry_rounds must be a positive integer, got {max_retry_rounds}"
            raise ValueError(msg)
        self.max_retry_rounds = max_retry_rounds

        if sum([prompt is not None, prompt_field is not None, prompt_file is not None]) != 1:
            msg = "Exactly one of prompt, prompt_field, or prompt_file must be specified"
            raise ValueError(msg)

        self.prompt = prompt
        self.prompt_field = prompt_field
        self.prompt_file = prompt_file
        self.chat_template_params = {"tokenize": False, "add_generation_prompt": True, **(apply_chat_template or {})}

        if isinstance(model, VLLMModel):
            self._vllm_model = model
        elif isinstance(model, Mapping):
            model_params = dict(model)
            extra = set(model_params) - self._VLLM_MODEL_KEYS
            if extra:
                msg = (
                    "Unsupported VLLMModel parameters: "
                    f"{sorted(extra)}. Supported parameters: {sorted(self._VLLM_MODEL_KEYS)}"
                )
                raise ValueError(msg)
            self._vllm_model = VLLMModel(**model_params)
        else:
            msg = "model must be a VLLMModel instance or a dict of VLLMModel params"
            raise TypeError(msg)

        self._configure_tensor_parallel_resources()

        if self.prompt_file:
            with open(self.prompt_file) as f:
                self.prompt_data = yaml.safe_load(f)
        else:
            self.prompt_data = None

        if self.prompt_data is not None and not isinstance(self.prompt_data, dict):
            msg = "Prompt YAML must contain role/content mappings and optional speakers"
            raise ValueError(msg)
        configured_speakers = speakers if speakers is not None else (self.prompt_data or {}).get("speakers")
        self.speakers = configured_speakers if configured_speakers is not None else ["speaker_0", "speaker_1"]
        if (
            not isinstance(self.speakers, list)
            or len(self.speakers) < _MIN_TURNS
            or any(not self._is_safe_label(speaker) for speaker in self.speakers)
            or len(set(self.speakers)) != len(self.speakers)
        ):
            msg = "speakers must contain at least two distinct, safe string labels"
            raise ValueError(msg)
        if (
            isinstance(min_overlap, bool)
            or isinstance(max_overlap, bool)
            or not math.isfinite(min_overlap)
            or not math.isfinite(max_overlap)
            or min_overlap > max_overlap
        ):
            msg = "overlap bounds must be finite and ordered"
            raise ValueError(msg)
        self.conversation_schema = {
            "type": "object",
            "properties": {
                "turns": {
                    "type": "array",
                    "minItems": len(self.speakers),
                    "items": {
                        "type": "object",
                        "properties": {
                            "speaker": {"type": "string", "enum": self.speakers},
                            "utterance": {"type": "string", "minLength": _MIN_CLEAN_TEXT_LEN},
                            "overlap": {"type": "number", "minimum": min_overlap, "maximum": max_overlap},
                        },
                        "required": ["speaker", "utterance", "overlap"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["turns"],
            "additionalProperties": False,
        }
        Draft202012Validator.check_schema(self.conversation_schema)

        self.tokenizer = None

    def _configure_tensor_parallel_resources(self) -> None:
        """Make the Ray/Xenna GPU reservation match vLLM tensor parallelism."""
        tensor_parallel_size = self._vllm_model.tensor_parallel_size
        if tensor_parallel_size is None:
            if self._vllm_model._llm is not None:
                msg = (
                    "A pre-initialized VLLMModel must set tensor_parallel_size so "
                    "the stage can reserve the GPUs already used by its engine."
                )
                raise ValueError(msg)
            # VLLMModel otherwise auto-detects every visible GPU. A stage must
            # reserve an explicit count, so its safe default is one GPU.
            tensor_parallel_size = 1
            self._vllm_model.tensor_parallel_size = tensor_parallel_size
        if (
            isinstance(tensor_parallel_size, bool)
            or not isinstance(tensor_parallel_size, int)
            or tensor_parallel_size < 1
        ):
            msg = f"tensor_parallel_size must be a positive integer, got {tensor_parallel_size!r}"
            raise ValueError(msg)
        self.resources = Resources(gpus=float(tensor_parallel_size))

    @staticmethod
    def _is_safe_label(value: object) -> bool:
        """Reject labels that could be interpreted as filesystem paths downstream."""
        return (
            isinstance(value, str)
            and bool(value.strip())
            and value not in {".", ".."}
            and "/" not in value
            and "\\" not in value
            and not any(char.isspace() or ord(char) < _FIRST_PRINTABLE_CHARACTER for char in value)
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.prompt_field] if self.prompt_field else []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            "conversation_id",
            "turn_index",
            "conversation_num_turns",
            "speaker",
            "utterance",
            "overlap",
            "topic",
        ]

    def generate_conversation_id(self, turns: list[dict], source: AudioTask | None = None) -> str:
        """Generate deterministic conversation ID from turns."""
        identity = {"version": 1, "turns": turns}
        if source is not None:
            row_id = source.task_id or source._metadata.setdefault("conversation_input_id", uuid.uuid4().hex)
            identity["source"] = {"row_id": row_id, "dataset": source.dataset_name, "data": dict(source.data)}
        canonical = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode()).hexdigest()[:24]

    def _clean_text(self, text: str) -> str:
        """Remove invisible Unicode characters and normalise whitespace."""
        if not text:
            return ""
        cleaned = "".join(
            char
            for char in text
            if unicodedata.category(char) not in ("Cc", "Cf", "Co", "Cs", "Cn") or char in "\n\t "
        )
        return " ".join(cleaned.split()).strip()

    def _is_valid_text(self, text: str) -> bool:
        """Check if text has actual speakable content after cleaning."""
        cleaned = self._clean_text(text)
        if not cleaned or len(cleaned) < _MIN_CLEAN_TEXT_LEN:
            return False
        return any(c.isalpha() for c in cleaned)

    def _turns_are_valid(self, turns: list[dict]) -> bool:
        """Check every turn has a speaker and speakable utterance."""
        return {turn["speaker"] for turn in turns} == set(self.speakers) and all(
            math.isfinite(turn["overlap"]) and self._is_valid_text(turn["utterance"]) for turn in turns
        )

    def validate_json_output(self, text: str) -> dict | None:
        """Validate and parse JSON output from LLM."""
        try:
            if not isinstance(text, str):
                return None
            fenced = re.fullmatch(r"\s*```(?:json)?\s*(\{.*\})\s*```\s*", text, re.DOTALL)
            parsed = json.loads(fenced.group(1) if fenced else text)
        except (json.JSONDecodeError, AttributeError):
            return None

        if not Draft202012Validator(self.conversation_schema).is_valid(parsed):
            return None
        turns = parsed["turns"]
        if not self._turns_are_valid(turns):
            return None

        for turn in turns:
            turn["utterance"] = self._clean_text(turn["utterance"])
        return parsed

    def generate_batch_with_retry(
        self,
        prompts: list[str],
        max_retry_rounds: int = 5,
    ) -> list[dict | None]:
        """Generate with retry logic for failed outputs."""
        current_prompts = prompts.copy()
        current_indices = list(range(len(prompts)))
        validated_outputs: list[dict | None] = [None] * len(prompts)

        for _ in range(max_retry_rounds):
            if not current_prompts:
                break

            generated_texts = self._vllm_model.generate(current_prompts, json_schema=self.conversation_schema)

            next_prompts: list[str] = []
            next_indices: list[int] = []

            if len(generated_texts) != len(current_prompts):
                logger.warning(
                    "vLLM returned {} outputs for {} prompts; retrying every missing output.",
                    len(generated_texts),
                    len(current_prompts),
                )

            for local_idx, (prompt, original_idx) in enumerate(zip(current_prompts, current_indices, strict=True)):
                text = generated_texts[local_idx] if local_idx < len(generated_texts) else ""
                validated = self.validate_json_output(text)

                if validated:
                    validated_outputs[original_idx] = validated
                else:
                    next_prompts.append(prompt)
                    next_indices.append(original_idx)

            if not next_prompts:
                break
            current_prompts = next_prompts
            current_indices = next_indices

        return validated_outputs

    def get_entry_prompt(self, entry: dict) -> str:
        """Build prompt for a single entry."""
        if self.prompt:
            role, content = self.prompt.split(":", 1)
            prompt = {role: content}
        elif self.prompt_field:
            if self.prompt_field not in entry:
                msg = f"Prompt field '{self.prompt_field}' not found in entry"
                raise ValueError(msg)
            role, content = entry[self.prompt_field].split(":", 1)
            prompt = {role: content}
        elif self.prompt_file:
            if not self.prompt_data:
                msg = "Prompt file was not loaded correctly"
                raise ValueError(msg)
            topic = entry.get("topic", "")
            prompt = {
                role: Template(content).safe_substitute(topic=topic)
                for role, content in self.prompt_data.items()
                if role in {"system", "user", "assistant"}
            }
        else:
            msg = "No prompt source specified"
            raise ValueError(msg)

        prompt = {
            role: Template(content).safe_substitute(topic=entry.get("topic", "")) for role, content in prompt.items()
        }
        prompt["system"] = prompt.get("system", "") + (
            "\nReturn only JSON matching this schema. Every configured speaker must appear.\n"
            + json.dumps(self.conversation_schema, ensure_ascii=False)
        )
        entry_chat = []
        for role in ["system", "user", "assistant"]:
            if role not in prompt:
                continue
            entry_chat.append({"role": role, "content": prompt[role]})

        entry_prompt = self.tokenizer.apply_chat_template(entry_chat, **self.chat_template_params)

        if isinstance(entry_prompt, list):
            entry_prompt = self.tokenizer.decode(entry_prompt, skip_special_tokens=False)

        return entry_prompt

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ANN401, ARG002
        """Initialise the VLLMModel engine and tokenizer on the GPU worker."""
        self._ensure_model()

    def _ensure_model(self) -> None:
        """Lazy-init fallback when setup() was not called by the executor.

        Guards on ``self.tokenizer`` rather than the model's ``_llm`` state:
        a pre-configured ``VLLMModel`` passed into ``__init__`` may already
        be set up (``_llm`` not ``None``) while this stage's own tokenizer
        is still unset, since ``setup()`` was never called on *this* stage
        instance. Checking only ``_llm`` would then skip tokenizer loading
        entirely and ``get_entry_prompt`` would fail on ``self.tokenizer``
        being ``None``. Mirrors the same fallback in
        ``nemo_curator.stages.math.modifiers.llm_cleanup.LLMCleanupStage``.
        """
        if self.tokenizer is not None:
            return
        if self._vllm_model._llm is None:
            self._vllm_model.setup()
        self.tokenizer = self._vllm_model.get_tokenizer()

    def process(self, task: AudioTask) -> list[AudioTask]:
        """Generate a conversation from a single topic entry.

        Delegates to ``process_batch`` so vLLM still runs batched inference.
        """
        return self.process_batch([task])

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Generate conversations from a batch of topic entries.

        Each input ``AudioTask`` contains one topic. For each topic, a
        multi-turn conversation is generated. Each turn becomes a separate
        ``AudioTask`` in the output. All turns of one conversation share
        the same ``conversation_id``.
        """
        if len(tasks) == 0:
            return []

        self._ensure_model()

        entry_prompts = [self.get_entry_prompt(t.data) for t in tasks]
        validated_outputs = self.generate_batch_with_retry(entry_prompts, max_retry_rounds=self.max_retry_rounds)

        output_tasks: list[AudioTask] = []
        for i, (task, output_generation) in enumerate(zip(tasks, validated_outputs, strict=False)):
            if output_generation is None:
                logger.warning(f"Skipping failed generation {i + 1}")
                continue

            try:
                conversation_id = self.generate_conversation_id(output_generation["turns"], source=task)
                topic = task.data.get("topic", "unknown")

                for turn_idx, turn in enumerate(output_generation["turns"]):
                    output_task = AudioTask(
                        data={
                            "conversation_id": conversation_id,
                            "turn_index": turn_idx,
                            "conversation_num_turns": len(output_generation["turns"]),
                            "speaker": turn["speaker"],
                            "utterance": turn["utterance"],
                            "overlap": turn.get("overlap", 0.0),
                            "topic": topic,
                        },
                        dataset_name=task.dataset_name,
                        _metadata=deepcopy(task._metadata),
                        _stage_perf=list(task._stage_perf),
                    )
                    output_task._source_id = task._source_id
                    output_tasks.append(output_task)

                logger.info(
                    f"[vLLM] Generated conversation {conversation_id[:8]} with {len(output_generation['turns'])} turns"
                )
            except Exception as e:  # noqa: BLE001 -- skip a bad generation without failing the batch
                logger.error(f"Failed to process output {i + 1}: {e}")

        return output_tasks
