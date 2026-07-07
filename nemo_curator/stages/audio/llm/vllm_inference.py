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
import json
import re
import unicodedata
from string import Template
from typing import Any, ClassVar

import yaml
from loguru import logger

from nemo_curator.models.vllm_model import VLLMModel
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_MIN_CLEAN_TEXT_LEN = 2
_MIN_TURNS = 2


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
    """

    name = "vLLMInference"
    resources = Resources(gpus=1)

    _VLLM_MODEL_KEYS: ClassVar[set[str]] = {
        "model", "max_model_len", "tensor_parallel_size",
        "max_num_batched_tokens", "temperature", "top_p", "top_k",
        "min_p", "max_tokens", "cache_dir", "disable_dual_chunk_attention",
    }

    def __init__(
        self,
        prompt: str | None = None,
        prompt_field: str | None = None,
        prompt_file: str | None = None,
        model: VLLMModel | dict | None = None,
        apply_chat_template: dict | None = None,
    ):
        super().__init__()

        if sum([prompt is not None, prompt_field is not None, prompt_file is not None]) != 1:
            msg = "Exactly one of prompt, prompt_field, or prompt_file must be specified"
            raise ValueError(msg)

        self.prompt = prompt
        self.prompt_field = prompt_field
        self.prompt_file = prompt_file
        self.chat_template_params = apply_chat_template or {}

        if isinstance(model, VLLMModel):
            self._vllm_model = model
        elif isinstance(model, dict):
            vllm_params = {k: v for k, v in model.items() if k in self._VLLM_MODEL_KEYS}
            extra = set(model) - self._VLLM_MODEL_KEYS
            if extra:
                logger.warning(
                    f"Ignoring parameters not supported by VLLMModel: {extra}. "
                    f"Pass a pre-configured VLLMModel instance for full control."
                )
            self._vllm_model = VLLMModel(**vllm_params)
        else:
            msg = "model must be a VLLMModel instance or a dict of VLLMModel params"
            raise TypeError(msg)

        if self.prompt_file:
            with open(self.prompt_file) as f:
                self.prompt_data = yaml.safe_load(f)
        else:
            self.prompt_data = None

        self.tokenizer = None

    def generate_conversation_id(self, turns: list[dict]) -> str:
        """Generate deterministic conversation ID from turns."""
        conversation_text = "".join(
            f"{turn['speaker']}:{turn['utterance']}" for turn in turns
        )
        return hashlib.sha256(conversation_text.encode()).hexdigest()[:16]

    def _clean_text(self, text: str) -> str:
        """Remove invisible Unicode characters and normalise whitespace."""
        if not text:
            return ""
        cleaned = "".join(
            char
            for char in text
            if unicodedata.category(char) not in ("Cc", "Cf", "Co", "Cs", "Cn")
            or char in "\n\t "
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
        for turn in turns:
            if not all(k in turn for k in ["speaker", "utterance"]):
                return False
            if not turn.get("speaker", "").strip():
                return False
            if not self._is_valid_text(turn.get("utterance", "")):
                return False
            if not isinstance(turn.get("overlap", 0), (int, float)):
                return False
        return True

    def validate_json_output(self, text: str) -> dict | None:
        """Validate and parse JSON output from LLM."""
        try:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if not json_match:
                return None
            parsed = json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            return None

        turns = parsed.get("turns") if isinstance(parsed, dict) else None
        if not isinstance(turns, list) or len(turns) < _MIN_TURNS:
            return None
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

            generated_texts = self._vllm_model.generate(current_prompts)

            next_prompts: list[str] = []
            next_indices: list[int] = []

            for local_idx, (text, original_idx) in enumerate(
                zip(generated_texts, current_indices, strict=False)
            ):
                validated = self.validate_json_output(text)

                if validated:
                    validated_outputs[original_idx] = validated
                else:
                    next_prompts.append(current_prompts[local_idx])
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
            }
        else:
            msg = "No prompt source specified"
            raise ValueError(msg)

        entry_chat = []
        for role in ["system", "user", "assistant"]:
            if role not in prompt:
                continue
            entry_chat.append({"role": role, "content": prompt[role]})

        entry_prompt = self.tokenizer.apply_chat_template(
            entry_chat, **self.chat_template_params
        )

        if isinstance(entry_prompt, list):
            entry_prompt = self.tokenizer.decode(
                entry_prompt, skip_special_tokens=False
            )

        return entry_prompt

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ANN401, ARG002
        """Initialise the VLLMModel engine and tokenizer on the GPU worker."""
        self._vllm_model.setup()
        self.tokenizer = self._vllm_model.get_tokenizer()

    def _ensure_model(self) -> None:
        """Lazy-init fallback when setup() was not called by the executor."""
        if self._vllm_model._llm is None:
            self.setup()

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
        if not tasks:
            return []

        self._ensure_model()

        entry_prompts = [self.get_entry_prompt(t.data) for t in tasks]
        validated_outputs = self.generate_batch_with_retry(
            entry_prompts, max_retry_rounds=5
        )

        output_tasks: list[AudioTask] = []
        for i, (task, output_generation) in enumerate(
            zip(tasks, validated_outputs, strict=False)
        ):
            if output_generation is None:
                logger.warning(f"Skipping failed generation {i + 1}")
                continue

            try:
                conversation_id = self.generate_conversation_id(
                    output_generation["turns"]
                )
                topic = task.data.get("topic", "unknown")

                for turn_idx, turn in enumerate(output_generation["turns"]):
                    output_tasks.append(
                        AudioTask(
                            data={
                                "conversation_id": conversation_id,
                                "turn_index": turn_idx,
                                "speaker": turn["speaker"],
                                "utterance": turn["utterance"],
                                "overlap": turn.get("overlap", 0.0),
                                "topic": topic,
                            },
                            task_id=f"{task.task_id}_conv_{conversation_id}_t{turn_idx}",
                            dataset_name=task.dataset_name,
                        )
                    )

                logger.info(
                    f"[vLLM] Generated conversation {conversation_id[:8]} "
                    f"with {len(output_generation['turns'])} turns"
                )
            except Exception as e:  # noqa: BLE001 -- skip a bad generation without failing the batch
                logger.error(f"Failed to process output {i + 1}: {e}")

        return output_tasks
