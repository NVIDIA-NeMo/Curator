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

"""vLLM-based conversation generation stages for audio data curation."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from string import Template
from typing import Any

import yaml
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, DocumentBatch


class vLLMInference(ProcessingStage[AudioTask, AudioTask]):
    """Generate multi-speaker conversations via vLLM.

    Receives ``AudioTask`` topic entries and returns ``AudioTask`` objects —
    one per conversation turn. All turns of one conversation share the same
    ``conversation_id``.

    Use ``process_batch`` for efficient GPU utilisation: it collects all
    topic prompts and runs vLLM inference in a single batched call with
    retry logic for failed outputs.

    Args:
        prompt: Static ``"role:content"`` prompt string.
        prompt_field: Key in each entry containing a ``"role:content"`` string.
        prompt_file: Path to a YAML file with ``role: content`` mappings.
            Supports ``$topic`` substitution from each entry's ``topic`` field.
        model: Kwargs forwarded to ``vllm.LLM()``.
        inference: Kwargs forwarded to ``vllm.SamplingParams()``.
        apply_chat_template: Kwargs forwarded to
            ``tokenizer.apply_chat_template()``.
    """

    name = "vLLMInference"
    resources = Resources(gpus=1)

    def __init__(
        self,
        prompt: str | None = None,
        prompt_field: str | None = None,
        prompt_file: str | None = None,
        model: dict | None = None,
        inference: dict | None = None,
        apply_chat_template: dict | None = None,
        **kwargs,
    ):
        super().__init__()

        if sum([prompt is not None, prompt_field is not None, prompt_file is not None]) != 1:
            raise ValueError(
                "Exactly one of prompt, prompt_field, or prompt_file must be specified"
            )

        self.prompt = prompt
        self.prompt_field = prompt_field
        self.prompt_file = prompt_file
        self.model_params = model or {}
        self.inference_params = inference or {}
        self.chat_template_params = apply_chat_template or {}

        if self.prompt_file:
            with open(self.prompt_file, "r") as f:
                self.prompt_data = yaml.safe_load(f)
        else:
            self.prompt_data = None

        self.llm = None
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
        if not cleaned or len(cleaned) < 2:
            return False
        return any(c.isalpha() for c in cleaned)

    def validate_json_output(self, text: str) -> dict | None:
        """Validate and parse JSON output from LLM."""
        try:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if not json_match:
                return None

            parsed = json.loads(json_match.group())

            if "turns" not in parsed or not isinstance(parsed["turns"], list):
                return None

            if len(parsed["turns"]) < 2:
                return None

            for turn in parsed["turns"]:
                if not all(k in turn for k in ["speaker", "utterance"]):
                    return None
                if not turn.get("speaker", "").strip():
                    return None
                if not self._is_valid_text(turn.get("utterance", "")):
                    return None
                overlap = turn.get("overlap", 0)
                if not isinstance(overlap, (int, float)):
                    return None

            for turn in parsed["turns"]:
                turn["utterance"] = self._clean_text(turn["utterance"])

            return parsed
        except (json.JSONDecodeError, AttributeError):
            return None

    def generate_batch_with_retry(
        self,
        llm: Any,
        prompts: list[str],
        max_retry_rounds: int = 5,
    ) -> list[dict | None]:
        """Generate with retry logic for failed outputs."""
        from vllm import SamplingParams

        current_prompts = prompts.copy()
        current_indices = list(range(len(prompts)))
        validated_outputs: list[dict | None] = [None] * len(prompts)

        for retry_round in range(max_retry_rounds):
            if not current_prompts:
                break

            sampling_params = SamplingParams(**self.inference_params)
            outputs = llm.generate(current_prompts, sampling_params)

            next_prompts: list[str] = []
            next_indices: list[int] = []

            for local_idx, (output, original_idx) in enumerate(
                zip(outputs, current_indices)
            ):
                generated_text = output.outputs[0].text
                validated = self.validate_json_output(generated_text)

                if validated:
                    validated_outputs[original_idx] = validated
                else:
                    next_prompts.append(current_prompts[local_idx])
                    next_indices.append(original_idx)

            if next_prompts:
                current_prompts = next_prompts
                current_indices = next_indices
            else:
                break

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

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002
        """Initialise the vLLM engine and tokenizer on the GPU worker."""
        from transformers import AutoTokenizer
        from vllm import LLM

        self.llm = LLM(**self.model_params)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_params["model"], trust_remote_code=True
        )

    def _ensure_llm(self) -> None:
        """Lazy-init fallback when setup() was not called by the executor."""
        if self.llm is None:
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

        self._ensure_llm()

        entry_prompts = [self.get_entry_prompt(t.data) for t in tasks]
        validated_outputs = self.generate_batch_with_retry(
            self.llm, entry_prompts, max_retry_rounds=5
        )

        output_tasks: list[AudioTask] = []
        for i, (task, output_generation) in enumerate(
            zip(tasks, validated_outputs)
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
            except Exception as e:
                logger.error(f"Failed to process output {i + 1}: {e}")

        return output_tasks


class DocumentToAudioStage(ProcessingStage):
    """Convert a ``DocumentBatch`` to a list of ``AudioTask`` objects."""

    name = "DocumentToAudioStage"

    def process(self, task: Any) -> list[AudioTask]:
        """Convert each row of a DocumentBatch into an individual AudioTask."""
        if isinstance(task, DocumentBatch):
            records = task.data.to_dict(orient="records")
        else:
            records = [task.data] if isinstance(task.data, dict) else task.data

        return [
            AudioTask(
                data=record,
                task_id=f"{task.task_id}_{i}",
                dataset_name=task.dataset_name,
            )
            for i, record in enumerate(records)
        ]


class TopicExpander(ProcessingStage):
    """Fan-out topics into N conversation-generation tasks.

    Args:
        num_conversations: Total conversations to generate.
        seed: Random seed for reproducibility.
    """

    name = "TopicExpander"

    def __init__(
        self,
        num_conversations: int,
        seed: int | None = None,
    ):
        super().__init__()
        self.num_conversations = num_conversations
        self.seed = seed
        self._rng = random.Random(seed)

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}

    def process(self, task: AudioTask) -> list[AudioTask]:
        """Expand one topic task into N conversation-generation tasks."""
        topic = task.data.get("topic", "")
        if not topic and isinstance(task.data, str):
            topic = task.data

        if not topic:
            logger.error("No topic found in input task")
            return []

        output_tasks: list[AudioTask] = []
        for i in range(self.num_conversations):
            output_tasks.append(
                AudioTask(
                    data={
                        "topic": topic,
                        "conversation_index": i,
                    },
                    task_id=f"conversation_{i}",
                    dataset_name=task.dataset_name,
                )
            )

        logger.info(
            f"Expanded topic '{topic}' into "
            f"{self.num_conversations} conversation tasks"
        )
        return output_tasks

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Expand multiple topic tasks into N conversation-generation tasks.

        Collects all unique topics from the input tasks, then randomly
        assigns them across ``num_conversations`` output tasks.
        """
        topics: list[str] = []
        for task in tasks:
            topic = task.data.get("topic", "")
            if not topic and isinstance(task.data, str):
                topic = task.data
            if topic:
                topics.append(topic)

        if not topics:
            logger.error("No topics found in input tasks")
            return []

        dataset_name = tasks[0].dataset_name if tasks else ""

        output_tasks: list[AudioTask] = []
        for i in range(self.num_conversations):
            output_tasks.append(
                AudioTask(
                    data={
                        "topic": self._rng.choice(topics),
                        "conversation_index": i,
                    },
                    task_id=f"conversation_{i}",
                    dataset_name=dataset_name,
                )
            )

        logger.info(
            f"Expanded {len(topics)} topics into "
            f"{self.num_conversations} conversation tasks"
        )
        return output_tasks
