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

"""Three-pass transcription cascade using in-process vLLM.

Runs all three passes (ASR, verification, PnC) using a single in-process
Qwen3-Omni vLLM engine -- no external server required.

Pass 1 (audio+text): ASR transcription with number normalization.
Pass 2 (audio+text): Verification of Pass 1 output against audio.
Pass 3 (text-only):  Punctuation and capitalization correction.

Used as a drop-in replacement for the server-based cascade in HIFI pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.models.qwen_omni import QwenOmni
from nemo_curator.stages.audio.request.prompt_template import (
    build_prompt_conversation,
    load_prompt_config,
)
from nemo_curator.stages.audio.request.transcription_cascade import (
    TranscriptionCascadeConfig,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class TranscriptionCascadeInProcessStage(ProcessingStage[AudioTask, AudioTask]):
    """Three-pass transcription cascade using in-process vLLM.

    Expects ``AudioTask.data`` with ``waveform`` and ``sample_rate`` keys
    (produced by ``NemoTarShardReaderStage``).

    Runs three sequential vLLM passes per batch using YAML prompt configs:
      1. ASR transcription → ``qwen3_omni_pred_text``
      2. Audio verification → ``qwen3_omni_verified_text``
      3. Text-only PnC → ``qwen3_llm_corrected_text``

    Args:
        model_id: HuggingFace model identifier for Qwen3-Omni.
        language: Language code for prompt selection (e.g. "En", "Ru").
        waveform_key: Key in AudioTask.data for the waveform array.
        sample_rate_key: Key in AudioTask.data for the sample rate.
        max_model_len: Maximum context length for vLLM.
        max_num_seqs: Maximum concurrent sequences in vLLM.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        max_output_tokens: Maximum tokens to generate per sample.
        temperature: Sampling temperature (0.0 = greedy).
        prep_workers: Thread-pool size for audio preprocessing.
    """

    name: str = "TranscriptionCascadeInProcess"
    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    language: str = "En"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    max_model_len: int = 32768
    max_num_seqs: int = 32
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int | None = None
    max_output_tokens: int = 512
    temperature: float = 0.0
    top_k: int = 1
    prep_workers: int = 8
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 32

    def __post_init__(self) -> None:
        self._model: QwenOmni | None = None
        self._pass_cfgs: list[dict[str, Any]] = []
        tp = self.tensor_parallel_size
        if tp and tp > 0:
            self.resources = Resources(gpus=float(tp))

    def _create_model(self) -> QwenOmni:
        return QwenOmni(
            model_id=self.model_id,
            prompt_text="",
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            prep_workers=self.prep_workers,
        )

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        self._model = self._create_model()
        self._model.setup()
        cascade_cfg = TranscriptionCascadeConfig(language=self.language)
        paths = cascade_cfg.resolve_paths()
        self._pass_cfgs = [load_prompt_config(p) for p in paths]
        logger.info("TranscriptionCascadeInProcess ready (language=%s)", self.language)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._model is None:
            self.setup_on_node()

    def teardown(self) -> None:
        if self._model is not None:
            self._model.teardown()
            self._model = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        output_fields = [cfg["output_field"] for cfg in self._pass_cfgs] if self._pass_cfgs else [
            "qwen3_omni_pred_text", "qwen3_omni_verified_text", "qwen3_llm_corrected_text",
        ]
        return [], output_fields

    def process(self, task: AudioTask) -> AudioTask:
        msg = "TranscriptionCascadeInProcessStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if not tasks or self._model is None:
            return []

        p1_cfg, p2_cfg, p3_cfg = self._pass_cfgs

        waveforms = [t.data[self.waveform_key] for t in tasks]
        sample_rates = [t.data[self.sample_rate_key] for t in tasks]

        # Pass 1: ASR (audio + prompt)
        p1_messages = [build_prompt_conversation(t.data, p1_cfg) for t in tasks]
        p1_results = self._model.generate_from_messages(p1_messages, waveforms, sample_rates)
        for task, text in zip(tasks, p1_results):
            task.data[p1_cfg["output_field"]] = text
        logger.info("Cascade Pass 1 (ASR): %d predictions", len(p1_results))

        # Pass 2: Verification (audio + Pass 1 output in prompt)
        p2_messages = [build_prompt_conversation(t.data, p2_cfg) for t in tasks]
        p2_results = self._model.generate_from_messages(p2_messages, waveforms, sample_rates)
        for task, text in zip(tasks, p2_results):
            task.data[p2_cfg["output_field"]] = text
        logger.info("Cascade Pass 2 (Verify): %d predictions", len(p2_results))

        # Pass 3: PnC (text-only, no audio)
        p3_messages = [build_prompt_conversation(t.data, p3_cfg) for t in tasks]
        p3_results = self._model.generate_from_messages(p3_messages)
        for task, text in zip(tasks, p3_results):
            task.data[p3_cfg["output_field"]] = text
            task.data.pop(self.waveform_key, None)
        logger.info("Cascade Pass 3 (PnC): %d predictions", len(p3_results))

        return tasks
