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

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.models.qwen_omni import QwenOmni
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class InferenceQwenOmniStage(ProcessingStage[AudioTask, AudioTask]):
    """Audio inference using Qwen3-Omni via in-process vLLM (thinker-only).

    Expects each ``AudioTask.data`` to carry:

    - ``waveform``: 1-D mono numpy float32 array (any sample rate)
    - ``sample_rate``: int

    These are produced by ``NemoTarShardReaderStage`` which decodes audio
    in memory from NeMo tarred datasets via lhotse/soundfile.

    The stage resamples to 16 kHz internally and passes numpy arrays
    directly to ``qwen_omni_utils`` — no temporary files are created.

    Overrides ``process_batch`` for batched GPU inference with
    thread-pool-parallel preprocessing.

    Args:
        model_id: HuggingFace model identifier.
        prompt_text: User prompt sent alongside the audio.
        system_prompt: Optional system prompt.
        waveform_key: Key in ``AudioTask.data`` for the waveform array.
        sample_rate_key: Key in ``AudioTask.data`` for the sample rate.
        pred_text_key: Key where the predicted text is stored.
        max_model_len: Maximum context length passed to vLLM.
        max_num_seqs: Maximum concurrent sequences in vLLM.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
            ``None`` means auto-detect.
        max_output_tokens: Maximum tokens to generate per sample.
        temperature: Sampling temperature (0.0 = greedy).
        top_k: Top-k sampling parameter.
        prep_workers: Thread-pool size for parallel audio preprocessing.
    """

    name: str = "QwenOmni_inference"
    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    prompt_text: str = "Transcribe the audio."
    followup_prompt: str | None = None
    system_prompt: str | None = None
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    pred_text_key: str = "qwen3_prediction_s1"
    disfluency_text_key: str = "qwen3_prediction_s2"
    max_model_len: int = 32768
    max_num_seqs: int = 32
    gpu_memory_utilization: float = 0.95
    tensor_parallel_size: int | None = None
    max_output_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 1
    prep_workers: int = 8
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 32

    def __post_init__(self) -> None:
        self._model: QwenOmni | None = None
        tp = self.tensor_parallel_size
        if tp and tp > 0:
            self.resources = Resources(gpus=float(tp))

    def _create_model(self) -> QwenOmni:
        return QwenOmni(
            model_id=self.model_id,
            prompt_text=self.prompt_text,
            followup_prompt=self.followup_prompt,
            system_prompt=self.system_prompt,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            prep_workers=self.prep_workers,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        """Pre-download model weights once per node and warm-start vLLM.

        Avoids torch.compile race conditions when multiple workers on
        the same node attempt to JIT-compile at the same time.
        """
        self._model = self._create_model()
        self._model.setup()
        logger.info("QwenOmni model ready on node")

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._model is None:
            self._model = self._create_model()
            self._model.setup()

    def teardown(self) -> None:
        if self._model is not None:
            self._model.teardown()
            self._model = None

    # ------------------------------------------------------------------
    # I/O contract
    # ------------------------------------------------------------------

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        keys = [self.pred_text_key]
        if self.followup_prompt:
            keys.append(self.disfluency_text_key)
        return [], keys

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, task: AudioTask) -> AudioTask:
        msg = "InferenceQwenOmniStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if not tasks:
            return []

        if self._model is None:
            msg = "Model not initialized — setup() was not called"
            raise RuntimeError(msg)

        waveforms = [t.data[self.waveform_key] for t in tasks]
        sample_rates = [t.data[self.sample_rate_key] for t in tasks]

        pred_texts, disfluency_texts = self._model.generate(waveforms, sample_rates)

        for task, pred, disfl in zip(tasks, pred_texts, disfluency_texts, strict=True):
            task.data[self.pred_text_key] = pred
            if self.followup_prompt:
                task.data[self.disfluency_text_key] = disfl
            task.data.pop(self.waveform_key, None)

        logger.info(f"QwenOmni: generated {len(pred_texts)} predictions (turn2={bool(self.followup_prompt)})")
        return tasks
