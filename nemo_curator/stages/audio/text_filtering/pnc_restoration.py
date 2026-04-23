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
from typing import Any

from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.models.qwen_text_llm import QwenTextLLM
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class PnCRestorationStage(ProcessingStage[AudioTask, AudioTask]):
    """Restore punctuation and capitalisation using a text-only LLM.

    Two-step process per text entry:

    1. **Completeness check** – ask the model whether the text is a
       complete sentence (via ``completeness_prompt``).
    2. **PnC restoration** – if complete, send the text with
       ``pnc_prompt`` and write the restored output to
       ``output_text_key`` (default ``"pnc_text"``).  If incomplete,
       copy the original ``text_key`` value as-is.

    Entries that are already flagged (non-empty ``skip_me_key``) or
    whose ``text_key`` is empty are passed through unchanged with
    ``output_text_key`` set to the original text.

    Uses ``QwenTextLLM`` (vLLM-based) for batched GPU inference.

    Args:
        model_id: HuggingFace model identifier for the text LLM.
        text_key: Input field containing cleaned text.
        output_text_key: Output field for PnC-restored text.
        skip_me_key: Field that flags entries to skip.
        completeness_prompt: Prompt template with ``{text}`` placeholder
            for the completeness check.
        pnc_prompt: Prompt template with ``{text}`` placeholder for
            PnC restoration.
        system_prompt: Optional system prompt for the LLM.
        max_model_len: Maximum context length for vLLM.
        max_num_seqs: Maximum concurrent sequences in vLLM.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
            ``None`` means auto-detect.
        max_output_tokens: Maximum tokens to generate per sample.
        temperature: Sampling temperature (0.0 = greedy).
        top_k: Top-k sampling parameter.
        prep_workers: Thread-pool size for parallel prompt preprocessing.
    """

    name: str = "PnCRestoration"
    model_id: str = "Qwen/Qwen3.5-35B-A3B"
    text_key: str = "cleaned_text"
    output_text_key: str = "pnc_text"
    skip_me_key: str = "_skip_me"
    completeness_prompt: str = (
        "Is the following text a complete sentence? Answer only 'yes' or 'no'.\n\nText: {text}"
    )
    pnc_prompt: str = (
        "Restore proper punctuation and capitalization to the following text. "
        "Output only the corrected text, nothing else.\n\nText: {text}"
    )
    system_prompt: str | None = None
    max_model_len: int = 8192
    max_num_seqs: int = 64
    gpu_memory_utilization: float = 0.95
    tensor_parallel_size: int | None = None
    max_output_tokens: int = 512
    temperature: float = 0.0
    top_k: int = 1
    prep_workers: int = 8
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 64

    def __post_init__(self) -> None:
        self._model: QwenTextLLM | None = None
        tp = self.tensor_parallel_size
        if tp and tp > 0:
            self.resources = Resources(gpus=float(tp))

    def _create_model(self) -> QwenTextLLM:
        return QwenTextLLM(
            model_id=self.model_id,
            completeness_prompt=self.completeness_prompt,
            pnc_prompt=self.pnc_prompt,
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
        self._model = self._create_model()
        self._model.setup()
        logger.info("PnCRestoration model ready on node")

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
        return [], [self.text_key, self.skip_me_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_text_key]

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, task: AudioTask) -> AudioTask:
        msg = "PnCRestorationStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if not tasks:
            return []

        if self._model is None:
            msg = "Model not initialized — setup() was not called"
            raise RuntimeError(msg)

        eligible_indices: list[int] = []
        eligible_texts: list[str] = []

        for i, task in enumerate(tasks):
            skip = task.data.get(self.skip_me_key, "")
            text = task.data.get(self.text_key, "")
            if skip or not text.strip():
                task.data[self.output_text_key] = text
            else:
                eligible_indices.append(i)
                eligible_texts.append(text)

        if not eligible_indices:
            logger.info("PnCRestoration: all {} tasks skipped (flagged or empty)", len(tasks))
            return tasks

        is_complete, pnc_texts = self._model.generate(eligible_texts)

        for idx, complete, pnc_text in zip(eligible_indices, is_complete, pnc_texts):
            tasks[idx].data[self.output_text_key] = pnc_text

        n_restored = sum(is_complete)
        logger.info(
            "PnCRestoration: {}/{} restored, {}/{} kept as-is",
            n_restored, len(eligible_indices),
            len(eligible_indices) - n_restored, len(eligible_indices),
        )
        return tasks
