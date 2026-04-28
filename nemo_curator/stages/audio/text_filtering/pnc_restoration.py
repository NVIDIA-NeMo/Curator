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
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

from nemo_curator.models.qwen_text_llm import QwenTextLLM
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_DEFAULT_PNC_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "pnc_prompt.md"


@dataclass
class PnCRestorationStage(ProcessingStage[AudioTask, AudioTask]):
    """Restore punctuation and capitalisation using a text-only LLM.

    Two-step process per text entry:

    1. **Completeness check** - ask the model whether the text is a
       complete sentence (via ``completeness_prompt``).
    2. **PnC restoration** - if complete, send the text with
       ``pnc_prompt`` and write the restored output to
       ``output_text_key`` (default ``"pnc_text"``).  If incomplete,
       copy the original ``text_key`` value as-is.

    Entries that are already flagged (non-empty ``skip_me_key``) or
    whose ``text_key`` is empty are passed through unchanged with
    ``output_text_key`` set to the original text.

    Uses ``QwenTextLLM`` (vLLM-based) for batched GPU inference.

    Resource requirements:
        - **GPU VRAM**: ~20 GB for Qwen3.5-35B-A3B-FP8. Requires 1x A100-40GB
          or 1x A100-80GB. Use ``tensor_parallel_size=2`` for 2x A100-40GB.
        - **Throughput**: ~100-200 samples/min on A100-80GB with
          ``batch_size=64`` (depends on text length).
        - **Model download**: ~18 GB on first run (cached via HuggingFace Hub).

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
        batch_size: Number of tasks per ``process_batch`` call.
    """

    name: str = "PnCRestoration"
    model_id: str = "Qwen/Qwen3.5-35B-A3B-FP8"
    text_key: str = "cleaned_text"
    output_text_key: str = "pnc_text"
    skip_me_key: str = "_skip_me"
    completeness_prompt: str = (
        "The following text is a transcript segment from an audio recording. "
        "It may be a complete, self-contained utterance or thought, "
        "or it may be cut off mid-sentence or mid-idea.\n\n"
        'Determine if the text is complete and self-contained (i.e., not cut off). '
        'Answer only "yes" or "no".\n\n'
        "Text: {text}"
    )
    pnc_prompt: str | None = None
    pnc_prompt_file: str | None = None
    system_prompt: str | None = None
    max_model_len: int = 4096
    max_num_seqs: int = 16
    gpu_memory_utilization: float = 0.95
    tensor_parallel_size: int | None = None
    max_output_tokens: int = 512
    temperature: float = 0.0
    top_k: int = 1
    prep_workers: int = 8
    batch_size: int = 64
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))

    def _resolve_pnc_prompt(self) -> str:
        if self.pnc_prompt:
            return self.pnc_prompt
        path = Path(self.pnc_prompt_file) if self.pnc_prompt_file else _DEFAULT_PNC_PROMPT_PATH
        logger.info("PnCRestoration: loading prompt from {}", path)
        return path.read_text(encoding="utf-8").strip()

    def __post_init__(self) -> None:
        self._model: QwenTextLLM | None = None
        tp = self.tensor_parallel_size
        if tp and tp > 0:
            self.resources = Resources(gpus=float(tp))

    def _create_model(self) -> QwenTextLLM:
        pnc_prompt = self._resolve_pnc_prompt()
        return QwenTextLLM(
            model_id=self.model_id,
            completeness_prompt=self.completeness_prompt,
            pnc_prompt=pnc_prompt,
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

    def _partition_tasks(self, tasks: list[AudioTask]) -> tuple[list[int], list[str]]:
        """Separate eligible tasks from those that are skipped or empty."""
        eligible_indices: list[int] = []
        eligible_texts: list[str] = []
        for i, task in enumerate(tasks):
            skip = task.data.get(self.skip_me_key, "")
            text = task.data.get(self.text_key, "")
            if skip:
                task.data[self.output_text_key] = ""
            elif not text.strip():
                task.data[self.output_text_key] = text
            else:
                eligible_indices.append(i)
                eligible_texts.append(text)
        return eligible_indices, eligible_texts

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task.task_id} missing required columns for {type(self).__name__}: {self.inputs()}"
                raise ValueError(msg)

        if self._model is None:
            msg = "Model not initialized — setup() was not called"
            raise RuntimeError(msg)

        eligible_indices, eligible_texts = self._partition_tasks(tasks)

        if not eligible_indices:
            logger.info("PnCRestoration: all {} tasks skipped (flagged or empty)", len(tasks))
            return tasks

        all_complete: list[bool] = []
        all_pnc: list[str] = []
        for start in range(0, len(eligible_texts), self.batch_size):
            chunk = eligible_texts[start : start + self.batch_size]
            is_complete, pnc_texts = self._model.generate(chunk)
            all_complete.extend(is_complete)
            all_pnc.extend(pnc_texts)

        for idx, _complete, pnc_text in zip(eligible_indices, all_complete, all_pnc, strict=False):
            tasks[idx].data[self.output_text_key] = pnc_text

        n_restored = sum(all_complete)
        logger.info(
            "PnCRestoration: {}/{} restored, {}/{} kept as-is",
            n_restored, len(eligible_indices),
            len(eligible_indices) - n_restored, len(eligible_indices),
        )
        return tasks
