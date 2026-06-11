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
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.models.qwen_asr import QwenASR
from nemo_curator.stages.audio.pipeline_utils import LANG_CODE_TO_NAME as _LANG_CODE_TO_NAME
from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

QWEN3_ASR_0_6B_LANGS: frozenset[str] = frozenset({
    "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it",
    "ko", "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv",
    "da", "fi", "pl", "cs", "fil", "fa", "el", "hu", "mk", "ro",
})


@dataclass
class InferenceQwenASRStage(ProcessingStage[AudioTask, AudioTask]):
    """Audio inference using Qwen3-ASR via the ``qwen_asr`` library (vLLM backend).

    Expects each ``AudioTask.data`` to carry:

    - ``waveform``: 1-D mono numpy float32 array (any sample rate)
    - ``sampling_rate``: int

    When ``run_only_if_key`` is set, the stage only runs inference on
    tasks where ``task.data[run_only_if_key]`` starts with
    ``run_only_if_prefix`` (default ``"Hallucination"``).  Non-matching
    tasks pass through unchanged.

    Args:
        model_id: HuggingFace model identifier or local path.
        source_lang_key: Key holding the per-sample source language code
            or name (for example ``"en"`` or ``"English"``).
        pred_text_key: Key where the predicted text is stored.
        language_key: Key where the detected language is stored.
        run_only_if_key: If set, only run inference on tasks where
            ``task.data[run_only_if_key]`` starts with ``run_only_if_prefix``.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        max_new_tokens: Maximum tokens to generate per sample.
        max_inference_batch_size: Batch size for internal vLLM batching.
        keep_waveform: When True the waveform stays on the task (needed when a
            downstream stage re-uses it, e.g. a recovery ASR stage).
    """

    name: str = "QwenASR_inference"
    model_id: str = "Qwen/Qwen3-ASR-0.6B"
    source_lang_key: str = "source_lang"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sampling_rate"
    pred_text_key: str = "asr_prediction"
    language_key: str = "asr_language"
    context_key: str | None = None
    run_only_if_key: str | None = None
    run_only_if_prefix: str = "Hallucination"
    notes_key: str = "additional_notes"
    gpu_memory_utilization: float = 0.7
    max_new_tokens: int = 4096
    max_inference_batch_size: int = 128
    keep_waveform: bool = False
    skip_if_output_exists: bool = False
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 128

    def __post_init__(self) -> None:
        self._model: QwenASR | None = None

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    def _create_model(self) -> QwenASR:
        return QwenASR(
            model_id=self.model_id,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_new_tokens=self.max_new_tokens,
            max_inference_batch_size=self.max_inference_batch_size,
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
        logger.info("QwenASR model ready on node")

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
        return [], [self.pred_text_key, self.language_key]

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, task: AudioTask) -> AudioTask:
        msg = "InferenceQwenASRStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:  # noqa: C901, PLR0912
        if len(tasks) == 0:
            return []

        if self._model is None:
            msg = "Model not initialized — setup() was not called"
            raise RuntimeError(msg)

        for task in tasks:
            task.data.setdefault(self.pred_text_key, "")
            task.data.setdefault(self.language_key, "")

        if self.run_only_if_key:
            run_indices = [
                i for i, t in enumerate(tasks)
                if str(t.data.get(self.run_only_if_key, "")).startswith(self.run_only_if_prefix)
            ]
        else:
            run_indices = list(range(len(tasks)))

        had_candidates = bool(run_indices)
        if self.skip_if_output_exists:
            run_indices = [i for i in run_indices if not tasks[i].data.get(self.pred_text_key)]

        if not run_indices:
            if not self.keep_waveform:
                for task in tasks:
                    task.data.pop(self.waveform_key, None)
            if not had_candidates:
                reason = "none matched run_only_if_key"
            elif self.skip_if_output_exists:
                reason = "output already exists"
            else:
                reason = "none matched run_only_if_key"
            logger.info(f"QwenASR: skipped entire batch of {len(tasks)} ({reason})")
            return tasks

        # Filter out samples with unsupported languages using the model's ISO-code list.
        eligible_indices: list[int] = []
        for i in run_indices:
            if self.source_lang_key:
                code = str(tasks[i].data.get(self.source_lang_key, "") or "").strip().lower()
                if code not in QWEN3_ASR_0_6B_LANGS:
                    set_note(tasks[i].data, self.name, f"skipped (unsupported language: {code})", self.notes_key)
                    set_note(tasks[i].data, self.pred_text_key, f"lang_not_supported:{code}", self.notes_key)
                    continue
            eligible_indices.append(i)

        if not eligible_indices:
            if not self.keep_waveform:
                for task in tasks:
                    task.data.pop(self.waveform_key, None)
            logger.info(f"QwenASR: skipped entire batch of {len(tasks)} (no eligible samples)")
            return tasks

        waveforms = [tasks[i].data[self.waveform_key] for i in eligible_indices]
        sample_rates = [tasks[i].data[self.sample_rate_key] for i in eligible_indices]
        contexts = (
            [tasks[i].data.get(self.context_key, "") for i in eligible_indices]
            if self.context_key else None
        )
        languages: list[str | None] | None = None
        if self.source_lang_key:
            languages = [
                _LANG_CODE_TO_NAME.get(code, code) if code else None
                for code in (tasks[i].data.get(self.source_lang_key) for i in eligible_indices)
            ]

        pred_texts, detected_langs = self._model.generate(waveforms, sample_rates, contexts, languages)

        for idx, pred, lang in zip(eligible_indices, pred_texts, detected_langs, strict=True):
            tasks[idx].data[self.pred_text_key] = pred
            tasks[idx].data[self.language_key] = lang

        if not self.keep_waveform:
            for task in tasks:
                task.data.pop(self.waveform_key, None)

        lang_skipped = len(run_indices) - len(eligible_indices)
        skipped = len(tasks) - len(run_indices)
        logger.info(
            f"QwenASR: generated {len(eligible_indices)} predictions, "
            f"skipped {skipped} (not hallucinated), {lang_skipped} (unsupported language)"
        )
        return tasks
