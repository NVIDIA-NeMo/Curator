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

import time
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


@dataclass
class InferenceQwenASRStage(ProcessingStage[AudioTask, AudioTask]):
    """Audio inference using Qwen3-ASR via the ``qwen_asr`` library (vLLM backend).

    Expects each ``AudioTask.data`` to carry:

    - ``waveform``: 1-D mono numpy float32 array (any sample rate)
    - ``sample_rate``: int

    When ``run_only_if_key`` is set, the stage only runs inference on
    tasks where ``task.data[run_only_if_key]`` starts with
    ``run_only_if_prefix`` (default ``"Hallucination"``).  Non-matching
    tasks pass through unchanged.

    Resource requirements:
        - **GPU VRAM**: ~2 GB for Qwen3-ASR-0.6B. Fits comfortably on any
          modern GPU (even 8 GB consumer cards).
        - **Throughput**: ~200-400 audio-seconds/GPU-second on A100 with
          ``batch_size=128``.
        - **Model download**: ~1.2 GB on first run (cached via HuggingFace Hub).

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
    """

    name: str = "QwenASR_inference"
    model_id: str = "Qwen/Qwen3-ASR-0.6B"
    source_lang_key: str = "source_lang"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    pred_text_key: str = "qwen3_asr_prediction"
    language_key: str = "qwen3_asr_language"
    context_key: str | None = None
    run_only_if_key: str | None = None
    run_only_if_prefix: str = "Hallucination"
    notes_key: str = "additional_notes"
    gpu_memory_utilization: float = 0.95
    max_new_tokens: int = 4096
    max_inference_batch_size: int = 128
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 128
    _supported_langs: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self._model: QwenASR | None = None

    def _get_supported_languages(self) -> set[str]:
        """Return QwenASR-supported language names when the package exposes them."""
        if not self._supported_langs:
            try:
                from qwen_asr.inference.utils import SUPPORTED_LANGUAGES

                self._supported_langs = set(SUPPORTED_LANGUAGES)
            except (ImportError, AttributeError):
                return set()
        return self._supported_langs

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        if self.num_workers_override is None:
            return {}
        return {"num_workers": self.num_workers_override}

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
        """Pre-download model weights once per node (no GPU allocation).

        Workers don't preserve ``_model`` across serialization, so
        ``setup()`` creates the vLLM engine independently.
        """
        try:
            from huggingface_hub import snapshot_download

            prefetch_t0 = time.perf_counter()
            snapshot_download(self.model_id)
            logger.info(
                "QwenASR weights cached on node for {} in {:.3f}s",
                self.model_id,
                time.perf_counter() - prefetch_t0,
            )
        except Exception:  # noqa: BLE001
            logger.warning("QwenASR: snapshot_download failed; setup() will download")

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._model is None:
            self._model = self._create_model()
            self._model.setup()
            logger.info("QwenASR model ready on worker")

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
        return [], [self.pred_text_key, self.language_key, self.notes_key]

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, task: AudioTask) -> AudioTask:
        msg = "InferenceQwenASRStage only supports process_batch"
        raise NotImplementedError(msg)

    def _select_run_indices(self, tasks: list[AudioTask]) -> list[int]:
        """Return indices of tasks that should be processed by this stage."""
        if self.run_only_if_key:
            return [
                i for i, t in enumerate(tasks)
                if str(t.data.get(self.run_only_if_key, "")).startswith(self.run_only_if_prefix)
            ]
        return list(range(len(tasks)))

    def _resolve_languages(self, tasks: list[AudioTask], indices: list[int]) -> list[str | None] | None:
        """Map source_lang codes to full language names for the selected indices."""
        if not self.source_lang_key:
            return None
        return [
            _LANG_CODE_TO_NAME.get(code, code) if code else None
            for code in (tasks[i].data.get(self.source_lang_key) for i in indices)
        ]

    def _filter_supported_languages(self, tasks: list[AudioTask], indices: list[int]) -> list[int]:
        """Drop tasks whose source language is known unsupported by QwenASR."""
        supported = self._get_supported_languages()
        if not self.source_lang_key or not supported:
            return indices

        eligible: list[int] = []
        for idx in indices:
            code = tasks[idx].data.get(self.source_lang_key, "")
            lang_name = _LANG_CODE_TO_NAME.get(code, code) if code else None
            if lang_name and lang_name not in supported:
                set_note(
                    tasks[idx].data,
                    self.name,
                    f"skipped unsupported language: {lang_name}",
                    self.notes_key,
                )
                continue
            eligible.append(idx)
        return eligible

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

        for task in tasks:
            task.data.setdefault(self.pred_text_key, "")
            task.data.setdefault(self.language_key, "")

        run_indices = self._select_run_indices(tasks)

        if not run_indices:
            skipped_audio_seconds = sum(
                float(task.data[self.waveform_key].shape[0]) / float(task.data[self.sample_rate_key])
                for task in tasks
                if task.data.get(self.sample_rate_key)
                and task.data.get(self.waveform_key) is not None
                and getattr(task.data.get(self.waveform_key), "size", 0) > 0
            )
            skipped_waveform_bytes = sum(
                float(getattr(task.data.get(self.waveform_key), "nbytes", 0))
                for task in tasks
                if task.data.get(self.waveform_key) is not None
            )
            for task in tasks:
                task.data.pop(self.waveform_key, None)
            self._log_metrics({
                "utterances_input": float(len(tasks)),
                "utterances_selected": 0.0,
                "utterances_skipped": float(len(tasks)),
                "audio_duration_s": 0.0,
                "skipped_audio_duration_s": skipped_audio_seconds,
                "waveform_bytes": 0.0,
                "skipped_waveform_bytes": skipped_waveform_bytes,
                "output_chars": 0.0,
                "inference_time_s": 0.0,
                "inference_time": 0.0,
            })
            logger.info(f"QwenASR: skipped entire batch of {len(tasks)} (none matched run_only_if_key)")
            return tasks

        eligible_indices = self._filter_supported_languages(tasks, run_indices)
        unsupported_indices = set(run_indices) - set(eligible_indices)

        if not eligible_indices:
            skipped_indices = set(range(len(tasks)))
            skipped_audio_seconds = sum(
                float(tasks[i].data[self.waveform_key].shape[0]) / float(tasks[i].data[self.sample_rate_key])
                for i in skipped_indices
                if tasks[i].data.get(self.sample_rate_key)
                and tasks[i].data.get(self.waveform_key) is not None
                and getattr(tasks[i].data.get(self.waveform_key), "size", 0) > 0
            )
            skipped_waveform_bytes = sum(
                float(getattr(tasks[i].data.get(self.waveform_key), "nbytes", 0))
                for i in skipped_indices
                if tasks[i].data.get(self.waveform_key) is not None
            )
            for task in tasks:
                task.data.pop(self.waveform_key, None)
            self._log_metrics({
                "utterances_input": float(len(tasks)),
                "utterances_selected": 0.0,
                "utterances_skipped": float(len(tasks)),
                "utterances_unsupported_language": float(len(unsupported_indices)),
                "audio_duration_s": 0.0,
                "skipped_audio_duration_s": skipped_audio_seconds,
                "waveform_bytes": 0.0,
                "skipped_waveform_bytes": skipped_waveform_bytes,
                "output_chars": 0.0,
                "inference_time_s": 0.0,
                "inference_time": 0.0,
            })
            logger.info(f"QwenASR: skipped entire batch of {len(tasks)} (no eligible samples)")
            return tasks

        waveforms = [tasks[i].data[self.waveform_key] for i in eligible_indices]
        sample_rates = [tasks[i].data[self.sample_rate_key] for i in eligible_indices]
        contexts = (
            [tasks[i].data.get(self.context_key, "") for i in eligible_indices]
            if self.context_key else None
        )
        languages = self._resolve_languages(tasks, eligible_indices)

        inference_t0 = time.perf_counter()
        pred_texts, detected_langs = self._model.generate(waveforms, sample_rates, contexts, languages)
        inference_elapsed = time.perf_counter() - inference_t0

        for idx, pred, lang in zip(eligible_indices, pred_texts, detected_langs, strict=True):
            tasks[idx].data[self.pred_text_key] = pred
            tasks[idx].data[self.language_key] = lang

        skipped = len(tasks) - len(eligible_indices)
        skipped_indices = set(range(len(tasks))) - set(eligible_indices)
        self._log_metrics({
            "utterances_input": float(len(tasks)),
            "utterances_selected": float(len(eligible_indices)),
            "utterances_skipped": float(skipped),
            "utterances_unsupported_language": float(len(unsupported_indices)),
            "audio_duration_s": sum(
                float(w.shape[0]) / float(sr)
                for w, sr in zip(waveforms, sample_rates, strict=False)
                if sr and w is not None and getattr(w, "size", 0) > 0
            ),
            "skipped_audio_duration_s": sum(
                float(tasks[i].data[self.waveform_key].shape[0]) / float(tasks[i].data[self.sample_rate_key])
                for i in skipped_indices
                if tasks[i].data.get(self.sample_rate_key)
                and tasks[i].data.get(self.waveform_key) is not None
                and getattr(tasks[i].data.get(self.waveform_key), "size", 0) > 0
            ),
            "waveform_bytes": sum(float(getattr(w, "nbytes", 0)) for w in waveforms if w is not None),
            "skipped_waveform_bytes": sum(
                float(getattr(tasks[i].data.get(self.waveform_key), "nbytes", 0))
                for i in skipped_indices
                if tasks[i].data.get(self.waveform_key) is not None
            ),
            "output_chars": float(sum(len(text) for text in pred_texts)),
            "inference_time_s": inference_elapsed,
            "inference_time": inference_elapsed,
        })
        for task in tasks:
            task.data.pop(self.waveform_key, None)
        logger.info(
            f"QwenASR: generated {len(eligible_indices)} predictions, "
            f"skipped {skipped} ({len(unsupported_indices)} unsupported language)"
        )
        return tasks
