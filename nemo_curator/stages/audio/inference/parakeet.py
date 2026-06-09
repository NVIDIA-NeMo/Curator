# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone NVIDIA Parakeet-TDT v3 inference stage (in-memory waveforms)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from loguru import logger

from nemo_curator.stages.audio.inference.asr_nemo import NemoASRModel
from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

PARAKEET_TDT_0_6B_V3_LANGS: frozenset[str] = frozenset({
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
    "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro",
    "ru", "sk", "sl", "sv", "uk",
})


@dataclass
class InferenceParakeetStage(ProcessingStage[AudioTask, AudioTask]):
    """Audio transcription using NVIDIA Parakeet-TDT v3 (in-memory waveforms).

    Designed for single-language-group invocations where every sample belongs to
    the Parakeet language family (WHISPER_PRIMARY_LANGUAGE_CODES recovery group).
    Waveforms are fed directly from memory — no temp ``.wav`` files are written.

    Args:
        model_id: NeMo / HuggingFace model identifier (e.g. ``"nvidia/parakeet-tdt-0.6b-v3"``).
        inference_batch_size: Batch size passed to Parakeet ``ASRModel.transcribe``.
        waveform_key: Task data key for the mono float32 numpy waveform.
        sample_rate_key: Task data key for the integer sample rate.
        pred_text_key: Output key for the predicted transcription.
        language_key: Output key storing the source language code (passed through).
        notes_key: Top-level key used for ``additional_notes`` metadata.
        source_lang_key: Task data key holding the per-sample ISO language code.
        keep_waveform: When True the waveform stays on the task so a downstream
            stage can re-use it.
        num_workers_override: Fixed Ray actor count. None = autoscaler decides.
    """

    name: str = "Parakeet_inference"
    model_id: str = "nvidia/parakeet-tdt-0.6b-v3"
    inference_batch_size: int = 16
    waveform_key: str = "waveform"
    sample_rate_key: str = "sampling_rate"
    pred_text_key: str = "asr_prediction"
    language_key: str = "asr_language"
    notes_key: str = "additional_notes"
    source_lang_key: str = "source_lang"
    keep_waveform: bool = False
    skip_if_output_exists: bool = False
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 128
    _wrapper: NemoASRModel | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Scaling hooks
    # ------------------------------------------------------------------

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _create_wrapper(self) -> NemoASRModel:
        return NemoASRModel(
            model_name=self.model_id,
            inference_batch_size=self.inference_batch_size,
        )

    @staticmethod
    def _cuda_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        self._wrapper = self._create_wrapper()
        self._wrapper.setup_on_node()
        logger.info(f"Parakeet model pre-warmed on node: {self.model_id}")

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._wrapper is None:
            self._wrapper = self._create_wrapper()
        if self._wrapper.asr_model is None:
            self._wrapper.setup(device=self._cuda_device())

    def teardown(self) -> None:
        if self._wrapper is not None:
            self._wrapper.teardown()
            self._wrapper = None

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
        msg = "InferenceParakeetStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        if self._wrapper is None or self._wrapper.asr_model is None:
            msg = "Parakeet model not initialized — setup() was not called"
            raise RuntimeError(msg)

        for task in tasks:
            task.data.setdefault(self.pred_text_key, "")
            task.data.setdefault(self.language_key, "")

        eligible_indices: list[int] = []
        output_exists_skipped = 0
        for i, task in enumerate(tasks):
            if self.skip_if_output_exists and task.data.get(self.pred_text_key):
                output_exists_skipped += 1
                continue
            lang = str(task.data.get(self.source_lang_key, "") or "").strip().lower()
            if lang not in PARAKEET_TDT_0_6B_V3_LANGS:
                set_note(task.data, self.name, f"skipped (unsupported language: {lang})", self.notes_key)
                set_note(task.data, self.pred_text_key, f"lang_not_supported:{lang}", self.notes_key)
            else:
                eligible_indices.append(i)

        lang_skipped = len(tasks) - len(eligible_indices) - output_exists_skipped
        if not eligible_indices:
            if not self.keep_waveform:
                for task in tasks:
                    task.data.pop(self.waveform_key, None)
            if output_exists_skipped:
                logger.info(f"Parakeet: skipped entire batch of {len(tasks)} (output already exists)")
            else:
                logger.info(f"Parakeet: skipped entire batch of {len(tasks)} (no supported languages)")
            return tasks

        eligible_tasks = [tasks[i] for i in eligible_indices]
        waveforms = [t.data[self.waveform_key] for t in eligible_tasks]
        sample_rates = [t.data[self.sample_rate_key] for t in eligible_tasks]

        pred_texts = self._wrapper.transcribe_waveforms(waveforms, sample_rates)

        for task_idx, pred in zip(eligible_indices, pred_texts, strict=True):
            tasks[task_idx].data[self.pred_text_key] = pred
            tasks[task_idx].data[self.language_key] = str(tasks[task_idx].data.get(self.source_lang_key, ""))

        if not self.keep_waveform:
            for task in tasks:
                task.data.pop(self.waveform_key, None)

        logger.info(
            f"Parakeet: generated {len(eligible_indices)} predictions, "
            f"skipped {lang_skipped} (unsupported language)"
        )
        return tasks
