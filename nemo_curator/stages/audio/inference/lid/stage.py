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

"""Generic adapter-backed spoken-language-identification stage.

The stage owns Curator task I/O, audio normalization, duration limits, resume
semantics, and JSON-safe result storage. Model adapters own only their native
model lifecycle and batch inference. This mirrors the SED stage/adapter split
and lets a YAML pipeline select an LID implementation by import path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from loguru import logger

from nemo_curator.models.audio.lid.base import AudioLIDAdapter, AudioLIDResult
from nemo_curator.stages.audio.common import ensure_mono, ensure_waveform_2d
from nemo_curator.stages.audio.inference.base import AdapterInferenceStage
from nemo_curator.stages.resources import Resources

if TYPE_CHECKING:
    from nemo_curator.tasks import AudioTask


_SKIP_ME_KEY = "_skipme"
_AUDIO_LOAD_ERROR = "audio_load_error"
_WAVEFORM_DIMENSIONS = 2


@dataclass
class AudioLIDInferenceStage(AdapterInferenceStage[AudioLIDAdapter]):
    """Identify spoken language through a YAML-selectable model adapter.

    ``model_id`` is a stable ensemble/output identity, such as
    ``"SpeechBrainLangID"``. It is deliberately not passed to the adapter:
    provider-native checkpoint names belong in ``adapter_kwargs``. A completed
    result is stored at ``task.data[results_key][model_id]`` as a plain mapping
    containing ``language``, ``confidence``, and ``tag``.

    With ``skip_if_output_exists=True``, membership of the stage's own
    ``model_id`` is the completion marker. In particular, an empty language or
    confidence ``0.0`` is a completed model result and is not recomputed.

    Missing, unreadable, empty, and too-short audio receive a completed empty
    result so another run can resume deterministically. Audio preparation
    failures set the shared ``_skipme`` field only when
    ``fail_on_audio_error=True``; a short clip is a valid model miss and never
    sets that shared terminal flag.

    Args:
        adapter_target: Import path of a class implementing ``AudioLIDAdapter``.
        model_id: Stable key used for this model in the shared result mapping.
        tag: Ensemble role recorded with every result.
        sample_rate: Canonical adapter sample rate.
        waveform_key: In-memory waveform key, or ``None`` to load a file.
        sample_rate_key: Source sample-rate key for in-memory waveforms.
        audio_filepath_key: File key used when ``waveform_key`` is ``None``.
        results_key: Shared mapping populated by every ensemble component.
        min_duration_sec: Clips shorter than this receive an empty result.
        max_duration_sec: Maximum leading audio sent to the adapter; non-positive
            values disable truncation.
        skip_if_output_exists: Reuse this model's existing mapping entry.
        fail_on_audio_error: Mark audio preparation failures terminal via
            ``_skipme=audio_load_error``.
        retry_skip_reasons: Exact prior skip reasons that this stage may clear
            before resume/inference. This lets a final selector-owned rejection
            re-enter the ensemble without discarding unrelated upstream skips.
        prefetch_fail_on_error: Whether node-level weight prefetch errors fail
            setup immediately.
        adapter_kwargs: Provider-native adapter constructor arguments.
    """

    adapter_target: str
    model_id: str
    tag: str
    name: str = "AudioLIDInference"

    sample_rate: int = 16000
    waveform_key: str | None = None
    sample_rate_key: str = "sample_rate"
    audio_filepath_key: str = "audio_filepath"
    results_key: str = "lid"

    min_duration_sec: float = 1.0
    max_duration_sec: float = 10.0
    skip_if_output_exists: bool = False
    fail_on_audio_error: bool = False
    retry_skip_reasons: tuple[str, ...] = ()
    prefetch_fail_on_error: bool = True

    adapter_kwargs: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 16
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=4.0))

    def __post_init__(self) -> None:
        super().__post_init__()
        for field_name in (
            "adapter_target",
            "model_id",
            "tag",
            "sample_rate_key",
            "audio_filepath_key",
            "results_key",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                msg = f"AudioLIDInferenceStage.{field_name} must be a non-empty string"
                raise ValueError(msg)
        if self.waveform_key is not None and (not isinstance(self.waveform_key, str) or not self.waveform_key.strip()):
            msg = "AudioLIDInferenceStage.waveform_key must be None or a non-empty string"
            raise ValueError(msg)
        if not isinstance(self.sample_rate, Integral) or isinstance(self.sample_rate, bool) or self.sample_rate <= 0:
            msg = f"AudioLIDInferenceStage.sample_rate must be a positive integer, got {self.sample_rate!r}"
            raise ValueError(msg)
        if not isinstance(self.batch_size, Integral) or isinstance(self.batch_size, bool) or self.batch_size <= 0:
            msg = f"AudioLIDInferenceStage.batch_size must be a positive integer, got {self.batch_size!r}"
            raise ValueError(msg)
        self.min_duration_sec = self._validated_duration("min_duration_sec", self.min_duration_sec)
        self.max_duration_sec = self._validated_duration("max_duration_sec", self.max_duration_sec)
        if 0 < self.max_duration_sec < self.min_duration_sec:
            msg = "AudioLIDInferenceStage.max_duration_sec must be zero or at least min_duration_sec"
            raise ValueError(msg)
        if isinstance(self.retry_skip_reasons, (str, bytes)):
            msg = "AudioLIDInferenceStage.retry_skip_reasons must contain only non-empty strings"
            raise TypeError(msg)
        try:
            retry_skip_reasons = tuple(self.retry_skip_reasons)
        except TypeError as exc:
            msg = "AudioLIDInferenceStage.retry_skip_reasons must be an iterable of non-empty strings"
            raise TypeError(msg) from exc
        if not all(isinstance(reason, str) and reason for reason in retry_skip_reasons):
            msg = "AudioLIDInferenceStage.retry_skip_reasons must contain only non-empty strings"
            raise ValueError(msg)
        self.retry_skip_reasons = retry_skip_reasons
        self.adapter_kwargs = dict(self.adapter_kwargs)

    @staticmethod
    def _validated_duration(field_name: str, value: object) -> float:
        if not isinstance(value, Real) or isinstance(value, bool):
            msg = f"AudioLIDInferenceStage.{field_name} must be a finite non-negative number, got {value!r}"
            raise TypeError(msg)
        duration = float(value)
        if not math.isfinite(duration) or duration < 0:
            msg = f"AudioLIDInferenceStage.{field_name} must be a finite non-negative number, got {value!r}"
            raise ValueError(msg)
        return duration

    def _create_adapter(self) -> AudioLIDAdapter:
        """Construct the configured adapter without conflating its model name with the result key."""
        adapter_cls = self._adapter_class()
        return cast(
            "AudioLIDAdapter",
            adapter_cls(
                sample_rate=int(self.sample_rate),
                **self.adapter_kwargs,
            ),
        )

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.results_key, _SKIP_ME_KEY]

    def process(self, task: AudioTask) -> AudioTask:
        msg = f"{type(self).__name__} only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Prepare eligible audio, identify it in one adapter call, and merge results."""
        retry_reasons = set(self.retry_skip_reasons)
        retried = 0
        for task in tasks:
            if task.data.get(_SKIP_ME_KEY) in retry_reasons:
                task.data.pop(_SKIP_ME_KEY, None)
                retried += 1
        if retried:
            logger.info("Audio LID {}: retrying {}/{} selector-rejected tasks", self.model_id, retried, len(tasks))
        skip_indices = {index for index, task in enumerate(tasks) if task.data.get(_SKIP_ME_KEY)}
        skip_indices.update(self._resume_indices(tasks))
        valid_indices, items = self._prepare_items(tasks, skip_indices)
        if not items:
            logger.info("Audio LID batch: no tasks require model inference ({})", len(tasks))
            return tasks

        if self._adapter is None:
            msg = f"{type(self).__name__}.setup() must run before process_batch()"
            raise RuntimeError(msg)
        results = self._adapter.identify_batch(items)
        if len(results) != len(items):
            msg = f"Audio LID adapter returned {len(results)} results for {len(items)} items (must match 1:1)"
            raise RuntimeError(msg)

        for task_index, result in zip(valid_indices, results, strict=True):
            self._write_result(tasks[task_index], result)
        logger.info("Audio LID batch: generated {} predictions for {}", len(results), self.model_id)
        return tasks

    def _resume_indices(self, tasks: list[AudioTask]) -> set[int]:
        if not self.skip_if_output_exists:
            return set()
        indices = {index for index, task in enumerate(tasks) if self._already_has_output(task)}
        if indices:
            logger.info(
                "Audio LID {}: reusing existing output for {}/{} tasks", self.model_id, len(indices), len(tasks)
            )
        return indices

    def _already_has_output(self, task: AudioTask) -> bool:
        results = task.data.get(self.results_key)
        return isinstance(results, dict) and self.model_id in results

    def _prepare_items(
        self,
        tasks: list[AudioTask],
        skip_indices: set[int],
    ) -> tuple[list[int], list[dict[str, Any]]]:
        valid_indices: list[int] = []
        items: list[dict[str, Any]] = []

        for index, task in enumerate(tasks):
            if index in skip_indices:
                continue
            audio_path = str(task.data.get(self.audio_filepath_key, "") or "")
            try:
                if self.waveform_key is not None:
                    waveform = task.data[self.waveform_key]
                    source_sample_rate = task.data[self.sample_rate_key]
                else:
                    waveform, source_sample_rate = self._load_audio(audio_path)
                prepared = self._prepare_waveform(waveform, source_sample_rate)
            except Exception as exc:  # noqa: BLE001 - malformed task audio is handled per row
                logger.warning(
                    "Audio LID {}: failed to prepare task {} from {}: {}", self.model_id, task.task_id, audio_path, exc
                )
                self._write_empty(task)
                if self.fail_on_audio_error:
                    task.data[_SKIP_ME_KEY] = _AUDIO_LOAD_ERROR
                continue

            if prepared.size / self.sample_rate < self.min_duration_sec:
                self._write_empty(task)
                continue
            if self.max_duration_sec > 0:
                max_samples = int(self.max_duration_sec * self.sample_rate)
                prepared = prepared[:max_samples]

            valid_indices.append(index)
            items.append({"waveform": prepared})

        return valid_indices, items

    def _prepare_waveform(self, waveform: object, sample_rate: object) -> np.ndarray:
        if not isinstance(sample_rate, Integral) or isinstance(sample_rate, bool) or sample_rate <= 0:
            msg = f"sample rate must be a positive integer, got {sample_rate!r}"
            raise ValueError(msg)

        tensor = ensure_waveform_2d(waveform)
        if tensor.ndim != _WAVEFORM_DIMENSIONS:
            msg = f"waveform must have one or two dimensions, got shape {tuple(tensor.shape)}"
            raise ValueError(msg)
        tensor = ensure_mono(tensor)
        prepared = tensor.squeeze(0).cpu().numpy()
        if int(sample_rate) != self.sample_rate:
            import librosa

            prepared = librosa.resample(prepared, orig_sr=int(sample_rate), target_sr=int(self.sample_rate))
        return np.ascontiguousarray(prepared, dtype=np.float32)

    def _result_mapping(self, task: AudioTask) -> dict[str, Any]:
        if self.results_key not in task.data:
            results = {}
            task.data[self.results_key] = results
        else:
            results = task.data[self.results_key]
        if not isinstance(results, dict):
            msg = f"task.data[{self.results_key!r}] must be a mapping, got {type(results).__name__}"
            raise TypeError(msg)
        return results

    def _write_empty(self, task: AudioTask) -> None:
        self._write_result(task, AudioLIDResult(language="", confidence=0.0))

    def _write_result(self, task: AudioTask, result: AudioLIDResult) -> None:
        canonical = AudioLIDResult(language=result.language, confidence=result.confidence)
        self._result_mapping(task)[self.model_id] = {
            "language": canonical.language,
            "confidence": canonical.confidence,
            "tag": self.tag,
        }
