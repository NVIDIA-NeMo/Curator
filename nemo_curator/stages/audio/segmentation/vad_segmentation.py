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

"""Adapter-backed voice-activity segmentation.

The stage owns Curator task I/O, audio loading, segment fan-out, metadata,
and error placeholders. The adapter selected by ``adapter_target`` owns the
model lifecycle, provider-specific sample-rate handling, and VAD inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from loguru import logger

from nemo_curator.models.audio.vad.base import VADAdapter
from nemo_curator.stages.audio.common import ensure_mono, ensure_waveform_2d
from nemo_curator.stages.audio.inference.base import AdapterInferenceStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    import torch

    from nemo_curator.models.audio.vad.base import VADResult, VADSegment


@dataclass
class VADSegmentationStage(AdapterInferenceStage[VADAdapter]):
    """Split recordings into speech segments through a selectable VAD adapter.

    The default adapter uses the official Silero TorchScript model. Select its
    ONNX runtime with ``adapter_kwargs={"backend": "onnx"}``, or select the
    TensorRT adapter explicitly and provide its engine path. Detection options
    remain stage fields so changing adapters does not change the task contract.

    Input may be an in-memory waveform plus sample rate or a file path. The
    adapter always receives contiguous mono float32 samples and the source
    sample rate; it is responsible for its own required resampling behavior.

    With ``nested=False`` (the default), speech intervals fan out into child
    tasks. With ``nested=True``, their dictionaries are stored under
    ``segments_key`` in the parent task. Fan-out mode preserves the historical
    behavior of dropping empty or failed recordings; set
    ``emit_audit_placeholders=True`` when a manifest workflow needs explicit
    ``vad_empty`` or ``read_error`` parent rows. Nested mode remains one-to-one
    and always retains its parent. Keep ``batch_size=1`` for deterministic
    fan-out IDs. Resume-oriented fan-out pipelines should also enable audit
    placeholders so an empty source still emits a completion record.
    """

    # Preserve the current-main positional constructor before adding adapter
    # selection fields. New code should use keyword arguments.
    min_interval_ms: int = 500
    min_duration_sec: float = 2.0
    max_duration_sec: float = 60.0
    threshold: float = 0.5
    speech_pad_ms: int = 300

    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    nested: bool = False

    name: str = "VADSegmentation"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=0.0))

    adapter_target: str = "nemo_curator.models.audio.vad.silero.SileroVADAdapter"
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)
    audio_filepath_key: str = "audio_filepath"
    duration_key: str = "duration"
    segments_key: str = "segments"
    read_error_key: str = "read_error"
    vad_empty_key: str = "vad_empty"
    emit_audit_placeholders: bool = False
    fail_on_audio_error: bool = False
    prefetch_fail_on_error: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_options()
        self.adapter_kwargs = dict(self.adapter_kwargs)
        stage_owned = {
            "threshold",
            "min_duration_sec",
            "max_duration_sec",
            "min_interval_ms",
            "speech_pad_ms",
        }
        conflicts = sorted(stage_owned.intersection(self.adapter_kwargs))
        if conflicts:
            msg = f"VAD detection options are stage fields and must not also appear in adapter_kwargs: {conflicts}"
            raise ValueError(msg)

    def _validate_options(self) -> None:
        if not math.isfinite(float(self.threshold)) or not 0.0 <= float(self.threshold) <= 1.0:
            msg = f"threshold must be finite and in [0, 1], got {self.threshold!r}"
            raise ValueError(msg)
        if not math.isfinite(float(self.min_duration_sec)) or self.min_duration_sec < 0:
            msg = f"min_duration_sec must be finite and non-negative, got {self.min_duration_sec!r}"
            raise ValueError(msg)
        if not math.isfinite(float(self.max_duration_sec)) or self.max_duration_sec <= self.min_duration_sec:
            msg = f"max_duration_sec must be finite and greater than min_duration_sec, got {self.max_duration_sec!r}"
            raise ValueError(msg)
        for option, value in (
            ("min_interval_ms", self.min_interval_ms),
            ("speech_pad_ms", self.speech_pad_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
                msg = f"{option} must be a non-negative integer, got {value!r}"
                raise ValueError(msg)
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, Integral) or self.batch_size <= 0:
            msg = f"batch_size must be a positive integer, got {self.batch_size!r}"
            raise ValueError(msg)
        if not self.nested and self.batch_size != 1:
            msg = (
                "VADSegmentationStage fan-out mode requires batch_size=1 so child task IDs "
                "and resumability counters remain attributable to one source recording; "
                "use nested=True to batch recordings"
            )
            raise ValueError(msg)
        self.min_interval_ms = int(self.min_interval_ms)
        self.speech_pad_ms = int(self.speech_pad_ms)
        self.batch_size = int(self.batch_size)

    def _create_adapter(self) -> VADAdapter:
        adapter_cls = self._adapter_class()
        return cast(
            "VADAdapter",
            adapter_cls(
                threshold=self.threshold,
                min_duration_sec=self.min_duration_sec,
                max_duration_sec=self.max_duration_sec,
                min_interval_ms=self.min_interval_ms,
                speech_pad_ms=self.speech_pad_ms,
                **self.adapter_kwargs,
            ),
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        """Accept either waveform/rate fields or an audio file path."""
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        if self.nested:
            return [], [self.segments_key, self.vad_empty_key, self.read_error_key]
        return [], [
            "waveform",
            "sample_rate",
            "start_ms",
            "end_ms",
            "segment_num",
            self.duration_key,
            "original_file",
            "num_speakers",
            self.vad_empty_key,
            self.read_error_key,
        ]

    def ray_stage_spec(self) -> dict[str, Any]:
        if self.nested:
            return {}
        return super().ray_stage_spec()

    def _resolve_audio(self, task: AudioTask) -> tuple[torch.Tensor, int]:
        waveform = task.data.get(self.waveform_key)
        sample_rate = task.data.get(self.sample_rate_key)
        if waveform is None:
            audio_path = task.data.get(self.audio_filepath_key)
            if not audio_path:
                msg = f"Task {task.task_id} has neither {self.waveform_key!r} nor {self.audio_filepath_key!r}"
                raise KeyError(msg)
            waveform, sample_rate = self._load_audio(str(audio_path))
        elif sample_rate is None:
            msg = f"Task {task.task_id} has {self.waveform_key!r} but no {self.sample_rate_key!r}"
            raise KeyError(msg)

        if isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral):
            msg = f"sample rate must be an integer, got {sample_rate!r}"
            raise TypeError(msg)
        resolved_rate = int(sample_rate)
        if resolved_rate <= 0:
            msg = f"sample rate must be positive, got {resolved_rate}"
            raise ValueError(msg)

        tensor = ensure_waveform_2d(waveform)
        if tensor.ndim != 2:  # noqa: PLR2004
            msg = f"waveform must be 1-D or 2-D channel-first audio, got {tuple(tensor.shape)}"
            raise ValueError(msg)
        import torch

        tensor = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if tensor.shape[-1] == 0:
            msg = "waveform must contain at least one sample"
            raise ValueError(msg)
        if not torch.isfinite(tensor).all():
            msg = "waveform contains non-finite samples"
            raise ValueError(msg)
        return tensor, resolved_rate

    @staticmethod
    def _adapter_item(waveform: torch.Tensor, sample_rate: int) -> dict[str, Any]:
        mono = ensure_mono(waveform).squeeze(0).numpy()
        return {
            "waveform": np.ascontiguousarray(mono, dtype=np.float32),
            "sample_rate": sample_rate,
        }

    def _as_read_error(self, task: AudioTask) -> AudioTask:
        task.data[self.read_error_key] = True
        task.data.pop(self.waveform_key, None)
        return task

    def _audit_placeholder(self, task: AudioTask) -> list[AudioTask]:
        """Retain failures only for one-to-one or explicitly audited flows."""
        placeholder = self._as_read_error(task)
        if self.nested:
            # Nested VAD feeds SegmentConcatenationStage, which treats an
            # empty segment list as a clean drop but rejects a missing field.
            placeholder.data[self.segments_key] = []
        return [placeholder] if self.nested or self.emit_audit_placeholders else []

    def _normalized_segments(
        self,
        result: VADResult,
        *,
        duration: float,
    ) -> list[VADSegment]:
        from nemo_curator.models.audio.vad.base import VADSegment

        normalized: list[VADSegment] = []
        previous_start = -1.0
        for segment in result.segments:
            start = float(segment.start)
            end = float(segment.end)
            if not math.isfinite(start) or not math.isfinite(end):
                msg = f"VAD adapter returned a non-finite segment: {segment!r}"
                raise ValueError(msg)
            start = min(max(start, 0.0), duration)
            end = min(max(end, 0.0), duration)
            if start < previous_start:
                msg = "VAD adapter returned segments out of order"
                raise ValueError(msg)
            previous_start = start
            if end <= start:
                logger.warning("Skipping degenerate VAD segment: {}", segment)
                continue
            normalized.append(VADSegment(start=start, end=end))
        return normalized

    def _build_segment_item(
        self,
        item: dict[str, Any],
        waveform: torch.Tensor,
        sample_rate: int,
        segment: VADSegment,
        segment_num: int,
    ) -> dict[str, Any]:
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        start_sample = min(max(int(segment.start * sample_rate), 0), waveform.shape[-1])
        end_sample = min(max(int(segment.end * sample_rate), start_sample), waveform.shape[-1])
        inherited_drop_keys = {
            self.waveform_key,
            self.sample_rate_key,
            "start_ms",
            "end_ms",
            "segment_num",
            self.duration_key,
            "duration",
            "duration_sec",
            "num_samples",
            self.segments_key,
            self.vad_empty_key,
        }
        segment_data = {key: value for key, value in item.items() if key not in inherited_drop_keys}
        segment_data.update(
            {
                "waveform": waveform[:, start_sample:end_sample].clone(),
                "sample_rate": sample_rate,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "segment_num": segment_num,
                self.duration_key: (end_ms - start_ms) / 1000.0,
                "original_file": item.get(
                    "original_file",
                    item.get(self.audio_filepath_key, "unknown"),
                ),
            }
        )

        diar_segments = item.get("diar_segments")
        if isinstance(diar_segments, list):
            speakers = {
                str(diar_segment["speaker"])
                for diar_segment in diar_segments
                if isinstance(diar_segment, dict)
                and {"start", "end", "speaker"} <= diar_segment.keys()
                and float(diar_segment["end"]) > segment.start
                and float(diar_segment["start"]) < segment.end
            }
            segment_data["num_speakers"] = len(speakers)
        return segment_data

    def _emit_segments(
        self,
        task: AudioTask,
        waveform: torch.Tensor,
        sample_rate: int,
        result: VADResult,
    ) -> AudioTask | list[AudioTask]:
        duration = waveform.shape[-1] / sample_rate
        segments = self._normalized_segments(result, duration=duration)
        if not segments:
            logger.warning("No speech segments detected for task {}", task.task_id)
            if not self.nested and not self.emit_audit_placeholders:
                task.data.pop(self.waveform_key, None)
                return []
            task.data[self.vad_empty_key] = True
            task.data.setdefault(self.duration_key, duration)
            task.data.pop(self.waveform_key, None)
            if self.nested:
                task.data[self.segments_key] = []
                return task
            return [task]

        if self.nested:
            task.data[self.segments_key] = [
                self._build_segment_item(task.data, waveform, sample_rate, segment, index)
                for index, segment in enumerate(segments)
            ]
            task.data.pop(self.waveform_key, None)
            return task

        output_tasks: list[AudioTask] = []
        for index, segment in enumerate(segments):
            segment_data = self._build_segment_item(task.data, waveform, sample_rate, segment, index)
            output_tasks.append(
                AudioTask(
                    data=segment_data,
                    dataset_name=task.dataset_name,
                    _metadata=dict(task._metadata),
                    _stage_perf=list(task._stage_perf),
                )
            )
        return output_tasks

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        results = self.process_batch([task])
        if self.nested:
            if len(results) != 1:
                msg = f"Nested VAD must return one task, got {len(results)}"
                raise RuntimeError(msg)
            return results[0]
        return results

    def _emit_adapter_result(
        self,
        task: AudioTask,
        waveform: torch.Tensor,
        sample_rate: int,
        result: VADResult,
    ) -> list[AudioTask]:
        """Convert one adapter result without allowing it to affect peer tasks."""
        if result.error is not None:
            logger.warning("VAD adapter failed task {}: {}", task.task_id, result.error)
            return self._audit_placeholder(task)
        try:
            emitted = self._emit_segments(task, waveform, sample_rate, result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VAD: invalid result for task {}: {}", task.task_id, exc)
            return self._audit_placeholder(task)
        return emitted if isinstance(emitted, list) else [emitted]

    def _prepare_tasks(
        self,
        tasks: list[AudioTask],
    ) -> tuple[list[tuple[int, AudioTask, torch.Tensor, int]], list[list[AudioTask] | None]]:
        """Prepare valid recordings while retaining placeholders at their indices."""
        ready: list[tuple[int, AudioTask, torch.Tensor, int]] = []
        results_by_index: list[list[AudioTask] | None] = [None] * len(tasks)
        for index, task in enumerate(tasks):
            if task.data.get(self.read_error_key):
                results_by_index[index] = self._audit_placeholder(task)
                continue
            try:
                waveform, sample_rate = self._resolve_audio(task)
            except Exception as exc:
                if self.fail_on_audio_error:
                    msg = f"Failed to prepare VAD audio for task {task.task_id}"
                    raise RuntimeError(msg) from exc
                logger.warning("VAD: failed to prepare task {}: {}", task.task_id, exc)
                results_by_index[index] = self._audit_placeholder(task)
                continue
            ready.append((index, task, waveform, sample_rate))
        return ready, results_by_index

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Prepare a ragged recording batch, delegate VAD, and fan out results."""
        if not tasks:
            return []
        if self._adapter is None:
            msg = "VAD adapter is not initialized; setup() was not called"
            raise RuntimeError(msg)
        if not self.nested and len(tasks) > 1:
            msg = (
                "VAD fan-out requires one input per process_batch call so emitted children can be "
                "attributed to their source recording"
            )
            raise ValueError(msg)

        ready, results_by_index = self._prepare_tasks(tasks)

        if ready:
            adapter_results = self._adapter.detect_batch(
                [self._adapter_item(waveform, sample_rate) for _, _, waveform, sample_rate in ready]
            )
            if len(adapter_results) != len(ready):
                msg = f"VAD adapter returned {len(adapter_results)} results for {len(ready)} items (must match 1:1)"
                raise RuntimeError(msg)
            for (index, task, waveform, sample_rate), result in zip(ready, adapter_results, strict=True):
                results_by_index[index] = self._emit_adapter_result(task, waveform, sample_rate, result)

        return [result for task_results in results_by_index if task_results for result in task_results]
