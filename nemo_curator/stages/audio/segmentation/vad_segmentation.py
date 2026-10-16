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
import os
from dataclasses import KW_ONLY, dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
from loguru import logger

from nemo_curator.models.audio.vad.base import VADAdapter, _validate_detection_options
from nemo_curator.stages.audio._agent._agent_ready import (
    AgentReady,
    ConditionalRead,
    ConditionalWrite,
    Gates,
    IOSpec,
    Role,
    StageContract,
    StaticHints,
)
from nemo_curator.stages.audio._agent._residency import (
    InputResidency,
    validate_audio_key_configuration,
    validate_input_residency,
)
from nemo_curator.stages.audio.common import ensure_mono, ensure_waveform_2d
from nemo_curator.stages.audio.inference.base import AdapterInferenceStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

    from nemo_curator.models.audio.vad.base import VADResult, VADSegment


_DEFAULT_ADAPTER_TARGET = "nemo_curator.models.audio.vad.silero.SileroVADAdapter"


@dataclass
class VADSegmentationStage(AgentReady, AdapterInferenceStage[VADAdapter]):
    """Split recordings into speech segments through a selectable VAD adapter.

    The default adapter uses the official Silero TorchScript model. Select its
    ONNX runtime with ``adapter_kwargs={"backend": "onnx"}``. Detection
    options remain stage fields so changing runtimes does not change the task
    contract.

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

    Each speech item stores an original-channel, source-rate waveform slice;
    ``start_ms_key`` and ``end_ms_key`` are clamped integer offsets from the
    parent recording, and ``duration_key`` is that interval in seconds.
    ``original_file_key`` is source provenance, not a path to the slice, while
    ``segment_num_key`` is zero-based after invalid intervals are removed. A
    structured ``diar_segments_key`` makes ``num_speakers_key`` the count of
    labels overlapping this VAD interval; otherwise an existing parent count
    is only propagated. On a retained empty parent, duration is instead the
    full source duration. ``vad_empty_key`` and ``read_error_key`` are true-only
    audit flags for no usable speech and failed preparation/inference.

    Args:
        input_residency: Select resident audio, file audio, or resident-first automatic resolution.
        keep_segment_waveform_in_task: Keep each speech item's waveform; disable for metadata-only output.
        output_waveform_key: Waveform field written to each speech item.
        output_sample_rate_key: Source-rate field written to each speech item.
        start_ms_key: Speech start offset field, in integer milliseconds.
        end_ms_key: Speech end offset field, in integer milliseconds.
        segment_num_key: Zero-based emitted-segment index field.
        original_file_key: Source-file provenance field.
        diar_segments_key: Optional diarization intervals used for overlap counts.
        diar_start_key: Start-time field inside an input diarization interval.
        diar_end_key: End-time field inside an input diarization interval.
        diar_speaker_key: Speaker-label field inside an input diarization interval.
        num_speakers_key: Optional propagated or overlap-derived speaker count.
        num_samples_key: Stale parent sample-count field removed from speech items.
        legacy_duration_keys: Older parent duration aliases removed before each
            speech item receives its configured interval duration.
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

    adapter_target: str = _DEFAULT_ADAPTER_TARGET
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)
    audio_filepath_key: str = "audio_filepath"
    duration_key: str = "duration"
    segments_key: str = "segments"
    read_error_key: str = "read_error"
    vad_empty_key: str = "vad_empty"
    emit_audit_placeholders: bool = False
    fail_on_audio_error: bool = False
    prefetch_fail_on_error: bool = True

    _: KW_ONLY
    input_residency: InputResidency = "auto"
    keep_segment_waveform_in_task: bool = True
    output_waveform_key: str = "waveform"
    output_sample_rate_key: str = "sample_rate"
    start_ms_key: str = "start_ms"
    end_ms_key: str = "end_ms"
    segment_num_key: str = "segment_num"
    original_file_key: str = "original_file"
    diar_segments_key: str = "diar_segments"
    diar_start_key: str = "start"
    diar_end_key: str = "end"
    diar_speaker_key: str = "speaker"
    num_speakers_key: str = "num_speakers"
    num_samples_key: str = "num_samples"
    legacy_duration_keys: tuple[str, ...] = ("duration", "duration_sec")

    BATCH_ONLY: ClassVar[bool] = True
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        cardinality_options=["1:N fan-out", "1:1 nested-list"],
        gates=Gates(lifecycle_side_effects=True, per_row_independent=False),
    )
    KEY_ROLE_OVERRIDES: ClassVar[Mapping[str, Role]] = {
        "output_waveform_key": "waveform",
        "output_sample_rate_key": "sample_rate",
        "diar_start_key": "start",
        "diar_end_key": "end",
        "diar_speaker_key": "speaker_id",
    }
    INTERNAL_KEY_FIELDS: ClassVar[frozenset[str]] = frozenset({"read_error_key", "vad_empty_key"})

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_input_residency(self.input_residency, stage_name=self.name)
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
        configurable_keys = {
            name: getattr(self, name)
            for name in (
                "waveform_key",
                "sample_rate_key",
                "audio_filepath_key",
                "duration_key",
                "segments_key",
                "read_error_key",
                "vad_empty_key",
                "output_waveform_key",
                "output_sample_rate_key",
                "start_ms_key",
                "end_ms_key",
                "segment_num_key",
                "original_file_key",
                "diar_segments_key",
                "diar_start_key",
                "diar_end_key",
                "diar_speaker_key",
                "num_speakers_key",
                "num_samples_key",
            )
        }
        empty = [name for name, key in configurable_keys.items() if not isinstance(key, str) or not key.strip()]
        if empty:
            msg = f"[{self.name}] configurable task keys must be non-empty strings: {empty}"
            raise ValueError(msg)
        validate_audio_key_configuration(
            self.name,
            input_keys={
                "waveform_key": self.waveform_key,
                "sample_rate_key": self.sample_rate_key,
                "audio_filepath_key": self.audio_filepath_key,
            },
            output_keys={},
        )
        validate_audio_key_configuration(
            self.name,
            input_keys={},
            output_keys={
                "output_waveform_key": self.output_waveform_key,
                "output_sample_rate_key": self.output_sample_rate_key,
                "start_ms_key": self.start_ms_key,
                "end_ms_key": self.end_ms_key,
                "segment_num_key": self.segment_num_key,
                "duration_key": self.duration_key,
                "original_file_key": self.original_file_key,
                "num_speakers_key": self.num_speakers_key,
            },
        )
        validate_audio_key_configuration(
            self.name,
            input_keys={
                "diar_start_key": self.diar_start_key,
                "diar_end_key": self.diar_end_key,
                "diar_speaker_key": self.diar_speaker_key,
            },
            output_keys={},
        )
        if len(self.legacy_duration_keys) != len(set(self.legacy_duration_keys)) or any(
            not isinstance(key, str) or not key.strip() for key in self.legacy_duration_keys
        ):
            msg = f"[{self.name}] legacy_duration_keys must contain distinct non-empty strings"
            raise ValueError(msg)
        if self.nested:
            if self.segments_key in {self.waveform_key, "waveform"}:
                msg = "segments_key must be distinct from waveform cleanup keys in nested mode"
                raise ValueError(msg)
            validate_audio_key_configuration(
                self.name,
                input_keys={},
                output_keys={
                    "segments_key": self.segments_key,
                    "sample_rate_key": self.sample_rate_key,
                    "duration_key": self.duration_key,
                    "read_error_key": self.read_error_key,
                    "vad_empty_key": self.vad_empty_key,
                },
            )
        self._validate_cross_role_keys()

    def _parent_read_keys(self) -> dict[str, str]:
        return {
            "waveform_key": self.waveform_key,
            "sample_rate_key": self.sample_rate_key,
            "audio_filepath_key": self.audio_filepath_key,
            "segments_key": self.segments_key,
            "read_error_key": self.read_error_key,
            "vad_empty_key": self.vad_empty_key,
            "duration_key": self.duration_key,
            "diar_segments_key": self.diar_segments_key,
            "original_file_key": self.original_file_key,
            "num_speakers_key": self.num_speakers_key,
            "num_samples_key": self.num_samples_key,
            **{f"legacy_duration_keys[{index}]": key for index, key in enumerate(self.legacy_duration_keys)},
        }

    def _validate_cross_role_keys(self) -> None:
        """Reject same-scope aliases that would overwrite a different value."""
        inputs = self._parent_read_keys()
        if self.nested:
            outputs = {
                "segments_key": self.segments_key,
                "sample_rate_key": self.sample_rate_key,
                "duration_key": self.duration_key,
                "read_error_key": self.read_error_key,
                "vad_empty_key": self.vad_empty_key,
            }
            allowed = {
                ("segments_key", "segments_key"),
                ("sample_rate_key", "sample_rate_key"),
                ("duration_key", "duration_key"),
                ("read_error_key", "read_error_key"),
                ("vad_empty_key", "vad_empty_key"),
            }
        else:
            outputs = {
                "output_waveform_key": self.output_waveform_key,
                "output_sample_rate_key": self.output_sample_rate_key,
                "start_ms_key": self.start_ms_key,
                "end_ms_key": self.end_ms_key,
                "segment_num_key": self.segment_num_key,
                "duration_key": self.duration_key,
                "original_file_key": self.original_file_key,
                "num_speakers_key": self.num_speakers_key,
            }
            if self.emit_audit_placeholders:
                outputs.update(read_error_key=self.read_error_key, vad_empty_key=self.vad_empty_key)
            allowed = {
                ("waveform_key", "output_waveform_key"),
                ("sample_rate_key", "output_sample_rate_key"),
                ("duration_key", "duration_key"),
                ("original_file_key", "original_file_key"),
                ("num_speakers_key", "num_speakers_key"),
                ("read_error_key", "read_error_key"),
                ("vad_empty_key", "vad_empty_key"),
            }
        allowed.update((name, "duration_key") for name in inputs if name.startswith("legacy_duration_keys["))
        collisions = sorted(
            (input_name, output_name, input_key)
            for input_name, input_key in inputs.items()
            for output_name, output_key in outputs.items()
            if input_key == output_key and (input_name, output_name) not in allowed
        )
        if collisions:
            input_name, output_name, key = collisions[0]
            msg = (
                f"[{self.name}] '{input_name}' and '{output_name}' both resolve to {key!r}; "
                "only same-role input/output aliases may share a task key"
            )
            raise ValueError(msg)

    def _validate_options(self) -> None:
        _validate_detection_options(
            threshold=self.threshold,
            min_duration_sec=self.min_duration_sec,
            max_duration_sec=self.max_duration_sec,
            min_interval_ms=self.min_interval_ms,
            speech_pad_ms=self.speech_pad_ms,
        )
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

    def with_(self, *args: object, **kwargs: object) -> VADSegmentationStage:
        """Apply executor overrides while preserving fan-out validation."""
        stage = cast("VADSegmentationStage", super().with_(*args, **kwargs))
        stage._validate_options()
        return stage

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
            *([self.output_waveform_key] if self.keep_segment_waveform_in_task else []),
            self.output_sample_rate_key,
            self.start_ms_key,
            self.end_ms_key,
            self.segment_num_key,
            self.duration_key,
            self.original_file_key,
            self.num_speakers_key,
            self.vad_empty_key,
            self.read_error_key,
        ]

    def describe(self) -> StageContract:
        """Describe resident/file inputs and the configured output shape."""
        resident = IOSpec(data_keys=[self.waveform_key, self.sample_rate_key], accepts=["waveform"])
        file_input = IOSpec(data_keys=[self.audio_filepath_key], accepts=["file"])
        upstream_error = IOSpec(data_keys=[self.read_error_key])
        segment_keys = [
            *([self.output_waveform_key] if self.keep_segment_waveform_in_task else []),
            self.output_sample_rate_key,
            self.start_ms_key,
            self.end_ms_key,
            self.segment_num_key,
            self.duration_key,
            self.original_file_key,
        ]
        conditional_writes: list[ConditionalWrite] = []
        produces = ["tensor"] if self.keep_segment_waveform_in_task else []

        if self.nested:
            writes = IOSpec(
                data_keys=[self.segments_key],
                segment_data_keys=segment_keys,
                produces=produces,
            )
            speaker_scope = IOSpec(segment_data_keys=[self.num_speakers_key])
            conditional_writes.extend(
                [
                    ConditionalWrite(
                        writes=IOSpec(data_keys=[self.vad_empty_key, self.duration_key]),
                        condition="the adapter yields no usable speech; duration is the full source duration",
                    ),
                    ConditionalWrite(
                        writes=IOSpec(data_keys=[self.read_error_key]),
                        condition="upstream input, audio preparation, or one adapter result fails",
                    ),
                ]
            )
            if self.input_residency != "waveform":
                conditional_writes.append(
                    ConditionalWrite(
                        writes=IOSpec(data_keys=[self.sample_rate_key]),
                        condition="file input is decoded and its discovered sample rate is recorded",
                        requires_keys=[self.audio_filepath_key],
                    )
                )
            removes_keys = self._audio_cleanup_keys()
        elif self.emit_audit_placeholders:
            writes = IOSpec()
            speaker_scope = IOSpec(data_keys=[self.num_speakers_key])
            conditional_writes.extend(
                [
                    ConditionalWrite(
                        writes=IOSpec(data_keys=segment_keys, produces=produces),
                        condition="the adapter yields at least one usable speech interval",
                    ),
                    ConditionalWrite(
                        writes=IOSpec(data_keys=[self.vad_empty_key, self.duration_key]),
                        condition="the adapter yields no usable speech; duration is the full source duration",
                    ),
                    ConditionalWrite(
                        writes=IOSpec(data_keys=[self.read_error_key]),
                        condition="upstream input, audio preparation, or one adapter result fails",
                    ),
                ]
            )
            removes_keys = []
        else:
            writes = IOSpec(data_keys=segment_keys, produces=produces)
            speaker_scope = IOSpec(data_keys=[self.num_speakers_key])
            emitted_keys = set(segment_keys)
            removes_keys = [
                key
                for key in (
                    *self._audio_cleanup_keys(include_sample_rate=True),
                    self.output_waveform_key,
                    self.num_samples_key,
                    self.segments_key,
                    self.vad_empty_key,
                    *self.legacy_duration_keys,
                )
                if key not in emitted_keys
            ]

        conditional_writes.extend(
            [
                ConditionalWrite(
                    writes=speaker_scope,
                    condition=f"'{self.diar_segments_key}' contains structured intervals with overlapping labels",
                ),
                ConditionalWrite(
                    writes=speaker_scope,
                    condition=(
                        f"no structured overlap count replaces an existing '{self.num_speakers_key}' parent value"
                    ),
                    value_origin="upstream_same_key",
                ),
            ]
        )

        audio_reads = [resident] if self.input_residency == "waveform" else [file_input]
        conditional_reads = []
        if self.input_residency == "auto":
            audio_reads = [resident, file_input]
            conditional_reads = [
                ConditionalRead(
                    reads_one_of=[resident],
                    condition=f"a non-null '{self.waveform_key}' is present, so resident audio wins",
                    requires_keys=[self.waveform_key],
                ),
                ConditionalRead(
                    reads_one_of=[file_input],
                    condition=f"'{self.waveform_key}' is absent, so the file fallback runs",
                    requires_keys=[self.audio_filepath_key],
                    forbids_keys=[self.waveform_key],
                ),
            ]
        conditional_reads.append(
            ConditionalRead(
                reads_one_of=[upstream_error],
                condition=f"'{self.read_error_key}' is true, so audio preparation is bypassed",
                requires_keys=[self.read_error_key],
            )
        )
        return StageContract(
            reads_one_of=[*audio_reads, upstream_error],
            conditional_reads=conditional_reads,
            preferred_reads=resident if self.input_residency == "auto" else None,
            optional_reads=IOSpec(
                data_keys=list(
                    dict.fromkeys(
                        [
                            self.audio_filepath_key,
                            "audio_filepath",
                            self.diar_segments_key,
                            self.original_file_key,
                            self.num_speakers_key,
                            *self.legacy_duration_keys,
                        ]
                    )
                )
            ),
            writes=writes,
            conditional_writes=conditional_writes,
            cardinality="1:1 nested-list" if self.nested else "1:N fan-out",
            cardinality_options=["1:N fan-out", "1:1 nested-list"],
            iteration_key=(
                self.segments_key if self.nested else None if self.emit_audit_placeholders else self.segment_num_key
            ),
            removes_keys=list(dict.fromkeys(removes_keys)),
            invalidates_keys=list(dict.fromkeys([self.audio_filepath_key, "audio_filepath"]))
            if not self.nested
            else [],
            preserves_upstream_keys=not (self.emit_audit_placeholders and not self.nested),
            preserves_upstream_segment_keys=not self.nested,
            gates=Gates(
                requires_gpu=self.resources.requires_gpu,
                lifecycle_side_effects=True,
                per_row_independent=self.adapter_target == _DEFAULT_ADAPTER_TARGET,
            ),
        )

    def ray_stage_spec(self) -> dict[str, Any]:
        if self.nested:
            return {}
        return super().ray_stage_spec()

    def _resolve_audio(self, task: AudioTask) -> tuple[torch.Tensor, int]:  # noqa: C901
        waveform = None if self.input_residency == "file" else task.data.get(self.waveform_key)
        sample_rate = task.data.get(self.sample_rate_key)
        loaded_from_file = waveform is None
        if waveform is None:
            if self.input_residency == "waveform":
                msg = f"Task {task.task_id} has no resident {self.waveform_key!r}"
                raise KeyError(msg)
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
        if loaded_from_file:
            task.data[self.sample_rate_key] = resolved_rate

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
        for key in self._audio_cleanup_keys():
            task.data.pop(key, None)
        return task

    def _audio_cleanup_keys(self, *, include_sample_rate: bool = False) -> list[str]:
        """Remove legacy audio aliases only when they have no other configured input role."""
        inputs = self._parent_read_keys()
        fields = {"waveform_key": "waveform"}
        if include_sample_rate:
            fields["sample_rate_key"] = "sample_rate"
        keys = []
        for field_name, canonical in fields.items():
            keys.append(inputs[field_name])
            if canonical not in {key for name, key in inputs.items() if name != field_name}:
                keys.append(canonical)
        return list(dict.fromkeys(keys))

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
            *self._audio_cleanup_keys(include_sample_rate=True),
            self.output_waveform_key,
            self.start_ms_key,
            self.end_ms_key,
            self.segment_num_key,
            self.duration_key,
            *self.legacy_duration_keys,
            self.num_samples_key,
            self.segments_key,
            self.vad_empty_key,
        }
        segment_data = {key: value for key, value in item.items() if key not in inherited_drop_keys}
        segment_data.update(
            {
                self.output_sample_rate_key: sample_rate,
                self.start_ms_key: start_ms,
                self.end_ms_key: end_ms,
                self.segment_num_key: segment_num,
                self.duration_key: (end_ms - start_ms) / 1000.0,
                self.original_file_key: self._original_file(item),
            }
        )
        if self.keep_segment_waveform_in_task:
            segment_data[self.output_waveform_key] = waveform[:, start_sample:end_sample].clone()

        diar_segments = item.get(self.diar_segments_key)
        if isinstance(diar_segments, list) and all(isinstance(entry, dict) for entry in diar_segments):
            segment_data[self.num_speakers_key] = self._overlapping_speaker_count(diar_segments, segment)
        return segment_data

    def _overlapping_speaker_count(self, intervals: list[dict[str, Any]], segment: VADSegment) -> int:
        """Count labels on valid intervals without allowing optional metadata to discard speech."""
        speakers = set()
        for interval in intervals:
            try:
                start = float(interval[self.diar_start_key])
                end = float(interval[self.diar_end_key])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            speaker = interval.get(self.diar_speaker_key)
            if speaker is None or not math.isfinite(start) or not math.isfinite(end) or end <= start:
                continue
            if end > segment.start and start < segment.end:
                speakers.add(str(speaker))
        return len(speakers)

    def _original_file(self, item: dict[str, Any]) -> Any:  # noqa: ANN401
        for key in dict.fromkeys([self.original_file_key, self.audio_filepath_key]):
            if key in item:
                return item[key]
        legacy_path = item.get("audio_filepath")
        return legacy_path if isinstance(legacy_path, (str, os.PathLike)) else "unknown"

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
            for key in self._audio_cleanup_keys():
                task.data.pop(key, None)
            if self.nested:
                task.data[self.segments_key] = []
                return task
            return [task]

        if self.nested:
            task.data[self.segments_key] = [
                self._build_segment_item(task.data, waveform, sample_rate, segment, index)
                for index, segment in enumerate(segments)
            ]
            for key in self._audio_cleanup_keys():
                task.data.pop(key, None)
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
        msg = f"{type(self).__name__} only supports process_batch"
        raise NotImplementedError(msg)

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
        except Exception as exc:
            if self._is_memory_exhaustion(exc):
                raise
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
                if self._is_memory_exhaustion(exc):
                    raise
                if self.fail_on_audio_error:
                    msg = f"Failed to prepare VAD audio for task {task.task_id}"
                    raise RuntimeError(msg) from exc
                logger.warning("VAD: failed to prepare task {}: {}", task.task_id, exc)
                results_by_index[index] = self._audit_placeholder(task)
                continue
            ready.append((index, task, waveform, sample_rate))
        return ready, results_by_index

    def _process_vad_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Prepare a validated recording batch, delegate VAD, and fan out results."""
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

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Run one VAD batch."""
        if len(tasks) == 0:
            return []

        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task!s} failed validation for stage {self}"
                raise ValueError(msg)
        return self._process_vad_batch(tasks)
