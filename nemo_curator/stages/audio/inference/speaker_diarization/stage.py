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

"""Adapter-backed whole-recording speaker diarization.

The stage owns Curator task I/O, audio loading, downmixing and resampling,
result assembly, resumability, and optional RTTM files. The adapter selected
by ``adapter_target`` owns provider weights, worker-local model state, and
diarization inference.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import KW_ONLY, dataclass, field
from numbers import Integral
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast
from urllib.parse import quote

import numpy as np
from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.models.audio.speaker_diarization.base import DiarizationAdapter, DiarizationInputError
from nemo_curator.stages.audio._agent._agent_ready import (
    AgentReady,
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
from nemo_curator.stages.audio.inference.base import (
    AdapterInferenceStage,
    _channel_first_waveform,
    _fanout_audio_segment,
    _fanout_original_file,
    _fanout_path_keys,
    _inference_audio_conditional_reads,
    _inference_audio_read_specs,
    _stable_audio_identity,
    _validate_fanout_key_contract,
    _validate_inference_audio_input,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nemo_curator.models.audio.speaker_diarization.base import DiarizationResult, DiarizationSegment


_DEFAULT_ADAPTER_TARGET = "nemo_curator.models.audio.speaker_diarization.sortformer.NeMoSortformerAdapter"


def _safe_rttm_basename(session_name: str) -> str:
    """Return a path-safe RTTM filename stem without changing RTTM identity."""
    encoded = quote(session_name, safe="")
    max_bytes = 255 - len(".rttm")
    if len(encoded) <= max_bytes:
        return encoded
    # Literal '!' is percent-encoded, reserving this suffix for shortened names.
    suffix = "!" + hashlib.sha256(session_name.encode("utf-8")).hexdigest()
    return encoded[: max_bytes - len(suffix)] + suffix


def _validated_relative_rttm_path(value: str, *, label: str) -> Path:
    """Return a normalized relative path, rejecting cross-platform escapes."""
    normalized = value.replace("\\", "/")
    windows_path = PureWindowsPath(value)
    if PurePosixPath(normalized).is_absolute() or windows_path.is_absolute() or windows_path.drive:
        msg = f"{label} must be relative to rttm_out_dir, got {value!r}"
        raise ValueError(msg)
    if any(part == ".." for part in normalized.split("/")):
        msg = f"{label} must not contain parent-directory traversal, got {value!r}"
        raise ValueError(msg)
    return Path(normalized)


def _resolve_under_rttm_root(rttm_out_dir: str, relative_path: Path, *, label: str) -> Path:
    """Resolve a path and prove that it remains below the configured root."""
    root = Path(rttm_out_dir).expanduser().resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        msg = f"Resolved {label} escapes rttm_out_dir: {path} is not below {root}"
        raise ValueError(msg) from exc
    return path


def _resolve_rttm_dir(rttm_out_dir: str, shard_key: str | None = None) -> str:
    """Use the configured RTTM root and preserve a safe task shard layout."""
    relative_path = Path()
    if shard_key:
        relative_path = _validated_relative_rttm_path(shard_key, label="RTTM shard key")
    return str(_resolve_under_rttm_root(rttm_out_dir, relative_path, label="RTTM shard directory"))


def _write_rttm(
    segments: list[DiarizationSegment],
    session_name: str,
    rttm_out_dir: str,
    *,
    shard_key: str | None = None,
) -> str:
    """Write one deterministic RTTM file and return its absolute path."""
    out_dir = Path(_resolve_rttm_dir(rttm_out_dir, shard_key))
    path = _resolve_under_rttm_root(
        rttm_out_dir,
        out_dir.relative_to(Path(rttm_out_dir).expanduser().resolve()) / f"{_safe_rttm_basename(session_name)}.rttm",
        label="RTTM output path",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for segment in segments:
        duration = float(segment["end"]) - float(segment["start"])
        if duration <= 0:
            logger.warning("Skipping degenerate diarization segment: {}", segment)
            continue
        lines.append(
            f"SPEAKER {session_name} 1 {float(segment['start']):.3f} {duration:.3f} "
            f"<NA> <NA> {segment['speaker']} <NA> <NA>\n"
        )
    with path.open("w", encoding="utf-8") as stream:
        stream.writelines(lines)
    return str(path)


@dataclass
class InferenceSortformerStage(AgentReady, AdapterInferenceStage[DiarizationAdapter]):
    """Diarize whole recordings through a YAML-selectable model adapter.

    By default, a configured ``waveform_key`` selects resident input; an explicit
    ``input_residency`` overrides that selection. Model input becomes contiguous mono
    float32 audio at ``target_sample_rate``. Leave it as ``None`` to pass
    ``audio_filepath_key`` to an adapter for provider-native decoding or
    bounded file streaming after a lightweight header probe.

    ``diar_segments_key`` contains ordered, retained intervals whose configured
    ``start_key``/``end_key`` values are seconds and whose ``speaker_key`` labels
    are local to one recording, not identities shared across recordings.
    ``num_speakers_key`` is the count of distinct retained labels for the whole
    recording, not for one interval. A zero accompanied by this stage's
    ``notes_key`` entry means preparation or decoding failed, not that silence was proven.
    ``rttm_filepath_key`` is a relative path below ``rttm_out_dir`` containing
    the same normalized intervals.

    Args:
        adapter_target: Import path of a class implementing
            :class:`DiarizationAdapter`.
        model_id: Stable provider model ID passed to the adapter.
        target_sample_rate: Sample rate presented to the adapter.
        waveform_key: Resident audio field; defaults to None for file input.
        input_residency: Optional file, waveform, or auto policy; omitted uses waveform_key.
        fanout: Emit a cloned source-audio task for each speaker turn.
        rttm_naming: task_id (default), legacy session names, or source_hash for reuse.
        pred_text_key: Parent transcript removed from fan-out children.
        audio_item_id_key: Fallback identity for resident fan-out provenance.
        sample_rate_key: Source sample-rate field for waveform input.
        audio_filepath_key: Audio file field for file input and RTTM naming.
        diar_segments_key: Manifest field for JSON-safe diarization intervals.
        num_speakers_key: Manifest field for the distinct speaker count, or None to disable.
        store_segments: Store intervals in the task in addition to any RTTM.
        rttm_out_dir: Optional RTTM root; task_id naming uses the existing shard
            and a task-ID digest, while source_hash uses selected source identity.
        skip_if_output_exists: Reuse complete output by key presence. An empty
            interval list is a complete result. When RTTM output is enabled,
            the referenced file must exist safely below ``rttm_out_dir``.
        fail_on_audio_error: Raise on task-local decode/preparation errors. If
            false, write an empty result and an explanatory note.
        adapter_kwargs: Provider-specific constructor settings.
        session_name_key: Optional recording identity used for RTTM rows and
            filenames before falling back to the audio filename or task ID.
        read_error_key: Upstream failure flag that bypasses inference.
        start_key: Start-time field inside each stored diarization interval.
        end_key: End-time field inside each stored diarization interval.
        speaker_key: Recording-local speaker-label field inside each interval.
    """

    # Keep every current-main constructor field in its original positional
    # order. They are compatibility aliases for the canonical stage/adapter
    # fields below; new code should use keyword arguments.
    model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2.1"
    model_path: str | None = None
    cache_dir: str | None = None
    diar_model: Any | None = field(default=None, repr=False)
    filepath_key: str = "audio_filepath"
    diar_segments_key: str = "diar_segments"
    rttm_out_dir: str | None = None
    chunk_len: int | None = 340
    chunk_left_context: int | None = 1
    chunk_right_context: int | None = 40
    fifo_len: int | None = 40
    spkcache_update_period: int | None = 300
    spkcache_len: int | None = 188
    inference_batch_size: int | None = 1
    name: str = "Sortformer_inference"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=8.0))

    adapter_target: str = _DEFAULT_ADAPTER_TARGET
    model_id: str = "nvidia/diar_streaming_sortformer_4spk-v2.1"
    target_sample_rate: int = 16000
    waveform_key: str | None = None
    sample_rate_key: str = "sample_rate"
    audio_filepath_key: str = "audio_filepath"
    num_speakers_key: str | None = "num_speakers"
    notes_key: str = "additional_notes"

    store_segments: bool = True
    rttm_filepath_key: str = "rttm_filepath"
    skip_if_output_exists: bool = False
    fail_on_audio_error: bool = False
    prefetch_fail_on_error: bool = True

    adapter_kwargs: dict[str, Any] = field(default_factory=dict)

    _: KW_ONLY
    session_name_key: str = "session_name"
    read_error_key: str = "read_error"
    start_key: str = "start"
    end_key: str = "end"
    speaker_key: str = "speaker"

    # None preserves main's waveform_key-based input selection.
    input_residency: InputResidency | None = None
    fanout: bool = False
    rttm_naming: Literal["task_id", "legacy", "source_hash"] = "task_id"
    start_ms_key: str = "start_ms"
    end_ms_key: str = "end_ms"
    duration_key: str = "duration"
    segment_num_key: str = "segment_num"
    original_file_key: str = "original_file"
    audio_item_id_key: str = "audio_item_id"
    pred_text_key: str = "pred_text"

    BATCH_ONLY: ClassVar[bool] = True
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        cardinality_options=["1:1", "1:1 nested-list", "1:N fan-out"],
        gates=Gates(
            requires_gpu=True,
            requires_internet_first_run=True,
            lifecycle_side_effects=True,
            per_row_independent=False,
        ),
    )
    KEY_ROLE_OVERRIDES: ClassVar[Mapping[str, Role]] = {
        "session_name_key": "item_id",
        "rttm_filepath_key": "output_path",
    }
    INTERNAL_KEY_FIELDS: ClassVar[frozenset[str]] = frozenset({"notes_key", "read_error_key"})

    def __post_init__(self) -> None:
        super().__post_init__()
        self._apply_legacy_aliases()
        self._configure_agent_options()
        if not isinstance(self.target_sample_rate, Integral) or isinstance(self.target_sample_rate, bool):
            msg = f"target_sample_rate must be a positive integer, got {self.target_sample_rate!r}"
            raise TypeError(msg)
        if int(self.target_sample_rate) <= 0:
            msg = f"target_sample_rate must be positive, got {self.target_sample_rate}"
            raise ValueError(msg)
        if not isinstance(self.batch_size, Integral) or isinstance(self.batch_size, bool) or self.batch_size <= 0:
            msg = f"batch_size must be a positive integer, got {self.batch_size!r}"
            raise ValueError(msg)
        if not self.store_segments and self.rttm_out_dir is None and not self.fanout:
            msg = "store_segments=False requires rttm_out_dir so diarization is not discarded"
            raise ValueError(msg)
        self.target_sample_rate = int(self.target_sample_rate)
        self.batch_size = int(self.batch_size)
        self.adapter_kwargs = dict(self.adapter_kwargs)
        input_keys = {
            "audio_filepath_key": self.audio_filepath_key,
            "session_name_key": self.session_name_key,
            "read_error_key": self.read_error_key,
        }
        if self.input_residency != "file" or self.fanout:
            input_keys.update(waveform_key=self.waveform_key, sample_rate_key=self.sample_rate_key)
        output_keys = {
            "notes_key": self.notes_key,
        }
        if self.num_speakers_key is not None:
            output_keys["num_speakers_key"] = self.num_speakers_key
        if self.store_segments or self.fanout:
            output_keys["diar_segments_key"] = self.diar_segments_key
        if self.fanout:
            input_keys["audio_item_id_key"] = self.audio_item_id_key
            output_keys.update(
                {
                    "start_ms_key": self.start_ms_key,
                    "end_ms_key": self.end_ms_key,
                    "duration_key": self.duration_key,
                    "segment_num_key": self.segment_num_key,
                    "speaker_key": self.speaker_key,
                    "original_file_key": self.original_file_key,
                    "start_key": self.start_key,
                    "end_key": self.end_key,
                }
            )
        if self.rttm_out_dir is not None:
            output_keys["rttm_filepath_key"] = self.rttm_filepath_key
        validate_audio_key_configuration(self.name, input_keys=input_keys, output_keys=output_keys)
        validate_audio_key_configuration(
            self.name,
            input_keys={},
            output_keys={
                "start_key": self.start_key,
                "end_key": self.end_key,
                "speaker_key": self.speaker_key,
            },
        )

    def _configure_agent_options(self) -> None:
        if self.input_residency is None:
            self.input_residency = "waveform" if self.waveform_key else "file"
        validate_input_residency(self.input_residency, stage_name=self.name)
        if (self.input_residency != "file" or self.fanout) and self.waveform_key is None:
            self.waveform_key = "waveform"
        if self.rttm_naming not in {"task_id", "legacy", "source_hash"}:
            msg = "rttm_naming must be 'task_id', 'legacy', or 'source_hash'"
            raise ValueError(msg)
        if self.fanout:
            _validate_fanout_key_contract(
                stage_name=self.name,
                audio_filepath_key=self.audio_filepath_key,
                output_keys=self._fanout_output_keys(),
                removed_container_keys=(self.diar_segments_key, self.pred_text_key),
            )
            self.is_resumable = False

    def _apply_legacy_aliases(self) -> None:
        """Translate the pre-adapter constructor surface without hiding conflicts."""
        default_model_id = "nvidia/diar_streaming_sortformer_4spk-v2.1"
        if self.model_name != default_model_id:
            if self.model_id not in {default_model_id, self.model_name}:
                msg = f"model_name={self.model_name!r} conflicts with model_id={self.model_id!r}"
                raise ValueError(msg)
            self.model_id = self.model_name
        if self.filepath_key != "audio_filepath":
            if self.audio_filepath_key not in {"audio_filepath", self.filepath_key}:
                msg = (
                    f"filepath_key={self.filepath_key!r} conflicts with audio_filepath_key={self.audio_filepath_key!r}"
                )
                raise ValueError(msg)
            self.audio_filepath_key = self.filepath_key

        adapter_kwargs = dict(self.adapter_kwargs)
        aliases = {
            "model_path": self.model_path,
            "cache_dir": self.cache_dir,
            "preloaded_model": self.diar_model,
            "chunk_len": self.chunk_len,
            "chunk_left_context": self.chunk_left_context,
            "chunk_right_context": self.chunk_right_context,
            "fifo_len": self.fifo_len,
            "spkcache_update_period": self.spkcache_update_period,
            "spkcache_len": self.spkcache_len,
            "inference_batch_size": self.inference_batch_size,
        }
        historic_defaults = {
            "model_path": None,
            "cache_dir": None,
            "preloaded_model": None,
            "chunk_len": 340,
            "chunk_left_context": 1,
            "chunk_right_context": 40,
            "fifo_len": 40,
            "spkcache_update_period": 300,
            "spkcache_len": 188,
            "inference_batch_size": 1,
        }
        for name, value in aliases.items():
            if name in adapter_kwargs:
                if (
                    value != historic_defaults[name]
                    and adapter_kwargs[name] is not value
                    and adapter_kwargs[name] != value
                ):
                    msg = f"legacy {name}={value!r} conflicts with adapter_kwargs[{name!r}]={adapter_kwargs[name]!r}"
                    raise ValueError(msg)
                continue
            if value != historic_defaults[name]:
                adapter_kwargs[name] = value
        self.adapter_kwargs = adapter_kwargs

        # Preserve introspection of the historic attributes, including their
        # old defaults, while keeping the canonical values authoritative.
        self.model_name = self.model_id
        self.filepath_key = self.audio_filepath_key
        self.model_path = cast("str | None", adapter_kwargs.get("model_path"))
        self.cache_dir = cast("str | None", adapter_kwargs.get("cache_dir"))
        self.diar_model = adapter_kwargs.get("preloaded_model")
        self.chunk_len = cast("int | None", adapter_kwargs.get("chunk_len", 340))
        self.chunk_left_context = cast("int | None", adapter_kwargs.get("chunk_left_context", 1))
        self.chunk_right_context = cast("int | None", adapter_kwargs.get("chunk_right_context", 40))
        self.fifo_len = cast("int | None", adapter_kwargs.get("fifo_len", 40))
        self.spkcache_update_period = cast("int | None", adapter_kwargs.get("spkcache_update_period", 300))
        self.spkcache_len = cast("int | None", adapter_kwargs.get("spkcache_len", 188))
        self.inference_batch_size = cast("int | None", adapter_kwargs.get("inference_batch_size", 1))

    def _create_adapter(self) -> DiarizationAdapter:
        adapter_cls = self._adapter_class()
        return cast(
            "DiarizationAdapter",
            adapter_cls(
                model_id=self.model_id,
                sample_rate=self.target_sample_rate,
                **self.adapter_kwargs,
            ),
        )

    def outputs(self) -> tuple[list[str], list[str]]:
        if self.fanout:
            return ["data"], self._fanout_output_keys()
        keys = [key for key in (self.num_speakers_key, self.notes_key) if key is not None]
        if self.store_segments:
            keys.append(self.diar_segments_key)
        if self.rttm_out_dir is not None:
            keys.append(self.rttm_filepath_key)
        return [], keys

    def describe(self) -> StageContract:
        """Describe configured task I/O without loading the selected adapter."""
        audio_inputs = _inference_audio_read_specs(
            self.input_residency,
            audio_filepath_key=self.audio_filepath_key,
            waveform_key=self.waveform_key,
            sample_rate_key=self.sample_rate_key,
        )
        result_keys = [self.num_speakers_key] if self.num_speakers_key is not None else []
        if self.store_segments and not self.fanout:
            result_keys.append(self.diar_segments_key)

        optional_reads = [self.notes_key]
        if self.fanout:
            optional_reads.extend([self.session_name_key, self.audio_item_id_key, self.original_file_key])
        if self.rttm_out_dir is not None:
            optional_reads.append(self.session_name_key)
            if self.waveform_key:
                optional_reads.append(self.audio_filepath_key)
        if self.skip_if_output_exists:
            if self.num_speakers_key is not None:
                optional_reads.append(self.num_speakers_key)
            if self.store_segments:
                optional_reads.append(self.diar_segments_key)
            if self.rttm_out_dir is not None:
                optional_reads.append(self.rttm_filepath_key)

        return StageContract(
            reads_one_of=[*audio_inputs, IOSpec(data_keys=[self.read_error_key])],
            conditional_reads=_inference_audio_conditional_reads(
                self.input_residency, waveform_key=self.waveform_key, sample_rate_key=self.sample_rate_key
            ),
            preferred_reads=(
                IOSpec(data_keys=[self.waveform_key, self.sample_rate_key], accepts=["waveform"])
                if self.input_residency != "file"
                else None
            ),
            writes=IOSpec(data_keys=self._fanout_output_keys(), produces=["tensor"]) if self.fanout else IOSpec(),
            removes_keys=(
                [*_fanout_path_keys(self.audio_filepath_key), self.diar_segments_key, self.pred_text_key]
                if self.fanout
                else []
            ),
            optional_reads=IOSpec(data_keys=list(dict.fromkeys(optional_reads))),
            conditional_writes=[
                ConditionalWrite(
                    writes=IOSpec(data_keys=result_keys),
                    condition=(
                        f"'{self.read_error_key}' is absent or false, so a result is produced or safely reused"
                    ),
                ),
                *(
                    [
                        ConditionalWrite(
                            writes=IOSpec(segment_data_keys=[self.start_key, self.end_key, self.speaker_key]),
                            condition="the retained diarization result contains at least one interval",
                        )
                    ]
                    if self.store_segments and not self.fanout
                    else []
                ),
                *(
                    [
                        ConditionalWrite(
                            writes=IOSpec(data_keys=[self.rttm_filepath_key], produces=["disk"]),
                            condition=(
                                f"'{self.read_error_key}' is absent or false and the adapter returns a diarization result"
                            ),
                        )
                    ]
                    if self.rttm_out_dir is not None
                    else []
                ),
                *(
                    [
                        ConditionalWrite(
                            writes=IOSpec(data_keys=[self.notes_key]),
                            condition="audio preparation or file decoding fails and fail_on_audio_error is false",
                            value_origin="augments_upstream_same_key",
                        )
                    ]
                    if not self.fail_on_audio_error
                    else []
                ),
            ],
            metadata_reads=["_shard_key"] if self.rttm_out_dir is not None and self.rttm_naming == "task_id" else [],
            cardinality="1:N fan-out" if self.fanout else ("1:1 nested-list" if self.store_segments else "1:1"),
            cardinality_options=["1:1", "1:1 nested-list", "1:N fan-out"],
            iteration_key=self.segment_num_key
            if self.fanout
            else (self.diar_segments_key if self.store_segments else None),
            gates=Gates(
                writes_to_disk=self.rttm_out_dir is not None,
                output_path_params=["rttm_out_dir"],
                requires_gpu=self.resources.requires_gpu,
                requires_internet_first_run=self.model_path is None and self.diar_model is None,
                lifecycle_side_effects=True,
                requires_stable_task_id=self.rttm_out_dir is not None and self.rttm_naming == "task_id",
                per_row_independent=(
                    self.adapter_target == _DEFAULT_ADAPTER_TARGET
                    and (self.rttm_out_dir is None or self.rttm_naming == "source_hash")
                ),
            ),
        )

    def validate_input(self, task: AudioTask) -> bool:
        return _validate_inference_audio_input(
            task,
            stage_name=self.name,
            residency=self.input_residency,
            audio_filepath_keys=(self.audio_filepath_key,),
            waveform_key=self.waveform_key,
            sample_rate_key=self.sample_rate_key,
        )

    def _uses_resident(self, task: AudioTask) -> bool:
        return self.input_residency == "waveform" or (
            self.input_residency == "auto" and task.data.get(self.waveform_key) is not None
        )

    def ray_stage_spec(self) -> dict[str, Any]:
        if self.fanout:
            return {RayStageSpecKeys.IS_FANOUT_STAGE: True}
        return super().ray_stage_spec()

    def _fanout_output_keys(self) -> list[str]:
        keys = [
            self.waveform_key,
            self.sample_rate_key,
            self.start_key,
            self.end_key,
            self.start_ms_key,
            self.end_ms_key,
            self.duration_key,
            self.segment_num_key,
            self.speaker_key,
            self.original_file_key,
        ]
        if self.num_speakers_key is not None:
            keys.append(self.num_speakers_key)
        if self.rttm_out_dir is not None:
            keys.append(self.rttm_filepath_key)
        return keys

    def _source_audio(self, task: AudioTask) -> tuple[np.ndarray, int]:
        if self._uses_resident(task):
            return _channel_first_waveform(task.data[self.waveform_key]), int(task.data[self.sample_rate_key])
        audio, rate = self._load_audio(str(task.data[self.audio_filepath_key]))
        return _channel_first_waveform(audio), rate

    def _export_session_name(self, task: AudioTask) -> str:
        if self.rttm_naming == "task_id":
            return self._session_name(task)
        path = (
            None
            if self.rttm_naming == "source_hash" and self._uses_resident(task)
            else task.data.get(self.audio_filepath_key)
        )
        if path or task.data.get(self.session_name_key):
            return str(task.data.get(self.session_name_key) or Path(str(path)).stem)
        waveform, rate = self._source_audio(task)
        return _stable_audio_identity({}, waveform, rate, source_path=None, explicit_keys=())

    def _export_shard_key(self, task: AudioTask) -> str | None:
        if self.rttm_naming == "task_id":
            return self._rttm_shard_key(task)
        if self.rttm_naming == "legacy":
            return None
        if self._uses_resident(task):
            waveform, rate = self._source_audio(task)
            source = _stable_audio_identity({}, waveform, rate, source_path=None, explicit_keys=())
            raw_waveform = task.data[self.waveform_key]
            source += "|" + str(getattr(raw_waveform, "dtype", type(raw_waveform).__name__))
        else:
            source = "file:" + os.path.realpath(str(task.data[self.audio_filepath_key]))
        identity = json.dumps([self._export_session_name(task), source], ensure_ascii=False)
        return "source_" + hashlib.sha256(identity.encode()).hexdigest()

    def _fanout_tasks(self, task: AudioTask) -> list[AudioTask]:
        notes = task.data.get(self.notes_key)
        if task.data.get(self.read_error_key) or (isinstance(notes, dict) and self.name in notes):
            return []
        segments = task.data.get(self.diar_segments_key, [])
        if not segments:
            return []
        waveform, rate = self._source_audio(task)
        path = str(task.data[self.audio_filepath_key]) if not self._uses_resident(task) else None
        original_file = _fanout_original_file(
            task.data,
            original_file_key=self.original_file_key,
            source_path=path,
            stable_identity=_stable_audio_identity(
                task.data,
                waveform,
                rate,
                source_path=path,
                explicit_keys=(self.session_name_key,),
                fallback_keys=(self.audio_item_id_key,),
            ),
        )
        removed = {
            *_fanout_path_keys(self.audio_filepath_key),
            self.diar_segments_key,
            self.pred_text_key,
            self.waveform_key,
            self.sample_rate_key,
        }
        children = []
        for index, segment in enumerate(segments):
            audio, start, end = _fanout_audio_segment(
                waveform, rate, start=float(segment[self.start_key]), end=float(segment[self.end_key])
            )
            if end <= start:
                continue
            data = {key: value for key, value in task.data.items() if key not in removed}
            data.update(
                {
                    self.waveform_key: audio,
                    self.sample_rate_key: rate,
                    self.start_key: start,
                    self.end_key: end,
                    self.start_ms_key: round(start * 1000),
                    self.end_ms_key: round(end * 1000),
                    self.duration_key: end - start,
                    self.segment_num_key: index,
                    self.speaker_key: segment[self.speaker_key],
                    self.original_file_key: original_file,
                }
            )
            children.append(
                AudioTask(
                    dataset_name=task.dataset_name,
                    data=data,
                    _metadata=dict(task._metadata or {}),
                    _stage_perf=list(task._stage_perf),
                )
            )
        return children

    def _prepare_waveform(self, waveform: object, sample_rate: object) -> np.ndarray:
        source_rate = int(sample_rate)
        if source_rate <= 0:
            msg = f"sample rate must be positive, got {source_rate}"
            raise ValueError(msg)
        tensor = ensure_waveform_2d(waveform)
        if tensor.ndim != 2:  # noqa: PLR2004
            msg = f"waveform must be 1-D or 2-D channel-first audio, got {tuple(tensor.shape)}"
            raise ValueError(msg)
        import torch

        tensor = ensure_mono(tensor).squeeze(0).to(device="cpu", dtype=torch.float32)
        if source_rate != self.target_sample_rate:
            import torchaudio

            tensor = torchaudio.functional.resample(tensor, source_rate, self.target_sample_rate)
        waveform_array = np.ascontiguousarray(tensor.numpy(), dtype=np.float32)
        if not np.isfinite(waveform_array).all():
            msg = "waveform contains non-finite samples"
            raise ValueError(msg)
        return waveform_array

    def diarize(self, audio_paths: list[str]) -> list[list[dict[str, Any]]]:
        """Diarize file paths through the initialized adapter or legacy model.

        This preserves the public method exposed by the original monolithic
        stage. Normal pipeline users should let the executor call ``setup``;
        callers that invoke this method directly must either do the same or
        provide the legacy ``diar_model`` constructor argument.
        """
        paths = [str(path) for path in audio_paths]
        if self._adapter is not None:
            results = self._adapter.diarize_batch([{"audio_filepath": path} for path in paths])
            if len(results) != len(paths):
                msg = f"Diarization adapter returned {len(results)} results for {len(paths)} paths (must match 1:1)"
                raise RuntimeError(msg)
            return [[dict(segment) for segment in result.segments] for result in results]

        if self.diar_model is not None:
            from nemo_curator.models.audio.speaker_diarization.sortformer import parse_sortformer_segments

            predicted_segments = self.diar_model.diarize(
                audio=paths,
                batch_size=self.inference_batch_size,
            )
            return [parse_sortformer_segments(segments) for segments in predicted_segments]

        msg = "Diarization adapter is not initialized; call setup() before diarize()"
        raise RuntimeError(msg)

    def process(self, task: AudioTask) -> AudioTask:
        msg = f"{type(self).__name__} only supports process_batch"
        raise NotImplementedError(msg)

    def _prepare_task_item(self, task: AudioTask) -> dict[str, Any] | None:
        if not self.validate_input(task):
            msg = f"Task {task.task_id} missing required inputs for {type(self).__name__}: {self.inputs()}"
            raise ValueError(msg)
        source = self.waveform_key or str(task.data.get(self.audio_filepath_key, ""))
        try:
            if self._uses_resident(task):
                waveform = task.data[self.waveform_key]
                if self.fanout:
                    waveform = _channel_first_waveform(waveform)
                sample_rate = task.data[self.sample_rate_key]
                prepared = self._prepare_waveform(waveform, sample_rate)
                return {
                    "waveform": prepared,
                    "sample_rate": self.target_sample_rate,
                    "audio_seconds": prepared.size / self.target_sample_rate,
                    "task_id": task.task_id,
                }
            else:
                import soundfile

                audio_filepath = str(task.data[self.audio_filepath_key])
                info = soundfile.info(audio_filepath)
                return {
                    "audio_filepath": audio_filepath,
                    "audio_seconds": float(info.frames) / float(info.samplerate),
                    "task_id": task.task_id,
                }
        except Exception as exc:
            if self._is_memory_exhaustion(exc):
                raise
            if not self._uses_resident(task):
                # Header probing is only a scheduling hint. The selected
                # adapter owns file decoding and may support a path or codec
                # that soundfile cannot inspect, so preserve provider-native
                # decoding and sort an unprobed item last.
                logger.debug("Sortformer: could not probe task {} from {}: {}", task.task_id, source, exc)
                return {
                    "audio_filepath": str(task.data[self.audio_filepath_key]),
                    "audio_seconds": float("inf"),
                    "task_id": task.task_id,
                }
            if self.fail_on_audio_error:
                msg = f"Failed to prepare diarization audio for task {task.task_id} from {source}"
                raise RuntimeError(msg) from exc
            logger.warning("Sortformer: failed to prepare task {} from {}: {}", task.task_id, source, exc)
            self._write_empty_error(task, "audio_load_error")
            return None

    def _diarize_items(
        self,
        items: list[dict[str, Any]],
    ) -> list[DiarizationResult | None]:
        """Run one adapter batch, bisecting only known file-local failures."""
        if self._adapter is None:
            msg = "Diarization adapter is not initialized; setup() was not called"
            raise RuntimeError(msg)
        try:
            results = self._adapter.diarize_batch(items)
        except DiarizationInputError as exc:
            if all("waveform" in item for item in items) or self.fail_on_audio_error:
                raise
            if len(items) == 1:
                logger.warning("Sortformer: adapter failed task {}: {}", items[0]["task_id"], exc)
                return [None]
            midpoint = len(items) // 2
            return self._diarize_items(items[:midpoint]) + self._diarize_items(items[midpoint:])
        if len(results) != len(items):
            msg = f"Diarization adapter returned {len(results)} results for {len(items)} items (must match 1:1)"
            raise RuntimeError(msg)
        return list(results)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Prepare, diarize, scatter results, and preserve input task order."""
        if len(tasks) == 0:
            return []
        if self._adapter is None:
            msg = "Diarization adapter is not initialized; setup() was not called"
            raise RuntimeError(msg)

        pending: list[tuple[int, AudioTask, dict[str, Any]]] = []
        for index, task in enumerate(tasks):
            if task.data.get(self.read_error_key) or self._already_has_output(task):
                continue
            item = self._prepare_task_item(task)
            if item is None:
                continue
            pending.append((index, task, item))

        pending.sort(key=lambda value: (value[2]["audio_seconds"], value[0]))
        items = [value[2] for value in pending]
        if items:
            results = self._diarize_items(items)
            for (_, task, _), result in zip(pending, results, strict=True):
                if result is None:
                    self._write_empty_error(task, "audio_load_error")
                else:
                    self._write_result(task, result)

        if self.fanout:
            return [child for task in tasks for child in self._fanout_tasks(task)]
        return tasks

    def _already_has_output(self, task: AudioTask) -> bool:
        notes = task.data.get(self.notes_key)
        complete_manifest_output = (
            self.skip_if_output_exists
            and (self.num_speakers_key is None or self.num_speakers_key in task.data)
            and (not self.store_segments or self.diar_segments_key in task.data)
            and not self.fanout
            and not (isinstance(notes, dict) and self.name in notes)
        )
        if not complete_manifest_output:
            return False
        if self.rttm_out_dir is None:
            return True
        rttm_filepath = task.data.get(self.rttm_filepath_key)
        if not isinstance(rttm_filepath, str) or not rttm_filepath:
            return False
        try:
            relative_path = _validated_relative_rttm_path(rttm_filepath, label="RTTM manifest path")
            path = _resolve_under_rttm_root(self.rttm_out_dir, relative_path, label="RTTM manifest path")
        except ValueError as exc:
            logger.warning("Ignoring unsafe RTTM resume path for task {}: {}", task.task_id, exc)
            return False
        return path.is_file()

    def _write_empty_error(self, task: AudioTask, reason: str) -> None:
        if self.num_speakers_key is not None:
            task.data[self.num_speakers_key] = 0
        if self.store_segments or self.fanout:
            task.data[self.diar_segments_key] = []
        notes = task.data.get(self.notes_key)
        if not isinstance(notes, dict):
            notes = {}
            task.data[self.notes_key] = notes
        notes[self.name] = reason

    def _write_result(self, task: AudioTask, result: DiarizationResult) -> None:
        segments: list[DiarizationSegment] = []
        for segment in result.segments:
            if float(segment["end"]) <= float(segment["start"]):
                logger.warning("Skipping degenerate diarization segment: {}", segment)
                continue
            segments.append(dict(segment))
        if self.num_speakers_key is not None:
            task.data[self.num_speakers_key] = len({str(segment["speaker"]) for segment in segments})
        if self.store_segments or self.fanout:
            task.data[self.diar_segments_key] = [
                {
                    self.start_key: segment["start"],
                    self.end_key: segment["end"],
                    self.speaker_key: segment["speaker"],
                }
                for segment in segments
            ]
        if self.rttm_out_dir is not None:
            path = _write_rttm(
                segments,
                self._export_session_name(task),
                self.rttm_out_dir,
                shard_key=self._export_shard_key(task),
            )
            rttm_root = Path(self.rttm_out_dir).expanduser().resolve()
            task.data[self.rttm_filepath_key] = Path(path).relative_to(rttm_root).as_posix()
        notes = task.data.get(self.notes_key)
        if isinstance(notes, dict):
            notes.pop(self.name, None)

    def _session_name(self, task: AudioTask) -> str:
        session_name = task.data.get(self.session_name_key)
        if session_name:
            return str(session_name)
        filepath = task.data.get(self.audio_filepath_key)
        if filepath:
            return os.path.splitext(os.path.basename(str(filepath)))[0]
        return task.task_id

    @staticmethod
    def _rttm_shard_key(task: AudioTask) -> str:
        if not task.task_id:
            msg = "RTTM output requires a non-empty task_id"
            raise ValueError(msg)
        task_key = hashlib.sha256(task.task_id.encode()).hexdigest()
        shard_key = task._metadata.get("_shard_key")
        return f"{shard_key}/{task_key}" if isinstance(shard_key, str) and shard_key else task_key
