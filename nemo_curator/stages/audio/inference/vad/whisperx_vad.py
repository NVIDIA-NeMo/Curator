# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
WhisperX VAD for NeMo Curator.

Provides WhisperXVADModel (shared VAD logic for pyannote and standalone VAD)
and WhisperXVADStage (ProcessingStage for VAD-only pipeline use).
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import soundfile as sf
import torch
from loguru import logger
from whisperx.audio import SAMPLE_RATE
from whisperx.vads.pyannote import Pyannote, load_vad_model

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract, StaticHints
from nemo_curator.stages.audio._agent._residency import (
    InputResidency,
    cleanup_temp_files,
    resolve_audio_path,
    validate_input_residency,
)
from nemo_curator.stages.audio.common import get_audio_duration
from nemo_curator.stages.audio.inference.base import (
    _channel_first_waveform,
    _fanout_audio_segment,
    _fanout_original_file,
    _fanout_path_keys,
    _inference_audio_input_spec,
    _inference_audio_read_specs,
    _resident_audio_duration,
    _stable_audio_identity,
    _stable_source_path,
    _validate_fanout_key_contract,
    _validate_inference_audio_input,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


class WhisperXVADModel:
    """Shared VAD model and get_vad_segments logic for PyAnnote and standalone VAD.

    Used by PyAnnoteDiarizationStage for sub-segment VAD and by WhisperXVADStage
    for VAD-only processing.
    """

    def __init__(
        self,
        device: str = "cuda",
        vad_onset: float = 0.5,
        vad_offset: float = 0.363,
        use_auth_token: str | None = None,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            msg = "CUDA device requested but not available. Set device='cpu' to run without GPU."
            raise RuntimeError(msg)
        self._device = device
        self._vad_onset = vad_onset
        self._vad_offset = vad_offset
        default_vad_options = {
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
        }

        prev = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "true"
        try:
            self._model = load_vad_model(torch.device(device), token=use_auth_token, **default_vad_options)
        finally:
            if prev is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = prev

    def to(self, device: str) -> None:
        """Move the model to the given device."""
        self._model = self._model.to(torch.device(device))

    def get_vad_segments(
        self,
        audio: "np.ndarray",
        merge_max_length: float,
        sample_rate: int = SAMPLE_RATE,
    ) -> list[dict]:
        """Get voice activity detection segments for the given audio.

        Args:
            audio: NumPy array of shape (C, N).
            merge_max_length: Maximum length for merging chunks in seconds.
            sample_rate: Sample rate of the audio.

        Returns:
            List of VAD segment dicts with "start" and "end" keys.
        """
        vad_segments = self._model(
            {
                "waveform": torch.from_numpy(audio),
                "sample_rate": sample_rate,
            }
        )
        return Pyannote.merge_chunks(vad_segments, merge_max_length, onset=self._vad_onset)


@dataclass
class WhisperXVADStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    Stage that performs Voice Activity Detection (VAD) using WhisperX's VAD model.

    Adds VAD segments to each entry under segments_key (e.g. "vad_segments").
    Entries shorter than min_length are skipped (not emitted).
    """

    min_length: float = 0.5
    max_length: float = 40.0
    vad_onset: float = 0.5
    vad_offset: float = 0.363
    segments_key: str = "vad_segments"
    audio_filepath_key: str = "resampled_audio_filepath"
    waveform_key: str = field(default="waveform", kw_only=True)
    sample_rate_key: str = field(default="sample_rate", kw_only=True)
    input_residency: InputResidency = field(default="file", kw_only=True)
    allow_audio_filepath_fallback: bool = field(default=False, kw_only=True)
    fanout: bool = field(default=False, kw_only=True)
    start_key: str = field(default="start", kw_only=True)
    end_key: str = field(default="end", kw_only=True)
    start_ms_key: str = field(default="start_ms", kw_only=True)
    end_ms_key: str = field(default="end_ms", kw_only=True)
    duration_key: str = field(default="duration", kw_only=True)
    segment_num_key: str = field(default="segment_num", kw_only=True)
    original_file_key: str = field(default="original_file", kw_only=True)

    name: str = "WhisperXVAD"
    resources: Resources = field(default_factory=lambda: Resources(gpus=1))

    _vad_model: Any = field(default=None, repr=False)

    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        cardinality_options=["1:1", "1:N fan-out"],
        gates=Gates(
            writes_to_disk=True,
            output_path_params=[],
            requires_gpu=True,
            requires_internet_first_run=True,
            per_row_independent=True,
        ),
    )

    def __post_init__(self) -> None:
        validate_input_residency(self.input_residency, stage_name=self.name)
        self.is_resumable = not self.fanout
        if self.fanout:
            _validate_fanout_key_contract(
                stage_name=self.name,
                audio_filepath_key=self.audio_filepath_key,
                output_keys=[
                    self.waveform_key,
                    self.sample_rate_key,
                    self.start_key,
                    self.end_key,
                    self.start_ms_key,
                    self.end_ms_key,
                    self.duration_key,
                    self.segment_num_key,
                    self.original_file_key,
                ],
                removed_container_keys=(self.segments_key,),
            )

    def inputs(self) -> tuple[list[str], list[str]]:
        return _inference_audio_input_spec(
            self.input_residency,
            audio_filepath_key=self.audio_filepath_key,
            waveform_key=self.waveform_key,
            sample_rate_key=self.sample_rate_key,
        )

    def validate_input(self, task: AudioTask) -> bool:
        return _validate_inference_audio_input(
            task,
            stage_name=self.name,
            residency=self.input_residency,
            audio_filepath_keys=tuple(
                dict.fromkeys(
                    [
                        self.audio_filepath_key,
                        *(["audio_filepath"] if self.allow_audio_filepath_fallback else []),
                    ]
                )
            ),
            waveform_key=self.waveform_key,
            sample_rate_key=self.sample_rate_key,
        )

    def outputs(self) -> tuple[list[str], list[str]]:
        if self.fanout:
            return [], [
                self.waveform_key,
                self.sample_rate_key,
                self.start_key,
                self.end_key,
                self.start_ms_key,
                self.end_ms_key,
                self.duration_key,
                self.segment_num_key,
                self.original_file_key,
            ]
        return [], [self.audio_filepath_key, self.segments_key]

    def describe(self) -> StageContract:
        if self.fanout:
            writes = [
                self.waveform_key,
                self.sample_rate_key,
                self.start_key,
                self.end_key,
                self.start_ms_key,
                self.end_ms_key,
                self.duration_key,
                self.segment_num_key,
                self.original_file_key,
            ]
            cardinality = "1:N fan-out"
        else:
            writes = [self.segments_key]
            cardinality = "1:1"
        return StageContract(
            reads_one_of=_inference_audio_read_specs(
                self.input_residency,
                audio_filepath_key=self.audio_filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
                fallback_audio_filepath_keys=("audio_filepath",) if self.allow_audio_filepath_fallback else (),
            ),
            writes=IOSpec(data_keys=writes, produces=["tensor"] if self.fanout else []),
            cardinality=cardinality,
            cardinality_options=["1:1", "1:N fan-out"],
            iteration_key=self.segment_num_key if self.fanout else None,
            removes_keys=(
                list(dict.fromkeys([*_fanout_path_keys(self.audio_filepath_key), self.segments_key]))
                if self.fanout
                else []
            ),
            # Speech boundaries come from this file's own samples against the configured
            # onset/offset, and a fan-out child is a slice of the row it came from.
            gates=Gates(
                requires_gpu=self.resources.requires_gpu,
                requires_internet_first_run=True,
                writes_to_disk=self.input_residency != "file",
                output_path_params=[] if self.input_residency != "file" else None,
                per_row_independent=True,
            ),
        )

    def ray_stage_spec(self) -> dict[str, Any]:
        if self.fanout:
            return {RayStageSpecKeys.IS_FANOUT_STAGE: True}
        return {}

    def _segment_child_data(  # noqa: PLR0913
        self,
        item: dict[str, Any],
        segment: dict[str, Any],
        segment_num: int,
        waveform: Any,  # noqa: ANN401
        sample_rate: int,
        original_file: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        excluded = {
            self.segments_key,
            self.waveform_key,
            self.sample_rate_key,
            *_fanout_path_keys(self.audio_filepath_key),
        }
        child = {k: v for k, v in item.items() if k not in excluded}
        raw_start = float(segment.get("start", 0.0))
        raw_end = float(segment.get("end", raw_start))
        # copy segment extras but never whisperx's raw internal keys: 'segments'
        # (a list of (start, end) tuples) collides with the semantic segments
        # role and crashes dict-shaped consumers; raw start/end are re-emitted
        # under the configured keys below.
        child.update({k: v for k, v in segment.items() if k not in {"start", "end", "segments", *excluded}})
        child[self.waveform_key], start, end = _fanout_audio_segment(
            waveform, sample_rate, start=raw_start, end=raw_end
        )
        child[self.sample_rate_key] = int(sample_rate)
        child[self.start_key] = start
        child[self.end_key] = end
        child[self.start_ms_key] = round(start * 1000)
        child[self.end_ms_key] = round(end * 1000)
        child[self.duration_key] = max(0.0, end - start)
        child[self.segment_num_key] = segment_num
        child[self.original_file_key] = original_file
        return child

    def _fanout_segments(
        self,
        task: AudioTask,
        segments: list[dict[str, Any]],
        waveform: Any,  # noqa: ANN401
        sample_rate: int,
        original_file: Any,  # noqa: ANN401
    ) -> list[AudioTask]:
        return [
            AudioTask(
                dataset_name=task.dataset_name,
                filepath_key=task.filepath_key or self.audio_filepath_key,
                data=self._segment_child_data(
                    task.data,
                    segment,
                    index,
                    waveform,
                    sample_rate,
                    original_file,
                ),
                _metadata=dict(task._metadata or {}),
                _stage_perf=list(task._stage_perf),
            )
            for index, segment in enumerate(segments)
        ]

    @property
    def _device(self) -> str:
        """Derive device from resources configuration."""
        return "cuda" if self.resources.requires_gpu else "cpu"

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        """Setup stage on node."""
        if self._vad_model is None:
            self._vad_model = WhisperXVADModel(
                device="cpu",
                vad_onset=self.vad_onset,
                vad_offset=self.vad_offset,
            )

    def setup(self, _: WorkerMetadata | None = None) -> None:
        if self._vad_model is None:
            self._vad_model = WhisperXVADModel(
                device=self._device,
                vad_onset=self.vad_onset,
                vad_offset=self.vad_offset,
            )
        self._vad_model.to(self._device)
        logger.info(f"[{self.name}] Initialized WhisperX VAD on {self._device}")

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        t0 = time.perf_counter()
        data_entry = task.data
        if not self.validate_input(task):
            msg = f"[{self.name}] task {task.task_id!r} has no valid {self.input_residency} audio input"
            raise ValueError(msg)
        resident_waveform = (
            _channel_first_waveform(data_entry[self.waveform_key])
            if self.input_residency != "file"
            and data_entry.get(self.waveform_key) is not None
            and data_entry.get(self.sample_rate_key) is not None
            else None
        )
        resident_sample_rate = int(data_entry[self.sample_rate_key]) if resident_waveform is not None else None
        source_path = (
            None
            if resident_waveform is not None
            else _stable_source_path(
                data_entry,
                self.audio_filepath_key,
                *(["audio_filepath"] if self.allow_audio_filepath_fallback else []),
            )
        )
        audio_input = data_entry if resident_waveform is None else {**data_entry, self.waveform_key: resident_waveform}
        temp_paths: list[str] = []
        file_path = resolve_audio_path(
            audio_input,
            residency=self.input_residency,  # type: ignore[arg-type]
            audio_filepath_key=self.audio_filepath_key,
            waveform_key=self.waveform_key,
            sample_rate_key=self.sample_rate_key,
            register_temp=temp_paths,
        )
        if file_path is None and self.allow_audio_filepath_fallback and self.audio_filepath_key != "audio_filepath":
            # Composability fallback: pipelines that did not run ResampleAudioStage
            # only have the original audio_filepath.
            file_path = resolve_audio_path(
                audio_input,
                residency=self.input_residency,  # type: ignore[arg-type]
                audio_filepath_key="audio_filepath",
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
                register_temp=temp_paths,
            )
        if file_path is None:
            msg = f"[{self.name}] Missing audio input for key '{self.audio_filepath_key}'"
            raise ValueError(msg)
        try:
            duration = (
                _resident_audio_duration(resident_waveform, resident_sample_rate)
                if resident_waveform is not None and resident_sample_rate is not None
                else data_entry.get("duration", get_audio_duration(file_path))
            )
            if duration < self.min_length:
                logger.warning(f"Skipping {file_path} because it is less than {self.min_length} seconds")
                data_entry[self.segments_key] = []
                self._log_metrics(
                    {
                        "process_time": time.perf_counter() - t0,
                        "audio_duration": duration,
                        "vad_segments_detected": 0,
                        "skipped_short": 1.0,
                    }
                )
                return [] if self.fanout else task

            data, sr = sf.read(file_path, dtype="float32")
            audio = np.expand_dims(data, axis=0) if data.ndim == 1 else data.T
            vad_segments = self._vad_model.get_vad_segments(audio, self.max_length, sample_rate=sr)
            data_entry[self.segments_key] = vad_segments
            self._log_metrics(
                {
                    "process_time": time.perf_counter() - t0,
                    "audio_duration": duration,
                    "vad_segments_detected": len(vad_segments),
                    "skipped_short": 0.0,
                }
            )
            if self.fanout:
                source_waveform = resident_waveform if resident_waveform is not None else audio
                source_sample_rate = resident_sample_rate if resident_sample_rate is not None else int(sr)
                stable_identity = _stable_audio_identity(
                    data_entry,
                    source_waveform,
                    source_sample_rate,
                    source_path=source_path,
                )
                original_file = _fanout_original_file(
                    data_entry,
                    original_file_key=self.original_file_key,
                    source_path=source_path,
                    stable_identity=stable_identity,
                )
                return self._fanout_segments(
                    task,
                    vad_segments,
                    source_waveform,
                    source_sample_rate,
                    original_file,
                )
            return task
        finally:
            cleanup_temp_files(temp_paths)
