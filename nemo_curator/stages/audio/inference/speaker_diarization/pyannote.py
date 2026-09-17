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
PyAnnote Diarization and Overlap Detection Stage.
"""

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

import soundfile as sf
import torch
from fsspec.core import url_to_fs
from loguru import logger

# Import pyannote components
from pyannote.audio import Pipeline as PyAnnotePipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment

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
from nemo_curator.stages.audio.inference.vad.whisperx_vad import WhisperXVADModel
from nemo_curator.stages.audio.tagging.utils import add_non_speaker_segments
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


def has_overlap(turn: Segment, overlaps: list) -> bool:
    """Check if a given turn overlaps with any segment in the overlaps list.

    Args:
        turn: A segment representing a speech turn
        overlaps: List of overlap segments, sorted by start time

    Returns:
        True if the turn overlaps with any segment, False otherwise
    """
    turn_overlaps = False
    for overlap in overlaps:
        if overlap.start > turn.end:
            # Overlap happens after turn, no need to keep looping since overlaps is sorted
            break
        elif overlap.start >= turn.start and overlap.start < turn.end:
            # overlap starts during turn
            turn_overlaps = True
            break
        elif (overlap.end < turn.end) and (overlap.end > turn.start):
            # overlap ends during turn
            turn_overlaps = True
            break
        elif overlap.start < turn.start and overlap.end > turn.end:
            # Overlap completely contains the turn
            turn_overlaps = True
            break
    return turn_overlaps


@dataclass
class PyAnnoteDiarizationStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    Stage that performs speaker diarization and overlap detection using PyAnnote.

    Identifies different speakers and detects overlapping speech segments.

    Args:
        hf_token: Optional Hugging Face token. Local pipeline directories do not require one.
        segmentation_batch_size: Batch size for segmentation
        embedding_batch_size: Batch size for speaker embeddings
        min_length: Minimum segment length in seconds
        max_length: Maximum segment length in seconds
        num_speakers_key: Optional output key for the distinct-speaker count derived
            from the diarization result. Disabled by default for legacy compatibility.
        xenna_num_workers: If set, caps workers cluster-wide. Prefer ``with_(num_workers=...)`` for new code.
    """

    hf_token: str | None = field(default=None, repr=False)

    # Diarization pipeline model ID on HuggingFace
    model_name: str = "pyannote/speaker-diarization-3.1"

    # Model parameters
    segmentation_batch_size: int = 128
    embedding_batch_size: int = 128

    # Segment length constraints
    min_length: float = 0.5
    max_length: float = 40.0

    audio_filepath_key: str = "resampled_audio_filepath"
    waveform_key: str = field(default="waveform", kw_only=True)
    sample_rate_key: str = field(default="sample_rate", kw_only=True)
    segments_key: str = "segments"
    overlap_segments_key: str = "overlap_segments"
    num_speakers_key: str | None = field(default=None, kw_only=True)
    input_residency: InputResidency = field(default="file", kw_only=True)
    allow_audio_filepath_fallback: bool = field(default=False, kw_only=True)
    write_rttm: bool = field(default=True, kw_only=True)
    vad_onset: float = field(default=0.5, kw_only=True)
    vad_offset: float = field(default=0.363, kw_only=True)
    fanout: bool = field(default=False, kw_only=True)
    start_key: str = field(default="start", kw_only=True)
    end_key: str = field(default="end", kw_only=True)
    start_ms_key: str = field(default="start_ms", kw_only=True)
    end_ms_key: str = field(default="end_ms", kw_only=True)
    duration_key: str = field(default="duration", kw_only=True)
    segment_num_key: str = field(default="segment_num", kw_only=True)
    speaker_key: str = field(default="speaker", kw_only=True)
    original_file_key: str = field(default="original_file", kw_only=True)
    overlap_key: str = field(default="is_overlap", kw_only=True)
    INTERNAL_KEY_FIELDS: ClassVar[frozenset[str]] = frozenset({"overlap_key"})

    # Stage metadata
    name: str = "PyAnnoteDiarization"
    resources: Resources = field(default_factory=lambda: Resources(gpus=1))

    # Xenna executor (optional; unset = default autoscaling)
    xenna_num_workers: int | None = None

    # Internal state (not serialized, initialized in setup() to allow deepcopy)
    _pipeline: Any = field(default=None, repr=False)
    _vad_model: Any = field(default=None, repr=False)  # WhisperXVADModel
    _rng: random.Random | None = field(default=None, repr=False)

    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        cardinality_options=["1:1", "1:N fan-out"],
        gates=Gates(
            writes_to_disk=True,
            output_path_params=[],
            requires_gpu=True,
            requires_internet_first_run=True,
            runtime_secrets=["HF_TOKEN"],
            per_row_independent=False,
        ),
    )

    def __post_init__(self) -> None:
        validate_input_residency(self.input_residency, stage_name=self.name)
        self.is_resumable = not self.fanout
        if self.num_speakers_key is not None and self.num_speakers_key in {
            self.audio_filepath_key,
            self.segments_key,
            self.overlap_segments_key,
        }:
            msg = "num_speakers_key must be distinct from path and segment output keys when enabled"
            raise ValueError(msg)
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
                    self.speaker_key,
                    self.original_file_key,
                    *([self.num_speakers_key] if self.num_speakers_key is not None else []),
                    self.overlap_key,
                ],
                removed_container_keys=(self.segments_key, self.overlap_segments_key),
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
            output_keys = [
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
                self.overlap_key,
            ]
            if self.num_speakers_key is not None:
                output_keys.append(self.num_speakers_key)
            return [], output_keys
        output_keys = [self.audio_filepath_key, self.segments_key, self.overlap_segments_key]
        if self.num_speakers_key is not None:
            output_keys.append(self.num_speakers_key)
        return [], output_keys

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
                self.speaker_key,
                self.original_file_key,
                self.overlap_key,
            ]
            if self.num_speakers_key is not None:
                writes.append(self.num_speakers_key)
            cardinality = "1:N fan-out"
        else:
            writes = [self.segments_key, self.overlap_segments_key]
            if self.num_speakers_key is not None:
                writes.append(self.num_speakers_key)
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
            # Fan-out children are one-per-diarization-segment, as Sortformer/WhisperX declare.
            iteration_key=self.segments_key if self.fanout else None,
            removes_keys=(
                list(
                    dict.fromkeys(
                        [
                            *_fanout_path_keys(self.audio_filepath_key),
                            self.segments_key,
                            self.overlap_segments_key,
                        ]
                    )
                )
                if self.fanout
                else []
            ),
            gates=Gates(
                requires_gpu=self.resources.requires_gpu,
                writes_to_disk=self.write_rttm or self.input_residency != "file",
                output_path_params=[] if self.write_rttm or self.input_residency != "file" else None,
                # Even a local diarization pipeline constructs WhisperXVADModel,
                # whose weights are downloaded on first use.
                requires_internet_first_run=True,
                # An empty token cannot authenticate, so ``""`` is as missing as ``None``.
                runtime_secrets=(
                    ["HF_TOKEN"]
                    if not self.hf_token and not os.path.exists(os.path.expanduser(self.model_name))
                    else []
                ),
                # ``add_vad_segments`` draws from one unseeded ``random.Random()`` built in
                # ``setup()``, so the durations a file gets depend on how many files that worker
                # segmented before it.
                per_row_independent=False,
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
            self.overlap_segments_key,
            self.waveform_key,
            self.sample_rate_key,
            *_fanout_path_keys(self.audio_filepath_key),
        }
        child = {k: v for k, v in item.items() if k not in excluded}
        child.update({k: v for k, v in segment.items() if k not in {"start", "end", "speaker", *excluded}})
        raw_start = float(segment.get("start", 0.0))
        raw_end = float(segment.get("end", raw_start))
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
        if "speaker" in segment:
            child[self.speaker_key] = segment["speaker"]
        child[self.original_file_key] = original_file
        child[self.overlap_key] = bool(segment.get(self.overlap_key, False))
        if self.num_speakers_key is not None:
            child[self.num_speakers_key] = item[self.num_speakers_key]
        return child

    def _fanout_segments(
        self,
        task: AudioTask,
        segments: list[dict[str, Any]],
        waveform: Any,  # noqa: ANN401
        sample_rate: int,
        original_file: Any,  # noqa: ANN401
    ) -> list[AudioTask]:
        fanout_segments = [
            {**segment, self.overlap_key: False} for segment in segments if segment.get("speaker") != "no-speaker"
        ]
        fanout_segments.extend({**segment, self.overlap_key: True} for segment in task.data[self.overlap_segments_key])
        fanout_segments.sort(key=lambda segment: (segment.get("start", 0.0), segment.get("end", 0.0)))
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
            for index, segment in enumerate(fanout_segments)
        ]

    def num_workers(self) -> int | None:
        return self.xenna_num_workers

    @property
    def _device(self) -> str:
        """Derive device from resources configuration."""
        return "cuda" if self.resources.requires_gpu else "cpu"

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        """Download model weights (called once per node)."""
        if self._pipeline is None:
            self._pipeline = PyAnnotePipeline.from_pretrained(self.model_name, token=self.hf_token)
        if self._vad_model is None:
            self._vad_model = WhisperXVADModel(
                device="cpu",
                vad_onset=self.vad_onset,
                vad_offset=self.vad_offset,
            )

    def setup(self, _: WorkerMetadata | None = None) -> None:
        """Load models to device (called per replica before processing)."""
        if self._pipeline is None:
            self._pipeline = PyAnnotePipeline.from_pretrained(self.model_name, token=self.hf_token)
        self._pipeline.segmentation_batch_size = self.segmentation_batch_size
        self._pipeline.embedding_batch_size = self.embedding_batch_size

        if self._vad_model is None:
            self._vad_model = WhisperXVADModel(
                device=self._device,
                vad_onset=self.vad_onset,
                vad_offset=self.vad_offset,
            )

        self._pipeline.to(torch.device(self._device))
        self._vad_model.to(self._device)

        self._rng = random.Random()  # noqa: S311
        logger.info(f"[{self.name}] Initialized PyAnnote diarization on {self._device}")

    def add_vad_segments(  # noqa: PLR0913
        self,
        audio: torch.Tensor,
        fs: int,
        start: float,
        end: float,
        segments: list[dict],
        speaker_id: str,
    ) -> None:
        """Add VAD segments for a given audio region to the segments list."""
        segment_duration = end - start

        if segment_duration > self.max_length:
            audio_seg = audio[:, int(start * fs) : int(end * fs)]
            vad_segments = self._vad_model.get_vad_segments(audio_seg.numpy(), self.max_length, sample_rate=fs)
            i = 0
            n = len(vad_segments)

            while i < n:
                random_duration = self._rng.uniform(self.min_length, self.max_length)
                start_seg = vad_segments[i]["start"]
                end_seg = vad_segments[i]["end"]

                if end_seg - start_seg >= random_duration:
                    segments.append(
                        {
                            "speaker": speaker_id,
                            "start": start + start_seg,
                            "end": start + end_seg,
                        }
                    )
                    i += 1
                    continue

                while i < n and (vad_segments[i]["end"] - start_seg) < random_duration:
                    end_seg = vad_segments[i]["end"]
                    i += 1

                segments.append(
                    {
                        "speaker": speaker_id,
                        "start": start + start_seg,
                        "end": start + end_seg,
                    }
                )
        else:
            segments.append({"speaker": speaker_id, "start": start, "end": end})

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        """Process a single entry for diarization and overlap detection."""
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
            entry_id = data_entry.get("audio_item_id", "unknown")
            msg = f"[{self.name}] Missing key '{self.audio_filepath_key}' in entry: {entry_id}"
            raise ValueError(msg)
        if self.write_rttm and file_path in temp_paths:
            temp_paths.append(os.path.splitext(file_path)[0] + ".rttm")
        try:
            return self._diarize_file(
                task,
                data_entry,
                file_path,
                t0,
                resident_waveform,
                resident_sample_rate,
                source_path,
            )
        finally:
            cleanup_temp_files(temp_paths)

    def _diarize_file(  # noqa: PLR0913
        self,
        task: AudioTask,
        data_entry: dict[str, Any],
        file_path: str,
        t0: float,
        resident_waveform: Any | None,  # noqa: ANN401
        resident_sample_rate: int | None,
        source_path: str | None,
    ) -> AudioTask | list[AudioTask]:
        # Load audio using soundfile (avoids torchcodec/FFmpeg dependency)
        data, fs = sf.read(file_path, dtype="float32")
        s = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data.T)
        source_waveform = resident_waveform if resident_waveform is not None else s.numpy()
        source_sample_rate = resident_sample_rate if resident_sample_rate is not None else int(fs)
        identity_source_path = source_path or _stable_source_path(data_entry, self.original_file_key)
        stable_identity = _stable_audio_identity(
            data_entry,
            source_waveform,
            source_sample_rate,
            source_path=identity_source_path,
            explicit_keys=("audio_item_id", "speaker_id"),
            fallback_keys=("session_name",),
        )
        original_file = _fanout_original_file(
            data_entry,
            original_file_key=self.original_file_key,
            source_path=source_path,
            stable_identity=stable_identity,
        )
        logger.info(f"Processing {file_path}")

        # Run diarization
        with ProgressHook() as hook:
            result = self._pipeline({"waveform": s, "sample_rate": fs}, hook=hook)

        # pyannote-audio 4.x returns DiarizeOutput; extract the Annotation
        diarization = result.speaker_diarization if hasattr(result, "speaker_diarization") else result

        overlaps = diarization.get_overlap().segments_list_

        # Crop to audio length (fix for PyAnnote bug)
        diarization = diarization.crop(Segment(0, len(s[0]) / fs))

        # Write RTTM file (cloud-aware via fsspec)
        if self.write_rttm:
            logger.info(f"Writing {len(diarization._tracks)} turns to RTTM file")
            rttm_filepath = os.path.splitext(file_path)[0] + ".rttm"
            rttm_fs, rttm_path = url_to_fs(rttm_filepath)
            with rttm_fs.open(rttm_path, "w") as rttm_file:
                diarization.write_rttm(rttm_file)

        segments = []
        overlap_segments = []

        # Process speaker turns
        for speech_turn, _track, speaker in diarization.itertracks(yield_label=True):
            speaker_id = f"{stable_identity}_{speaker}"

            if has_overlap(speech_turn, overlaps):
                overlap_segments.append(
                    {
                        "speaker": speaker_id,
                        "start": speech_turn.start,
                        "end": speech_turn.end,
                    }
                )
            else:
                speech_duration = speech_turn.end - speech_turn.start
                if speech_duration > self.min_length:
                    self.add_vad_segments(
                        s,
                        fs,
                        speech_turn.start,
                        speech_turn.end,
                        segments,
                        speaker_id,
                    )

        # Add non-speaker segments
        audio_duration = (
            _resident_audio_duration(source_waveform, source_sample_rate)
            if resident_waveform is not None
            else data_entry.get("duration", get_audio_duration(file_path))
        )
        add_non_speaker_segments(segments, audio_duration, self.max_length)

        # Update entry
        data_entry[self.segments_key] = segments
        data_entry[self.overlap_segments_key] = overlap_segments

        # Distinct speakers across turns AND overlap-only turns; "no-speaker" is a
        # silence/VAD placeholder, not a real speaker.
        speakers = {seg["speaker"] for seg in (*segments, *overlap_segments) if seg.get("speaker") != "no-speaker"}
        if self.num_speakers_key is not None:
            data_entry[self.num_speakers_key] = len(speakers)
        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "segments_detected": len(segments),
                "overlap_segments_detected": len(overlap_segments),
                "speakers_detected": len(speakers),
                "audio_duration": audio_duration,
            }
        )
        if self.fanout:
            return self._fanout_segments(
                task,
                segments,
                source_waveform,
                source_sample_rate,
                original_file,
            )
        return task
