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
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from fsspec.core import url_to_fs
from loguru import logger

# Import pyannote components
from pyannote.audio import Pipeline as PyAnnotePipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio.common import get_audio_duration
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
class PyAnnoteDiarizationStage(ProcessingStage[AudioTask, AudioTask]):
    """
    Stage that performs speaker diarization and overlap detection using PyAnnote.

    Identifies different speakers and detects overlapping speech segments.

    Args:
        hf_token: HuggingFace authentication token
        segmentation_batch_size: Batch size for segmentation
        embedding_batch_size: Batch size for speaker embeddings
        min_length: Minimum segment length in seconds
        max_length: Maximum segment length in seconds
        xenna_num_workers: If set, caps total Xenna/Ray workers for this stage (cluster-wide).
        xenna_num_workers_per_node: If set, caps workers per node. Omitted keys use Xenna defaults (autoscaling).
    """

    hf_token: str

    # Diarization pipeline model ID on HuggingFace
    model_name: str = "pyannote/speaker-diarization-3.1"

    # Model parameters
    segmentation_batch_size: int = 128
    embedding_batch_size: int = 128

    # Segment length constraints
    min_length: float = 0.5
    max_length: float = 40.0

    audio_filepath_key: str = "resampled_audio_filepath"
    segments_key: str = "segments"
    overlap_segments_key: str = "overlap_segments"

    # Stage metadata
    name: str = "PyAnnoteDiarization"
    resources: Resources = field(default_factory=lambda: Resources(gpus=1))

    # Xenna executor (optional; unset = default autoscaling)
    xenna_num_workers: int | None = None
    xenna_num_workers_per_node: int | None = None

    # Internal state (not serialized, initialized in setup() to allow deepcopy)
    _pipeline: Any = field(default=None, repr=False)
    _vad_model: Any = field(default=None, repr=False)  # WhisperXVADModel
    _rng: random.Random | None = field(default=None, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, self.segments_key, self.overlap_segments_key]

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.xenna_num_workers is not None:
            spec["num_workers"] = self.xenna_num_workers
        if self.xenna_num_workers_per_node is not None:
            spec["num_workers_per_node"] = self.xenna_num_workers_per_node
        return spec

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
                vad_onset=0.5,
                vad_offset=0.363,
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
                vad_onset=0.5,
                vad_offset=0.363,
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

    def process(self, task: AudioTask) -> AudioTask:
        """Process a single entry for diarization and overlap detection."""
        t0 = time.perf_counter()
        data_entry = task.data
        file_path = data_entry.get(self.audio_filepath_key)
        if not file_path:
            msg = f"[{self.name}] Missing key '{self.audio_filepath_key}' in entry: {data_entry.get('audio_item_id', 'unknown')}"
            raise ValueError(msg)

        # Load audio using soundfile (avoids torchcodec/FFmpeg dependency)
        data, fs = sf.read(file_path, dtype="float32")
        s = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data.T)
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
        logger.info(f"Writing {len(diarization._tracks)} turns to RTTM file")
        rttm_filepath = os.path.splitext(file_path)[0] + ".rttm"
        rttm_fs, rttm_path = url_to_fs(rttm_filepath)
        with rttm_fs.open(rttm_path, "w") as rttm_file:
            diarization.write_rttm(rttm_file)

        segments = []
        overlap_segments = []

        # Process speaker turns
        for speech_turn, _track, speaker in diarization.itertracks(yield_label=True):
            if "audio_item_id" in data_entry:
                speaker_id = data_entry["audio_item_id"] + "_" + speaker
            elif "speaker_id" in data_entry:
                speaker_id = data_entry["speaker_id"] + "_" + speaker
            elif self.audio_filepath_key in data_entry:
                speaker_id = Path(data_entry[self.audio_filepath_key]).stem + "_" + speaker
            else:
                msg = f"No speaker identifier in {file_path}"
                raise ValueError(msg)

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
        audio_duration = data_entry.get("duration", get_audio_duration(file_path))
        add_non_speaker_segments(segments, audio_duration, self.max_length)

        # Update entry
        data_entry[self.segments_key] = segments
        data_entry[self.overlap_segments_key] = overlap_segments

        speakers = {seg["speaker"] for seg in segments if seg.get("speaker") != "no-speaker"}
        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "segments_detected": len(segments),
                "overlap_segments_detected": len(overlap_segments),
                "speakers_detected": len(speakers),
                "audio_duration": audio_duration,
            }
        )
        return task
