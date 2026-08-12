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

"""Bandwidth estimation stage."""

from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class BandwidthEstimationStage(ProcessingStage[AudioTask, AudioTask]):
    """
    Stage that estimates audio bandwidth by analyzing power spectra.

    Analyzes audio files to estimate their effective bandwidth by examining
    the power spectrum and determining the highest frequency with significant
    energy content above a threshold.

    Args:
        n_fft: Size of FFT window. Defaults to 512.
        stride_seconds: Time between successive FFT windows in seconds. Defaults to 0.01.
        top_db: Maximum decibel value for power spectrum normalization. Defaults to 100.0.
        frequency_threshold: Threshold in dB below peak for bandwidth estimation. Defaults to -50.0.
        audio_filepath_key: Key for the audio file path in the manifest. Defaults to "audio_filepath".
        segments_key: Key for the segments in the manifest. Defaults to "segments".

    Returns:
        The same data as in the input data, but with bandwidth estimates added to each segment.
    """

    n_fft: int = 512
    stride_seconds: float = 0.01
    top_db: float = 100.0
    frequency_threshold: float = -50.0
    audio_filepath_key: str = "audio_filepath"
    segments_key: str = "segments"

    # Stage metadata
    name: str = "BandwidthEstimation"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, "metrics"]

    def validate_input(self, task: AudioTask) -> bool:
        """OR-shaped: needs audio_filepath AND (segments OR duration)."""
        data = task.data
        if not hasattr(data, self.audio_filepath_key):
            logger.error(f"Task {task.task_id} missing '{self.audio_filepath_key}'")
            return False
        if hasattr(data, self.segments_key) or hasattr(data, "duration"):
            return True
        logger.error(f"Task {task.task_id} missing required attributes: need '{self.segments_key}' OR 'duration'")
        return False

    def _estimate_bandwidth(self, audio: "np.ndarray", sample_rate: int) -> int:
        """Estimate the bandwidth of an audio signal."""
        hop_length = int(sample_rate * self.stride_seconds)

        spec = librosa.stft(y=audio, n_fft=self.n_fft, hop_length=hop_length, window="blackmanharris")
        power_spec = np.abs(spec) ** 2
        power_spec = np.mean(power_spec, axis=1)
        power_spec = librosa.power_to_db(power_spec, ref=self.n_fft, top_db=self.top_db)

        bandwidth = 0
        peak = np.max(power_spec)
        freq_width = sample_rate / self.n_fft

        for idx in range(len(power_spec) - 1, -1, -1):
            if power_spec[idx] - peak > self.frequency_threshold:
                bandwidth = idx * freq_width
                break

        return bandwidth

    def get_bandwidth(self, audio_segment: dict[str, Any], audio: "np.ndarray", sample_rate: int) -> None:
        """Get the bandwidth of an audio segment."""
        segment_speaker = audio_segment.get("speaker")
        segment_text = audio_segment.get("text")

        if (segment_speaker is not None and segment_speaker == "no-speaker") or (
            segment_text is not None and segment_text.strip() == ""
        ):
            return

        start = audio_segment.get("start", 0.0)
        end = audio_segment.get("end", audio_segment.get("duration", 0.0))
        if end is None or start >= end:
            msg = f"[{self.name}] Invalid segment time range: start={start}, end={end}"
            raise ValueError(msg)

        segment_audio_array = audio[int(start * sample_rate) : int(end * sample_rate)]
        bandwidth = self._estimate_bandwidth(segment_audio_array, sample_rate)

        if "metrics" not in audio_segment:
            audio_segment["metrics"] = {}

        audio_segment["metrics"]["bandwidth"] = int(bandwidth)

    def process(self, task: AudioTask) -> AudioTask:
        """Estimate bandwidth for audio entry."""
        data_entry = task.data
        audio_path = data_entry.get(self.audio_filepath_key)
        if not audio_path:
            msg = (
                f"[{self.name}] Missing '{self.audio_filepath_key}' for entry: "
                f"{data_entry.get('audio_item_id', 'unknown')}"
            )
            raise ValueError(msg)
        try:
            audio, sample_rate = librosa.load(path=audio_path, sr=None)
        except Exception as ex:
            msg = f"[{self.name}] Failed to load audio: {audio_path}"
            raise RuntimeError(msg) from ex

        if self.segments_key in data_entry:
            for segment in data_entry[self.segments_key]:
                try:
                    self.get_bandwidth(segment, audio, sample_rate)
                except ValueError as ex:
                    logger.warning(f"[{self.name}] skipping segment in {task.task_id}: {ex}")
                    segment.setdefault("metrics", {})["metric_skip_reason"] = str(ex)
        else:
            self.get_bandwidth(data_entry, audio, sample_rate)

        return task
