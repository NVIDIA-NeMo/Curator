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

    Returns:
        The same data as in the input data, but with bandwidth estimates added to each segment.
    """

    n_fft: int = 512
    stride_seconds: float = 0.01
    top_db: float = 100.0
    frequency_threshold: float = -50.0
    audio_filepath_key: str = "audio_filepath"

    # Stage metadata
    name: str = "BandwidthEstimation"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, "segments"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, "segments", "metrics"]

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

    def process(self, task: AudioTask) -> AudioTask:
        """Estimate bandwidth for audio entry."""
        data_entry = task.data
        audio_path = data_entry.get(self.audio_filepath_key)
        if not audio_path:
            logger.error(
                f"[{self.name}] Missing '{self.audio_filepath_key}' for entry: "
                f"{data_entry.get('audio_item_id', 'unknown')}"
            )
            return task
        try:
            audio, sample_rate = librosa.load(path=audio_path, sr=None)
        except Exception as ex:  # noqa: BLE001
            logger.error(f"Failed to load audio path: {audio_path}, exception={ex}")
            return task
        segments = data_entry.get("segments", [])

        for segment in segments:
            if segment.get("speaker") == "no-speaker" or segment.get("text", "").strip() == "":
                continue

            start = segment["start"]
            end = segment["end"]

            audio_segment = audio[int(start * sample_rate) : int(end * sample_rate)]
            bandwidth = self._estimate_bandwidth(audio_segment, sample_rate)

            if "metrics" not in segment:
                segment["metrics"] = {}
            segment["metrics"]["bandwidth"] = int(bandwidth)
        return task
