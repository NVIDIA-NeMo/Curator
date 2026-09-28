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

"""Annotate audio bandwidth (TTS Granary OrigAudioPipeline.Bandwidth)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from nemo_curator.stages.audio.common import resolve_waveform_from_item
from nemo_curator.stages.audio.tts.fields import set_dotted
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_MIN_N_FFT = 2


def estimate_bandwidth(  # noqa: PLR0913
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    *,
    stride_seconds: float = 0.01,
    top_db: float = 100.0,
    frequency_threshold: float = -50.0,
) -> int:
    """Estimate the highest frequency bin near the signal's spectral peak.

    Port of TTS Granary ``pipeline.stages.bandwidth.processor.estimate_bandwidth``,
    using a numpy STFT instead of librosa.
    """
    hop_length = int(sample_rate * stride_seconds)
    if hop_length < 1:
        msg = f"stride_seconds={stride_seconds} produces invalid hop_length={hop_length} for sample_rate={sample_rate}"
        raise ValueError(msg)
    if n_fft < _MIN_N_FFT:
        msg = "n_fft must be at least 2"
        raise ValueError(msg)

    window = np.blackman(n_fft).astype(np.float64)
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if audio.size == 0:
        return 0

    n_frames = 1 + max(0, (audio.size - n_fft) // hop_length)
    frames: list[np.ndarray] = []
    for i in range(max(n_frames, 1)):
        start = i * hop_length
        frame = audio[start : start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size))
        frames.append(np.abs(np.fft.rfft(frame * window)) ** 2)

    power_spec = np.mean(np.stack(frames, axis=0), axis=0)
    ref = float(n_fft)
    log_spec = 10.0 * np.log10(np.maximum(power_spec, 1e-20) / ref)
    log_spec = np.maximum(log_spec, float(np.max(log_spec)) - top_db)

    bandwidth = 0.0
    peak = float(np.max(log_spec))
    freq_width = sample_rate / n_fft
    for idx in range(len(log_spec) - 1, -1, -1):
        if log_spec[idx] - peak > frequency_threshold:
            bandwidth = idx * freq_width
            break
    return int(bandwidth)


def empty_annotation(error: str) -> dict[str, Any]:
    return {"bandwidth": None, "bandwidth_sample_rate": None, "error": error}


@dataclass
class BandwidthAnnotationStage(ProcessingStage[AudioTask, AudioTask]):
    """Estimate original-audio bandwidth and write the annotation without dropping rows.

    Port of TTS Granary ``BandwidthProcessor``. Uses in-memory ``waveform`` /
    ``audio_filepath`` via ``resolve_waveform_from_item`` (Curator JSONL) rather
    than tar-member offsets.
    """

    output_key: str = "bandwidth"
    n_fft: int = 512
    stride_seconds: float = 0.01
    top_db: float = 100.0
    frequency_threshold: float = -50.0
    name: str = "BandwidthAnnotation"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 16

    def __post_init__(self) -> None:
        super().__init__()
        if self.n_fft < _MIN_N_FFT:
            msg = "n_fft must be at least 2"
            raise ValueError(msg)
        if self.stride_seconds <= 0:
            msg = "stride_seconds must be positive"
            raise ValueError(msg)
        if self.top_db <= 0:
            msg = "top_db must be positive"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key.split(".")[0]]

    def annotate_item(self, item: dict[str, Any], task_id: str) -> dict[str, Any]:
        audio = resolve_waveform_from_item(item, task_id)
        if audio is None:
            return empty_annotation("missing_audio")
        waveform, sample_rate = audio
        try:
            samples = waveform.detach().cpu().numpy().reshape(-1)
            bandwidth = estimate_bandwidth(
                samples,
                int(sample_rate),
                self.n_fft,
                stride_seconds=self.stride_seconds,
                top_db=self.top_db,
                frequency_threshold=self.frequency_threshold,
            )
            return {
                "bandwidth": bandwidth,
                "bandwidth_sample_rate": int(sample_rate),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Bandwidth estimation failed for {}: {}", task_id, exc)
            return empty_annotation("bandwidth_failed")

    def _annotate_task(self, task: AudioTask) -> None:
        segments = task.data.get("segments")
        if isinstance(segments, list):
            for i, segment in enumerate(segments):
                if isinstance(segment, dict):
                    set_dotted(segment, self.output_key, self.annotate_item(segment, f"{task.task_id}:{i}"))
            return
        set_dotted(task.data, self.output_key, self.annotate_item(task.data, task.task_id))

    def process(self, task: AudioTask) -> AudioTask:
        self._annotate_task(task)
        return task
