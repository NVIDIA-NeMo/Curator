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
from nemo_curator.stages.audio.tts.audio_io import (
    TAR_KEY_FIELD,
    TAR_MEMBER_FIELD,
    SegmentAudioError,
    load_segment_audio,
    make_audio_config,
)
from nemo_curator.stages.audio.tts.fields import get_dotted, set_dotted
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

    Exact port of TTS Granary ``pipeline.stages.bandwidth.processor.estimate_bandwidth`` (librosa STFT with a
    periodic Blackman-Harris window, mean power over time, ``power_to_db(ref=n_fft)``).
    """
    import librosa

    hop_length = int(sample_rate * stride_seconds)
    if hop_length < 1:
        msg = f"stride_seconds={stride_seconds} produces invalid hop_length={hop_length} for sample_rate={sample_rate}"
        raise ValueError(msg)
    if n_fft < _MIN_N_FFT:
        msg = "n_fft must be at least 2"
        raise ValueError(msg)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 0

    spec = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, window="blackmanharris")
    power_spec = np.mean(np.abs(spec) ** 2, axis=1)
    power_spec = librosa.power_to_db(power_spec, ref=n_fft, top_db=top_db)

    bandwidth = 0.0
    peak = float(np.max(power_spec))
    freq_width = sample_rate / n_fft
    for idx in range(len(power_spec) - 1, -1, -1):
        if power_spec[idx] - peak > frequency_threshold:
            bandwidth = idx * freq_width
            break
    return int(bandwidth)


def empty_annotation(error: str) -> dict[str, Any]:
    return {"bandwidth": None, "bandwidth_sample_rate": None, "error": error}


@dataclass
class BandwidthAnnotationStage(ProcessingStage[AudioTask, AudioTask]):
    """Estimate original-audio bandwidth and write the annotation without dropping rows.

    Port of TTS Granary ``BandwidthProcessor``. Audio comes from an in-memory ``waveform`` / ``audio_filepath``
    (``resolve_waveform_from_item``) or, for Granary segments, the tar member addressed by
    ``GranaryHifi.AudioMetadata`` at its native sample rate (``tar_root`` / ``strip_tar_key_prefixes`` resolve
    relative ``tar_key`` values).
    """

    output_key: str = "bandwidth"
    n_fft: int = 512
    stride_seconds: float = 0.01
    top_db: float = 100.0
    frequency_threshold: float = -50.0
    tar_root: str | None = None
    strip_tar_key_prefixes: tuple[str, ...] = ()
    audio_source: str = "original"
    enhanced_key: str = "enhanced_audio"
    enhanced_fallback: bool = False
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

    def _load_audio(self, item: dict[str, Any], task_id: str) -> tuple[np.ndarray, int] | str:
        if get_dotted(item, TAR_MEMBER_FIELD) or get_dotted(item, TAR_KEY_FIELD):
            try:
                return load_segment_audio(item, None, make_audio_config(self))
            except SegmentAudioError as exc:
                return exc.code
        audio = resolve_waveform_from_item(item, task_id)
        if audio is None:
            return "missing_audio"
        waveform, sample_rate = audio
        return waveform.detach().cpu().numpy().reshape(-1), int(sample_rate)

    def annotate_item(self, item: dict[str, Any], task_id: str) -> dict[str, Any]:
        audio = self._load_audio(item, task_id)
        if isinstance(audio, str):
            return empty_annotation(audio)
        samples, sample_rate = audio
        try:
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
