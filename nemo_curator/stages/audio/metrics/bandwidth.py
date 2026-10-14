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
    """Estimate the highest frequency bin near the signal's spectral peak."""
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

    peak = float(np.max(power_spec))
    freq_width = sample_rate / n_fft
    for idx in range(len(power_spec) - 1, -1, -1):
        if power_spec[idx] - peak > frequency_threshold:
            return int(idx * freq_width)
    return 0


def empty_annotation(error: str) -> dict[str, Any]:
    return {"bandwidth": None, "bandwidth_sample_rate": None, "error": error}


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
    output_key: str | None = None
    tar_root: str | None = None
    strip_tar_key_prefixes: tuple[str, ...] = ()
    audio_source: str = "original"
    enhanced_key: str = "enhanced_audio"
    enhanced_fallback: bool = False

    # Stage metadata
    name: str = "BandwidthEstimation"

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
        if self.output_key is not None:
            return [], []
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        if self.output_key is not None:
            return [], [self.output_key.split(".")[0]]
        return [], [self.audio_filepath_key, "metrics"]

    def validate_input(self, task: AudioTask) -> bool:
        """OR-shaped: needs audio_filepath AND (segments OR duration)."""
        data = task.data
        if self.output_key is not None:
            return True
        if not data.get(self.audio_filepath_key):
            logger.error(f"Task {task.task_id} missing '{self.audio_filepath_key}'")
            return False
        if self.segments_key in data or "duration" in data:
            return True
        logger.error(f"Task {task.task_id} missing required attributes: need '{self.segments_key}' OR 'duration'")
        return False

    def _estimate_bandwidth(self, audio: "np.ndarray", sample_rate: int) -> int:
        """Estimate the bandwidth of an audio signal."""
        return estimate_bandwidth(
            audio,
            sample_rate,
            self.n_fft,
            stride_seconds=self.stride_seconds,
            top_db=self.top_db,
            frequency_threshold=self.frequency_threshold,
        )

    def _load_annotation_audio(self, item: dict[str, Any], task_id: str) -> tuple[np.ndarray, int] | str:
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

    def _annotation(self, item: dict[str, Any], task_id: str) -> dict[str, Any]:
        audio = self._load_annotation_audio(item, task_id)
        if isinstance(audio, str):
            return empty_annotation(audio)
        samples, sample_rate = audio
        try:
            return {
                "bandwidth": self._estimate_bandwidth(samples, sample_rate),
                "bandwidth_sample_rate": sample_rate,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Bandwidth estimation failed for {}: {}", task_id, exc)
            return empty_annotation("bandwidth_failed")

    def _annotate_task(self, task: AudioTask) -> None:
        if self.output_key is None:
            msg = "output_key is required for annotation mode"
            raise RuntimeError(msg)
        segments = task.data.get(self.segments_key)
        if isinstance(segments, list):
            for index, segment in enumerate(segments):
                if isinstance(segment, dict):
                    set_dotted(segment, self.output_key, self._annotation(segment, f"{task.task_id}:{index}"))
            return
        set_dotted(task.data, self.output_key, self._annotation(task.data, task.task_id))

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
        if self.output_key is not None:
            self._annotate_task(task)
            return task
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
