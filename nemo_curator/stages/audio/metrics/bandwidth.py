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

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, ConditionalWrite, Gates, IOSpec, StageContract
from nemo_curator.stages.audio._agent._residency import InputResidency, residency_read_specs
from nemo_curator.stages.audio.common import ensure_mono, ensure_waveform_2d
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class BandwidthEstimationStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
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
        waveform_key: Key for an in-memory waveform tensor. Defaults to "waveform".
        sample_rate_key: Key for the in-memory waveform sample rate. Defaults to "sample_rate".
        input_residency: Which input to use — "file" (audio_filepath only; default, unchanged),
            "waveform" (in-memory only), or "auto" (waveform first, file fallback).

    Returns:
        The same data as in the input data, but with bandwidth estimates added to each segment.
    """

    n_fft: int = 512
    stride_seconds: float = 0.01
    top_db: float = 100.0
    frequency_threshold: float = -50.0
    audio_filepath_key: str = "audio_filepath"
    segments_key: str = "segments"
    duration_key: str = "duration"
    metrics_key: str = "metrics"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    input_residency: InputResidency = "file"

    # Stage metadata
    name: str = "BandwidthEstimation"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, self.metrics_key]

    def describe(self) -> StageContract:
        # An audio source (file or in-memory waveform, per input_residency) AND
        # (segments OR duration). Each audio-source shape is paired with both refinements.
        reads_one_of = []
        for spec in residency_read_specs(
            self.input_residency,
            audio_filepath_key=self.audio_filepath_key,
            waveform_key=self.waveform_key,
            sample_rate_key=self.sample_rate_key,
        ):
            reads_one_of.append(IOSpec(data_keys=[*spec.data_keys, self.segments_key], accepts=list(spec.accepts)))
            reads_one_of.append(IOSpec(data_keys=[*spec.data_keys, self.duration_key], accepts=list(spec.accepts)))
        return StageContract(
            reads_one_of=reads_one_of,
            writes=IOSpec(data_keys=[self.metrics_key], segment_data_keys=[self.metrics_key]),
            conditional_writes=[
                ConditionalWrite(
                    writes=IOSpec(data_keys=[self.metrics_key]),
                    condition=(
                        f"audio resolves, '{self.segments_key}' is absent, the top-level item is not skipped "
                        "for speaker/text, its time range is valid, and bandwidth estimation completes"
                    ),
                    value_origin="augments_upstream_same_key",
                ),
                ConditionalWrite(
                    writes=IOSpec(segment_data_keys=[self.metrics_key]),
                    condition=(
                        f"audio resolves, '{self.segments_key}' is present, and an individual segment is not "
                        "skipped for speaker/text, has a valid range, and bandwidth estimation completes"
                    ),
                    value_origin="augments_upstream_same_key",
                ),
                ConditionalWrite(
                    writes=IOSpec(segment_data_keys=[self.metrics_key]),
                    condition=(
                        f"'{self.segments_key}' is present, an individual segment raises a caught ValueError, "
                        f"and '{self.metrics_key}.metric_skip_reason' is assigned"
                    ),
                    value_origin="augments_upstream_same_key",
                ),
            ],
            # The threshold is measured against the peak of this clip's own power spectrum, not
            # against a level taken over the corpus.
            gates=Gates(per_row_independent=True),
        )

    def validate_input(self, task: AudioTask) -> bool:
        """Needs an audio source AND (segments OR duration).

        The audio source is ``audio_filepath_key`` (default) or, when ``input_residency``
        allows it, an in-memory ``waveform_key``+``sample_rate_key``.
        """
        data = task.data
        has_waveform = data.get(self.waveform_key) is not None and data.get(self.sample_rate_key) is not None
        has_file = self.audio_filepath_key in data
        if self.input_residency == "waveform":
            has_audio = has_waveform
        elif self.input_residency == "file":
            has_audio = has_file
        else:  # auto
            has_audio = has_waveform or has_file
        if not has_audio:
            logger.error(
                f"Task {task.task_id} missing audio input for input_residency={self.input_residency!r}: "
                f"need '{self.audio_filepath_key}' or '{self.waveform_key}'+'{self.sample_rate_key}'"
            )
            return False
        if self.segments_key in data or self.duration_key in data:
            return True
        logger.error(
            f"Task {task.task_id} missing required attributes: need '{self.segments_key}' OR '{self.duration_key}'"
        )
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

        if self.metrics_key not in audio_segment:
            audio_segment[self.metrics_key] = {}

        audio_segment[self.metrics_key]["bandwidth"] = int(bandwidth)

    def _resolve_entry_audio(self, data_entry: dict[str, Any]) -> tuple["np.ndarray", int]:
        """Return ``(mono_1d_audio, sample_rate)`` from a waveform or the file.

        When ``input_residency`` allows it and an in-memory waveform is present, it is used
        directly; otherwise the audio file is loaded at its native rate (default, unchanged).
        """
        if self.input_residency != "file":
            waveform = data_entry.get(self.waveform_key)
            sr = data_entry.get(self.sample_rate_key)
            if waveform is not None and sr is not None:
                audio = ensure_mono(ensure_waveform_2d(waveform)).squeeze(0)
                return audio.detach().cpu().numpy(), int(sr)
            if self.input_residency == "waveform":
                msg = (
                    f"[{self.name}] Missing '{self.waveform_key}'+'{self.sample_rate_key}' for entry: "
                    f"{data_entry.get('audio_item_id', 'unknown')} (input_residency='waveform')"
                )
                raise ValueError(msg)
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
        return audio, sample_rate

    def process(self, task: AudioTask) -> AudioTask:
        """Estimate bandwidth for audio entry."""
        data_entry = task.data
        audio, sample_rate = self._resolve_entry_audio(data_entry)

        if self.segments_key in data_entry:
            for segment in data_entry[self.segments_key]:
                try:
                    self.get_bandwidth(segment, audio, sample_rate)
                except ValueError as ex:
                    logger.warning(f"[{self.name}] skipping segment in {task.task_id}: {ex}")
                    segment.setdefault(self.metrics_key, {})["metric_skip_reason"] = str(ex)
        else:
            self.get_bandwidth(data_entry, audio, sample_rate)

        return task
