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
Resample Audio Stage

Resamples audio files to a target sample rate and format.
Follows the exact pattern from NeMo Curator:
https://github.com/NVIDIA-NeMo/Curator/blob/main/nemo_curator/stages/audio/common.py

"""

import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass

import torch
from fsspec.core import url_to_fs

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio.common import get_audio_duration
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class ResampleAudioStage(ProcessingStage[AudioTask, AudioTask]):
    """
    Stage for resampling audio files in a TTS/ALM dataset.

    Takes a manifest containing audio file paths and resamples them to
    target sample rate and format, while creating a new manifest with
    updated paths.

    """

    # Processing parameters
    resampled_audio_dir: str
    input_format: str = "wav"
    target_sample_rate: int = 16000
    target_format: str = "wav"
    target_nchannels: int = 1
    write_resampled_audio: bool = True
    emit_waveform: bool = False

    # Key names
    audio_filepath_key: str = "audio_filepath"
    resampled_audio_filepath_key: str = "resampled_audio_filepath"
    duration_key: str = "duration"
    audio_item_id_key: str = "audio_item_id"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"

    # Stage metadata
    name: str = "ResampleAudio"

    def __post_init__(self) -> None:
        if not isinstance(self.write_resampled_audio, bool):
            msg = (
                f"ResampleAudioStage.write_resampled_audio must be a bool, "
                f"got {type(self.write_resampled_audio).__name__}"
            )
            raise TypeError(msg)
        if not isinstance(self.emit_waveform, bool):
            msg = f"ResampleAudioStage.emit_waveform must be a bool, got {type(self.emit_waveform).__name__}"
            raise TypeError(msg)
        if not self.write_resampled_audio and not self.emit_waveform:
            msg = "ResampleAudioStage must either write_resampled_audio or emit_waveform"
            raise ValueError(msg)
        if self.target_sample_rate <= 0:
            msg = f"target_sample_rate must be > 0, got {self.target_sample_rate}"
            raise ValueError(msg)
        if self.target_nchannels <= 0:
            msg = f"target_nchannels must be > 0, got {self.target_nchannels}"
            raise ValueError(msg)

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        if not shutil.which("ffmpeg"):
            msg = "ResampleAudioStage requires 'ffmpeg'. Install with: sudo apt-get install -y ffmpeg"
            raise RuntimeError(msg)
        if self.write_resampled_audio:
            fs, path = url_to_fs(self.resampled_audio_dir)
            fs.makedirs(path, exist_ok=True)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        outputs = [
            self.audio_filepath_key,
            self.audio_item_id_key,
            self.duration_key,
        ]
        if self.write_resampled_audio:
            outputs.append(self.resampled_audio_filepath_key)
        if self.emit_waveform:
            outputs.extend([self.waveform_key, self.sample_rate_key, "is_mono", "num_samples"])
        return [], outputs

    def process(self, task: AudioTask) -> AudioTask:
        """
        Process a single task by resampling the audio file.

        Args:
            task: AudioTask with data dict containing audio_filepath and audio_item_id(optional)

        Returns:
            AudioTask with updated metadata
        """
        t0 = time.perf_counter()
        data_entry = task.data

        if self.audio_filepath_key not in data_entry:
            msg = "Absolute audio filepath is required"
            raise ValueError(msg)

        original_audio_filepath = data_entry[self.audio_filepath_key]
        _, local_audio_path = url_to_fs(original_audio_filepath)
        if self.audio_item_id_key not in data_entry:
            stem = os.path.splitext(os.path.basename(local_audio_path))[0]
            path_hash = hashlib.sha256(local_audio_path.encode()).hexdigest()[:8]
            data_entry[self.audio_item_id_key] = f"{stem}_{path_hash}"

        input_audio_path = local_audio_path
        output_audio_path = os.path.join(
            self.resampled_audio_dir,
            data_entry[self.audio_item_id_key] + "." + self.target_format,
        )

        duration = -1.0
        skipped_conversion = False
        waveform_bytes = 0.0

        if self.write_resampled_audio:
            fs, output_path = url_to_fs(output_audio_path)
            skipped_conversion = fs.exists(output_path)
            if not skipped_conversion:
                self._write_resampled_audio_file(input_audio_path, output_audio_path)
            data_entry[self.resampled_audio_filepath_key] = output_audio_path
            duration = get_audio_duration(output_audio_path)

        if self.emit_waveform:
            waveform, sample_rate = self._load_resampled_waveform(input_audio_path)
            num_samples = int(waveform.shape[-1])
            data_entry[self.waveform_key] = waveform
            data_entry[self.sample_rate_key] = sample_rate
            data_entry["is_mono"] = waveform.shape[0] == 1
            data_entry["num_samples"] = num_samples
            duration = num_samples / sample_rate
            waveform_bytes = float(waveform.element_size() * waveform.nelement())

        # Update metadata — preserve original URL for cloud paths
        data_entry[self.audio_filepath_key] = original_audio_filepath
        data_entry[self.duration_key] = duration

        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "duration": max(duration, 0.0),
                "skipped_conversion": float(skipped_conversion),
                "write_resampled_audio": float(self.write_resampled_audio),
                "emit_waveform": float(self.emit_waveform),
                "waveform_bytes": waveform_bytes,
            }
        )
        return task

    def _write_resampled_audio_file(self, input_audio_path: str, output_audio_path: str) -> None:
        """Run ffmpeg and write a normalized audio file to disk/object storage path."""
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            input_audio_path,
            "-ar",
            str(self.target_sample_rate),
            "-ac",
            str(self.target_nchannels),
            "-acodec",
            "pcm_s16le",
            output_audio_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            msg = f"Error converting {input_audio_path}: {e.stderr or e}"
            raise RuntimeError(msg) from e

    def _load_resampled_waveform(self, input_audio_path: str) -> tuple[torch.Tensor, int]:
        """Decode/resample audio with ffmpeg and return a 2-D float32 waveform."""
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            input_audio_path,
            "-ar",
            str(self.target_sample_rate),
            "-ac",
            str(self.target_nchannels),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ]
        try:
            completed = subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else e.stderr
            msg = f"Error loading resampled waveform from {input_audio_path}: {stderr or e}"
            raise RuntimeError(msg) from e

        if not completed.stdout:
            msg = f"ffmpeg produced an empty waveform for {input_audio_path}"
            raise RuntimeError(msg)

        samples = torch.frombuffer(bytearray(completed.stdout), dtype=torch.float32).clone()
        channels = int(self.target_nchannels)
        if samples.numel() % channels != 0:
            msg = (
                f"Decoded sample count {samples.numel()} is not divisible by "
                f"target_nchannels={channels} for {input_audio_path}"
            )
            raise RuntimeError(msg)
        if channels == 1:
            return samples.unsqueeze(0), self.target_sample_rate
        return samples.reshape(-1, channels).T.contiguous(), self.target_sample_rate
