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

    # Key names
    audio_filepath_key: str = "audio_filepath"
    resampled_audio_filepath_key: str = "resampled_audio_filepath"
    duration_key: str = "duration"
    audio_item_id_key: str = "audio_item_id"

    # Stage metadata
    name: str = "ResampleAudio"

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        if not shutil.which("sox") and not shutil.which("ffmpeg"):
            msg = "ResampleAudioStage requires 'sox' or 'ffmpeg'. Install with: sudo apt-get install -y sox ffmpeg"
            raise RuntimeError(msg)
        fs, path = url_to_fs(self.resampled_audio_dir)
        fs.makedirs(path, exist_ok=True)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            self.audio_filepath_key,
            self.audio_item_id_key,
            self.resampled_audio_filepath_key,
            self.duration_key,
        ]

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

        # Convert audio file if not already done
        fs, output_path = url_to_fs(output_audio_path)
        skipped_conversion = fs.exists(output_path)
        if not skipped_conversion:
            if input_audio_path.lower().endswith(".wav"):
                cmd = [
                    "sox",
                    "--no-dither",
                    "-V1",
                    input_audio_path,
                    "-r",
                    str(self.target_sample_rate),
                    "-c",
                    str(self.target_nchannels),
                    "-b",
                    "16",
                    output_audio_path,
                ]
            else:
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
                msg = f"Error converting {input_audio_path}: {e}"
                raise RuntimeError(msg) from e

        # Update metadata — preserve original URL for cloud paths
        data_entry[self.audio_filepath_key] = original_audio_filepath
        data_entry[self.resampled_audio_filepath_key] = output_audio_path
        duration = get_audio_duration(output_audio_path)
        data_entry[self.duration_key] = duration

        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "duration": max(duration, 0.0),
                "skipped_conversion": float(skipped_conversion),
            }
        )
        return task
