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
Resample Audio Stage - Native NeMo Curator Implementation.

Resamples audio files to a target sample rate and format.
Follows the exact pattern from NeMo Curator:
https://github.com/NVIDIA-NeMo/Curator/blob/main/nemo_curator/stages/audio/common.py

"""

import os
import subprocess
from dataclasses import dataclass
from typing import Any

from fsspec.core import url_to_fs
from nemo_curator.stages.audio.common import LegacySpeechStage
from nemo_curator.tasks import AudioBatch

from nemo_curator.stages.audio.common import get_audio_duration


@dataclass
class ResampleAudioStage(LegacySpeechStage):
    """
    Stage for resampling audio files in a TTS/ALM dataset.

    Takes a manifest containing audio file paths and resamples them to
    target sample rate and format, while creating a new manifest with
    updated paths.

    """

    # Processing parameters
    input_format: str = "wav"
    resampled_audio_dir: str = "/local/resampled"
    target_sample_rate: int = 16000
    target_format: str = "wav"
    target_nchannels: int = 1

    # Stage metadata
    name: str = "ResampleAudio"

    def setup(self, worker_metadata: Any = None):
        fs, path = url_to_fs(self.resampled_audio_dir)
        self.resampled_audio_dir = path
        fs.makedirs(path, exist_ok=True)

    def process_dataset_entry(self, data_entry: dict[str, Any]) -> list[AudioBatch]:
        """
        Process a single dataset entry by resampling the audio file.

        Args:
            data_entry: Dictionary with audio_filepath and audio_item_id(optional)

        Returns:
            List containing AudioBatch with updated metadata
        """

        assert "audio_filepath" in data_entry, "Absolute audio filepath is required"

        _, audio_filepath = url_to_fs(data_entry["audio_filepath"])
        if "audio_item_id" not in data_entry:
            data_entry["audio_item_id"] = os.path.splitext(os.path.basename(audio_filepath))[0]

        input_audio_path = audio_filepath
        output_audio_path = os.path.join(
            self.resampled_audio_dir,
            data_entry["audio_item_id"] + "." + self.target_format,
        )

        # Convert audio file if not already done
        fs, output_path = url_to_fs(output_audio_path)
        if not fs.exists(output_path):
            if input_audio_path.lower().endswith(".wav"):
                cmd = f'sox --no-dither -V1 "{input_audio_path}" -r {self.target_sample_rate} -c {self.target_nchannels} -b 16 "{output_audio_path}"'
            else:
                cmd = f'ffmpeg -i "{input_audio_path}" -ar {self.target_sample_rate} -ac {self.target_nchannels} -ab 16 "{output_audio_path}" -v error'

            try:
                subprocess.run(
                    cmd, check=True, capture_output=True, text=True, shell=True
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Error converting {input_audio_path}: {e}") from e

        # Update metadata
        data_entry["audio_filepath"] = input_audio_path
        data_entry["resampled_audio_filepath"] = output_audio_path
        data_entry["duration"] = get_audio_duration(output_audio_path)

        return [AudioBatch(data=[data_entry])]
