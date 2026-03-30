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

"""
Audio mono conversion stage.

Converts multi-channel audio to mono and verifies sample rate.
Typically the first stage in an audio processing pipeline.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.preprocessing import MonoConversionStage

    pipeline = Pipeline(name="audio_pipeline")
    pipeline.add_stage(MonoConversionStage(output_sample_rate=48000))
"""

import os
from dataclasses import dataclass, field

import torch
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

from nemo_curator.stages.audio.common import load_audio_file


@dataclass
class MonoConversionStage(ProcessingStage[AudioTask, AudioTask]):
    """
    Audio mono conversion and sample rate verification stage.

    Converts multi-channel audio to mono by averaging channels.
    Optionally verifies that audio matches expected sample rate.

    Args:
        output_sample_rate: Expected sample rate in Hz (default: 48000)
        audio_filepath_key: Key in data dict for audio file path
        strict_sample_rate: If True, reject audio with wrong sample rate
    """

    output_sample_rate: int = 48000
    audio_filepath_key: str = "audio_filepath"
    strict_sample_rate: bool = True

    name: str = "MonoConversion"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self):
        super().__init__()

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["waveform", "sample_rate", "is_mono", "duration", "num_samples"]

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        """
        Convert audio to mono and verify sample rate.

        Mutates task.data in-place with waveform data.
        Returns task if successful, None if doesn't meet requirements.
        """
        audio_filepath = task.data.get(self.audio_filepath_key)

        if not audio_filepath or not os.path.exists(audio_filepath):
            logger.error(f"Audio file not found: {audio_filepath}")
            return []

        try:
            waveform, sample_rate = load_audio_file(audio_filepath, mono=False)
            num_channels = waveform.shape[0]

            if self.strict_sample_rate and sample_rate != self.output_sample_rate:
                logger.warning(
                    f"Sample rate {sample_rate}Hz != expected {self.output_sample_rate}Hz: {audio_filepath}"
                )
                return []

            if num_channels > 1:
                mono_waveform = torch.mean(waveform, dim=0, keepdim=True)
                logger.debug(f"Converted {num_channels} channels to mono")
            else:
                mono_waveform = waveform

            task.data["waveform"] = mono_waveform
            task.data["sample_rate"] = sample_rate
            task.data["is_mono"] = True
            task.data["duration"] = mono_waveform.shape[1] / sample_rate
            task.data["num_samples"] = mono_waveform.shape[1]

            return task

        except (OSError, RuntimeError) as e:
            logger.error(f"Error processing {audio_filepath}: {e}")
            return []
