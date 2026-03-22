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
Advanced Audio Processing Pipelines.

This module provides composite pipeline stages that combine multiple
audio processing steps into single, easy-to-use stages.

Stages:
    - AudioDataFilterStage: Complete audio curation pipeline with VAD,
      quality filtering, speaker separation, and timestamp tracking.

Example:
    from nemo_curator.stages.audio.advance_pipelines import (
        AudioDataFilterStage,
        AudioDataFilterConfig,
    )
    
    # Create config
    config = AudioDataFilterConfig(
        enable_utmos=True,
        enable_sigmos=True,
        enable_speaker_separation=True,
        utmos_mos_threshold=3.5,
    )
    
    # Add to pipeline (CompositeStage decomposes into independent stages)
    pipeline.add_stage(AudioDataFilterStage(config=config))
"""

from .audio_data_filter import AudioDataFilterStage
from .config import AudioDataFilterConfig

__all__ = [
    "AudioDataFilterStage",
    "AudioDataFilterConfig",
]
