# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
NeMo Curator Audio Processing Stages.

This module provides stages for audio curation pipelines including:

Preprocessing:
    - MonoConversionStage: Convert multi-channel audio to mono
    - SegmentConcatenationStage: Concatenate audio segments

Segmentation:
    - VADSegmentationStage: Voice Activity Detection segmentation
    - SpeakerSeparationStage: Speaker diarization and separation

Filtering:
    - NISQAFilterStage: NISQA speech quality filtering
    - SIGMOSFilterStage: SIGMOS quality filtering
    - BandFilterStage: Bandwidth classification filtering

Advanced Pipelines:
    - AudioDataFilterStage: Complete audio curation pipeline (VAD + Quality + Speaker Sep)
    - AudioDataFilterConfig: Configuration for AudioDataFilterStage

Common:
    - GetAudioDurationStage: Extract audio duration
    - PreserveByValueStage: Filter by field value

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio import (
        MonoConversionStage,
        VADSegmentationStage,
        NISQAFilterStage,
    )
    
    pipeline = Pipeline(name="audio_curation")
    pipeline.add_stage(MonoConversionStage(output_sample_rate=48000))
    pipeline.add_stage(VADSegmentationStage(min_duration_sec=2.0))
    pipeline.add_stage(NISQAFilterStage(mos_threshold=4.5))
    
    # Or use the unified AudioDataFilterStage:
    from nemo_curator.stages.audio import AudioDataFilterStage, AudioDataFilterConfig
    
    config = AudioDataFilterConfig(enable_nisqa=True, enable_speaker_separation=True)
    stage = AudioDataFilterStage(config=config)
"""

# Preprocessing stages
from nemo_curator.stages.audio.preprocessing import (
    MonoConversionStage,
    SegmentConcatenationStage,
)

# Segmentation stages
from nemo_curator.stages.audio.segmentation import (
    VADSegmentationStage,
    SpeakerSeparationStage,
)

# Filtering stages
from nemo_curator.stages.audio.filtering import (
    NISQAFilterStage,
    SIGMOSFilterStage,
    BandFilterStage,
)

# Common stages
from nemo_curator.stages.audio.common import (
    GetAudioDurationStage,
    PreserveByValueStage,
)

# Advanced Pipelines (Composite stages)
from nemo_curator.stages.audio.advance_pipelines import (
    AudioDataFilterStage,
    AudioDataFilterConfig,
)

# Configurations
from nemo_curator.stages.audio.configs import (
    VADConfig,
    NISQAConfig,
    SIGMOSConfig,
    BandFilterConfig,
    SpeakerSeparationConfig,
    MonoConversionConfig,
    SegmentConcatenationConfig,
)

__all__ = [
    # Preprocessing
    "MonoConversionStage",
    "SegmentConcatenationStage",
    # Segmentation
    "VADSegmentationStage",
    "SpeakerSeparationStage",
    # Filtering
    "NISQAFilterStage",
    "SIGMOSFilterStage",
    "BandFilterStage",
    # Advanced Pipelines
    "AudioDataFilterStage",
    "AudioDataFilterConfig",
    # Common
    "GetAudioDurationStage",
    "PreserveByValueStage",
    # Configs
    "VADConfig",
    "NISQAConfig",
    "SIGMOSConfig",
    "BandFilterConfig",
    "SpeakerSeparationConfig",
    "MonoConversionConfig",
    "SegmentConcatenationConfig",
]

