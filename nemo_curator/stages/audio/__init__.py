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
Audio curation stages for NeMo Curator.

This module provides stages for processing and curating audio data,
including ASR inference, quality assessment, ALM data preparation,
audio preprocessing (mono conversion, segment concatenation, timestamp mapping),
audio quality filtering (SIGMOS, UTMOS, bandwidth classification filtering),
VAD segmentation, speaker diarization/separation,
and advanced audio processing pipelines.
"""

from nemo_curator.stages.audio.advanced_pipelines import AudioDataFilterStage
from nemo_curator.stages.audio.alm import ALMDataBuilderStage, ALMDataOverlapStage
from nemo_curator.stages.audio.common import (
    GetAudioDurationStage,
    ManifestReader,
    ManifestWriterStage,
    PreserveByValueStage,
)
from nemo_curator.stages.audio.filtering import (
    BandFilterStage,
    SIGMOSFilterStage,
    UTMOSFilterStage,
)
from nemo_curator.stages.audio.postprocessing import (
    TimestampMapperStage,
)
from nemo_curator.stages.audio.preprocessing import (
    MonoConversionStage,
    SegmentConcatenationStage,
)
from nemo_curator.stages.audio.segmentation import (
    SpeakerSeparationStage,
    VADSegmentationStage,
)

__all__ = [
    "ALMDataBuilderStage",
    "ALMDataOverlapStage",
    "AudioDataFilterStage",
    "BandFilterStage",
    "GetAudioDurationStage",
    "ManifestReader",
    "ManifestWriterStage",
    "MonoConversionStage",
    "PreserveByValueStage",
    "SIGMOSFilterStage",
    "SegmentConcatenationStage",
    "SpeakerSeparationStage",
    "TimestampMapperStage",
    "UTMOSFilterStage",
    "VADSegmentationStage",
]
