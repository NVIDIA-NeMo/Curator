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

import importlib as _importlib

__all__ = [
    "ALMDataBuilderStage",
    "ALMDataOverlapStage",
    "AudioDataFilterStage",
    "BandFilterStage",
    "GetAudioDurationStage",
    "MonoConversionStage",
    "PreserveByValueStage",
    "SIGMOSFilterStage",
    "SegmentConcatenationStage",
    "SpeakerSeparationStage",
    "TimestampMapperStage",
    "UTMOSFilterStage",
    "VADSegmentationStage",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ALMDataBuilderStage": ("nemo_curator.stages.audio.alm", "ALMDataBuilderStage"),
    "ALMDataOverlapStage": ("nemo_curator.stages.audio.alm", "ALMDataOverlapStage"),
    "AudioDataFilterStage": ("nemo_curator.stages.audio.advanced_pipelines", "AudioDataFilterStage"),
    "BandFilterStage": ("nemo_curator.stages.audio.filtering", "BandFilterStage"),
    "SIGMOSFilterStage": ("nemo_curator.stages.audio.filtering", "SIGMOSFilterStage"),
    "UTMOSFilterStage": ("nemo_curator.stages.audio.filtering", "UTMOSFilterStage"),
    "GetAudioDurationStage": ("nemo_curator.stages.audio.common", "GetAudioDurationStage"),
    "PreserveByValueStage": ("nemo_curator.stages.audio.common", "PreserveByValueStage"),
    "MonoConversionStage": ("nemo_curator.stages.audio.preprocessing", "MonoConversionStage"),
    "SegmentConcatenationStage": ("nemo_curator.stages.audio.preprocessing", "SegmentConcatenationStage"),
    "SpeakerSeparationStage": ("nemo_curator.stages.audio.segmentation", "SpeakerSeparationStage"),
    "VADSegmentationStage": ("nemo_curator.stages.audio.segmentation", "VADSegmentationStage"),
    "TimestampMapperStage": ("nemo_curator.stages.audio.postprocessing", "TimestampMapperStage"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
