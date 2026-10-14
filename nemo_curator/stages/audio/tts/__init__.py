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

"""GranaryHiFi TTS data stages; see ``tutorials/audio/granary_hifi/README.md``."""

from importlib import import_module

_LAZY = {
    "BandwidthAnnotationStage": "nemo_curator.stages.audio.tts.bandwidth",
    "CodecCodesStage": "nemo_curator.stages.audio.tts.codec",
    "ContextSelectionStage": "nemo_curator.stages.audio.tts.context",
    "EarlyCutOffEndpointGateStage": "nemo_curator.stages.audio.tts.early_cutoff.endpoint_gate",
    "EarlyCutOffMFAResolverStage": "nemo_curator.stages.audio.tts.early_cutoff.mfa_resolver",
    "LhotseSharBuildStage": "nemo_curator.stages.audio.tts.shar",
    "ManifestIpaStage": "nemo_curator.stages.audio.tts.ipa",
    "SegmentDiarizationStage": "nemo_curator.stages.audio.tts.diarization",
    "SegmentFilterStage": "nemo_curator.stages.audio.tts.segment_filter",
    "SegmentSEDStage": "nemo_curator.stages.audio.tts.sed",
    "SharValidationStage": "nemo_curator.stages.audio.tts.shar_validation",
    "SpeakerClusteringStage": "nemo_curator.stages.audio.tts.speaker_clustering",
    "ResembleEnhanceStage": "nemo_curator.stages.audio.tts.enhance_reuse",
    "StudioVoiceEnhancementStage": "nemo_curator.stages.audio.tts.studio_voice",
    "TrainingFlagsStage": "nemo_curator.stages.audio.tts.training_flags",
    "UTMOSv2AnnotationStage": "nemo_curator.stages.audio.tts.utmos",
    "publish_shars": "nemo_curator.stages.audio.tts.shar_validation",
}

__all__ = [
    "BandwidthAnnotationStage",
    "CodecCodesStage",
    "ContextSelectionStage",
    "EarlyCutOffEndpointGateStage",
    "EarlyCutOffMFAResolverStage",
    "LhotseSharBuildStage",
    "ManifestIpaStage",
    "ResembleEnhanceStage",
    "SegmentDiarizationStage",
    "SegmentFilterStage",
    "SegmentSEDStage",
    "SharValidationStage",
    "SpeakerClusteringStage",
    "StudioVoiceEnhancementStage",
    "TrainingFlagsStage",
    "UTMOSv2AnnotationStage",
    "publish_shars",
]


def __getattr__(name: str) -> object:
    target = _LAZY.get(name)
    if target is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(import_module(target), name)


def __dir__() -> list[str]:
    return __all__
