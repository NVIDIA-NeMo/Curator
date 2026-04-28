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
text filtering (hallucination detection, LID, PnC, ITN),
and advanced audio processing pipelines.
"""

import importlib as _importlib

__all__ = [
    "ALMDataBuilderStage",
    "ALMDataOverlapStage",
    "ALMManifestReader",
    "ALMManifestReaderStage",
    "ALMManifestWriterStage",
    "AbbreviationConcatStage",
    "AudioDataFilterStage",
    "BandFilterStage",
    "DisfluencyWerGuardStage",
    "FastTextLIDStage",
    "FinalizeFieldsStage",
    "GetAudioDurationStage",
    "ITNRestorationStage",
    "InferenceQwenASRStage",
    "InferenceQwenOmniStage",
    "InitializeFieldsStage",
    "ManifestReader",
    "ManifestReaderStage",
    "ManifestWriterStage",
    "MonoConversionStage",
    "NemoTarredAudioReader",
    "PnCContentGuardStage",
    "PnCRestorationStage",
    "PreserveByValueStage",
    "RegexSubstitutionStage",
    "SIGMOSFilterStage",
    "SegmentConcatenationStage",
    "SegmentExtractionStage",
    "SelectBestPredictionStage",
    "ShardedManifestWriterStage",
    "SpeakerSeparationStage",
    "TimestampMapperStage",
    "UTMOSFilterStage",
    "VADSegmentationStage",
    "WhisperHallucinationStage",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # ALM
    "ALMDataBuilderStage": ("nemo_curator.stages.audio.alm", "ALMDataBuilderStage"),
    "ALMDataOverlapStage": ("nemo_curator.stages.audio.alm", "ALMDataOverlapStage"),
    # Advanced pipelines
    "AudioDataFilterStage": ("nemo_curator.stages.audio.advanced_pipelines", "AudioDataFilterStage"),
    # Common
    "GetAudioDurationStage": ("nemo_curator.stages.audio.common", "GetAudioDurationStage"),
    "ManifestReader": ("nemo_curator.stages.audio.common", "ManifestReader"),
    "ManifestReaderStage": ("nemo_curator.stages.audio.common", "ManifestReaderStage"),
    "ManifestWriterStage": ("nemo_curator.stages.audio.common", "ManifestWriterStage"),
    "PreserveByValueStage": ("nemo_curator.stages.audio.common", "PreserveByValueStage"),
    # Filtering
    "BandFilterStage": ("nemo_curator.stages.audio.filtering", "BandFilterStage"),
    "SIGMOSFilterStage": ("nemo_curator.stages.audio.filtering", "SIGMOSFilterStage"),
    "UTMOSFilterStage": ("nemo_curator.stages.audio.filtering", "UTMOSFilterStage"),
    # Inference
    "InferenceQwenOmniStage": ("nemo_curator.stages.audio.inference.qwen_omni", "InferenceQwenOmniStage"),
    "InferenceQwenASRStage": ("nemo_curator.stages.audio.inference.qwen_asr", "InferenceQwenASRStage"),
    # I/O
    "ALMManifestReader": ("nemo_curator.stages.audio.io.alm_manifest_reader", "ALMManifestReader"),
    "ALMManifestReaderStage": ("nemo_curator.stages.audio.io.alm_manifest_reader", "ALMManifestReaderStage"),
    "ALMManifestWriterStage": ("nemo_curator.stages.audio.io.alm_manifest_writer", "ALMManifestWriterStage"),
    "NemoTarredAudioReader": ("nemo_curator.stages.audio.io.nemo_tarred_reader", "NemoTarredAudioReader"),
    "SegmentExtractionStage": ("nemo_curator.stages.audio.io.extract_segments", "SegmentExtractionStage"),
    "ShardedManifestWriterStage": ("nemo_curator.stages.audio.io.sharded_manifest_writer", "ShardedManifestWriterStage"),
    # Postprocessing
    "TimestampMapperStage": ("nemo_curator.stages.audio.postprocessing", "TimestampMapperStage"),
    # Preprocessing
    "MonoConversionStage": ("nemo_curator.stages.audio.preprocessing", "MonoConversionStage"),
    "SegmentConcatenationStage": ("nemo_curator.stages.audio.preprocessing", "SegmentConcatenationStage"),
    # Segmentation
    "SpeakerSeparationStage": ("nemo_curator.stages.audio.segmentation", "SpeakerSeparationStage"),
    "VADSegmentationStage": ("nemo_curator.stages.audio.segmentation", "VADSegmentationStage"),
    # Text filtering
    "AbbreviationConcatStage": ("nemo_curator.stages.audio.text_filtering", "AbbreviationConcatStage"),
    "DisfluencyWerGuardStage": ("nemo_curator.stages.audio.text_filtering", "DisfluencyWerGuardStage"),
    "FastTextLIDStage": ("nemo_curator.stages.audio.text_filtering", "FastTextLIDStage"),
    "FinalizeFieldsStage": ("nemo_curator.stages.audio.text_filtering", "FinalizeFieldsStage"),
    "ITNRestorationStage": ("nemo_curator.stages.audio.text_filtering", "ITNRestorationStage"),
    "InitializeFieldsStage": ("nemo_curator.stages.audio.text_filtering", "InitializeFieldsStage"),
    "PnCContentGuardStage": ("nemo_curator.stages.audio.text_filtering", "PnCContentGuardStage"),
    "PnCRestorationStage": ("nemo_curator.stages.audio.text_filtering", "PnCRestorationStage"),
    "RegexSubstitutionStage": ("nemo_curator.stages.audio.text_filtering", "RegexSubstitutionStage"),
    "SelectBestPredictionStage": ("nemo_curator.stages.audio.text_filtering", "SelectBestPredictionStage"),
    "WhisperHallucinationStage": ("nemo_curator.stages.audio.text_filtering", "WhisperHallucinationStage"),
}


def __getattr__(name: str) -> type:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path)
        return getattr(module, attr)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
