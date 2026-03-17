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

"""Audio tagging stages -- processors for labeling unlabelled audio data.

Covers VAD, diarization, ASR alignment, text normalization, quality metrics,
and segment preparation.  Applicable across modalities (ASR, TTS, etc.).

Uses lazy imports to avoid loading heavy dependencies (NeMo, PyAnnote, etc.)
until they are actually needed.
"""

import importlib
from typing import Any

_LAZY_IMPORTS = {
    # --- Manifest I/O (now in common.py) ---
    "ManifestReader": "nemo_curator.stages.audio.common",
    "ManifestReaderStage": "nemo_curator.stages.audio.common",
    "ManifestWriterStage": "nemo_curator.stages.audio.common",
    # --- Preprocessing (tagging/) ---
    "ResampleAudioStage": "nemo_curator.stages.audio.tagging.resample_audio",
    "SplitLongAudioStage": "nemo_curator.stages.audio.tagging.split",
    "JoinSplitAudioMetadataStage": "nemo_curator.stages.audio.tagging.split",
    "SplitASRAlignJoinStage": "nemo_curator.stages.audio.tagging.split",
    "MergeAlignmentDiarizationStage": "nemo_curator.stages.audio.tagging.merge_alignment_diarization",
    "PrepareModuleSegmentsStage": "nemo_curator.stages.audio.tagging.prepare_module_segments",
    # --- Inference (tagging/inference/) ---
    "BaseASRProcessorStage": "nemo_curator.stages.audio.tagging.inference.nemo_asr_align",
    "NeMoASRAlignerStage": "nemo_curator.stages.audio.tagging.inference.nemo_asr_align",
    "PyAnnoteDiarizationStage": "nemo_curator.stages.audio.inference.speaker_diarization.pyannote",
    "WhisperXVADStage": "nemo_curator.stages.audio.inference.vad.whisperx_vad",
    # --- Metrics (tagging/metrics/) ---
    "TorchSquimQualityMetricsStage": "nemo_curator.stages.audio.tagging.metrics.squim",
    "BandwidthEstimationStage": "nemo_curator.stages.audio.tagging.metrics.bandwidth",
    "ComputeWERStage": "nemo_curator.stages.audio.tagging.metrics.wer",
    # --- Text processing (tagging/text/) ---
    "InverseTextNormalizationStage": "nemo_curator.stages.audio.tagging.text.itn",
    "ArabicRemoveDiacriticsStage": "nemo_curator.stages.audio.tagging.text.arabic_remove_diacritics",
    "PNCwithBERTStage": "nemo_curator.stages.audio.tagging.text.pnc",
    "ChineseConversionStage": "nemo_curator.stages.audio.tagging.text.chinese_conversion",
}

_cache: dict[str, Any] = {}


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy import handler - only imports modules when accessed."""
    if name in _cache:
        return _cache[name]

    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        attr = getattr(module, name)
        _cache[name] = attr
        return attr

    msg = f"module 'nemo_curator.stages.audio.tagging' has no attribute '{name}'"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """List available attributes for autocomplete."""
    return list(_LAZY_IMPORTS.keys())


__all__ = list(_LAZY_IMPORTS.keys())
