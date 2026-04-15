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
VAD segmentation, bandwidth classification filtering,
audio preprocessing (mono conversion, segment concatenation, timestamp mapping),
audio quality filtering (UTMOS), and speaker diarization/separation.

Imports are guarded with try/except so that missing optional dependencies
(e.g. NeMo, pydub, pyloudnorm) do not prevent importing stages that
do not need them.
"""

import importlib
import logging

_logger = logging.getLogger(__name__)

__all__: list[str] = []


def _try_import(module_path: str, names: list[str]) -> None:
    """Import *names* from *module_path*, silently skipping on failure."""
    try:
        mod = importlib.import_module(module_path)
        for name in names:
            globals()[name] = getattr(mod, name)
            __all__.append(name)
    except Exception as exc:  # noqa: BLE001
        _logger.debug("Skipping %s: %s", module_path, exc)


_try_import("nemo_curator.stages.audio.alm", ["ALMDataBuilderStage", "ALMDataOverlapStage"])
_try_import("nemo_curator.stages.audio.common", ["GetAudioDurationStage", "PreserveByValueStage"])
_try_import("nemo_curator.stages.audio.filtering", ["BandFilterStage", "UTMOSFilterStage"])
_try_import("nemo_curator.stages.audio.postprocessing", ["TimestampMapperStage"])
_try_import("nemo_curator.stages.audio.preprocessing", ["MonoConversionStage", "SegmentConcatenationStage"])
_try_import("nemo_curator.stages.audio.segmentation", ["SpeakerSeparationStage", "VADSegmentationStage"])
