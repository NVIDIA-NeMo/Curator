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

"""Audio curation stages for NeMo Curator.

Lazy attribute resolution (PEP 562). Eagerly importing every subpackage at
package-init time pulls in heavy ML deps (e.g. NeMo ASR -> lightning.pytorch
-> torchvision) that most consumers don't need. Worse, it makes importing
*any* nemo_curator.stages.audio.* submodule fail when those heavy deps are
broken in the environment. Resolving names on first attribute access fixes
both: each name still imports lazily from the same submodule it used to,
preserving the public API.
"""

from importlib import import_module

# Map each public name to the submodule that defines it. Adding/removing
# a stage means updating this dict and `__all__` together.
_LAZY = {
    "ALMDataBuilderStage": "nemo_curator.stages.audio.alm",
    "ALMDataOverlapStage": "nemo_curator.stages.audio.alm",
    "AudioDataFilterStage": "nemo_curator.stages.audio.advanced_pipelines",
    "BandFilterStage": "nemo_curator.stages.audio.filtering",
    "GetAudioDurationStage": "nemo_curator.stages.audio.common",
    "ManifestReader": "nemo_curator.stages.audio.common",
    "ManifestWriterStage": "nemo_curator.stages.audio.common",
    "MonoConversionStage": "nemo_curator.stages.audio.preprocessing",
    "PreserveByValueStage": "nemo_curator.stages.audio.common",
    "SIGMOSFilterStage": "nemo_curator.stages.audio.filtering",
    "SegmentConcatenationStage": "nemo_curator.stages.audio.preprocessing",
    "SpeakerSeparationStage": "nemo_curator.stages.audio.segmentation",
    "TimestampMapperStage": "nemo_curator.stages.audio.postprocessing",
    "UTMOSFilterStage": "nemo_curator.stages.audio.filtering",
    "VADSegmentationStage": "nemo_curator.stages.audio.segmentation",
}

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


def __getattr__(name: str) -> object:
    target = _LAZY.get(name)
    if target is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(import_module(target), name)


def __dir__() -> list[str]:
    return __all__
