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

"""Audio segmentation stages.

Submodules are imported lazily so optional segmentation dependencies are only
loaded when a specific stage is requested.
"""

__all__ = ["SegmentExtractorStage", "SpeakerSeparationStage", "VADSegmentationStage"]


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "SegmentExtractorStage": (
        "nemo_curator.stages.audio.segmentation.segment_extractor",
        "SegmentExtractorStage",
    ),
    "SpeakerSeparationStage": (
        "nemo_curator.stages.audio.segmentation.speaker_separation",
        "SpeakerSeparationStage",
    ),
    "VADSegmentationStage": ("nemo_curator.stages.audio.segmentation.vad_segmentation", "VADSegmentationStage"),
}


def __getattr__(name: str) -> type:
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
