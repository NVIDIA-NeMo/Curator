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

Submodules are imported lazily to avoid hard dependency on
``nemo_curator.backends.experimental`` (not present in all environments).
Import individual stages directly, e.g.::

    from nemo_curator.stages.audio.segmentation.vad_segmentation import VADSegmentationStage
"""

__all__: list[str] = []


def __getattr__(name: str):
    if name == "VADSegmentationStage":
        from .vad_segmentation import VADSegmentationStage

        return VADSegmentationStage
    if name == "SpeakerSeparationStage":
        from .speaker_separation import SpeakerSeparationStage

        return SpeakerSeparationStage
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
