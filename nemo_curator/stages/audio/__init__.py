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
NeMo Curator Audio Processing Stages.

Preprocessing:
    - MonoConversionStage: Convert multi-channel audio to mono
    - SegmentConcatenationStage: Concatenate audio segments
"""

from nemo_curator.stages.audio.preprocessing import (
    MonoConversionStage,
    SegmentConcatenationStage,
)

from nemo_curator.stages.audio.postprocessing import (
    TimestampMapperStage,
)

from nemo_curator.stages.audio.configs import (
    MonoConversionConfig,
    SegmentConcatenationConfig,
    TimestampMapperConfig,
)

__all__ = [
    "MonoConversionStage",
    "SegmentConcatenationStage",
    "TimestampMapperStage",
    "MonoConversionConfig",
    "SegmentConcatenationConfig",
    "TimestampMapperConfig",
]
