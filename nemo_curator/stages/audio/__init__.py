# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
including ASR inference, quality assessment, and ALM data preparation.
"""

from nemo_curator.stages.audio.alm import ALMDataBuilderStage, ALMDataOverlapStage
from nemo_curator.stages.audio.common import (
    GetAudioDurationStage,
    LegacySpeechStage,
    PreserveByValueStage,
)

__all__ = [
    "ALMDataBuilderStage",
    "ALMDataOverlapStage",
    "GetAudioDurationStage",
    "LegacySpeechStage",
    "PreserveByValueStage",
]
