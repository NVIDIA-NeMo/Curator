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

"""Compatibility imports for the adapter-backed Sortformer stage.

New configurations should import :class:`InferenceSortformerStage` from
``speaker_diarization.stage``. This module remains so existing pipelines keep
their established import path without loading NeMo on the driver.
"""

from nemo_curator.models.audio.speaker_diarization.sortformer import parse_sortformer_segments
from nemo_curator.stages.audio.inference.speaker_diarization.stage import (
    InferenceSortformerStage,
    _write_rttm,
)

_parse_sortformer_segments = parse_sortformer_segments

__all__ = [
    "InferenceSortformerStage",
    "_parse_sortformer_segments",
    "_write_rttm",
    "parse_sortformer_segments",
]
