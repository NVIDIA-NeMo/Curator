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

"""Speaker-diarization model adapters.

Concrete adapters are intentionally not imported here. Hydra resolves them
from their complete module paths without making a package import load NeMo,
Hugging Face Hub, or a model checkpoint.
"""

from __future__ import annotations

from importlib import import_module

_LAZY = {
    "DiarizationAdapter": "nemo_curator.models.audio.speaker_diarization.base",
    "DiarizationResult": "nemo_curator.models.audio.speaker_diarization.base",
    "DiarizationSegment": "nemo_curator.models.audio.speaker_diarization.base",
    "NeMoSortformerAdapter": "nemo_curator.models.audio.speaker_diarization.sortformer",
    "parse_sortformer_segments": "nemo_curator.models.audio.speaker_diarization.sortformer",
}

__all__ = [
    "DiarizationAdapter",
    "DiarizationResult",
    "DiarizationSegment",
    "NeMoSortformerAdapter",
    "parse_sortformer_segments",
]


def __getattr__(name: str) -> object:
    target = _LAZY.get(name)
    if target is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(import_module(target), name)


def __dir__() -> list[str]:
    return __all__
