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

"""Lazy public API for audio voice-activity model adapters."""

from __future__ import annotations

from importlib import import_module

_LAZY = {
    "SileroVADAdapter": "nemo_curator.models.audio.vad.silero",
    "TensorRTSileroVADAdapter": "nemo_curator.models.audio.vad.silero_tensorrt",
    "VADAdapter": "nemo_curator.models.audio.vad.base",
    "VADResult": "nemo_curator.models.audio.vad.base",
    "VADSegment": "nemo_curator.models.audio.vad.base",
}

__all__ = [
    "SileroVADAdapter",
    "TensorRTSileroVADAdapter",
    "VADAdapter",
    "VADResult",
    "VADSegment",
]


def __getattr__(name: str) -> object:
    target = _LAZY.get(name)
    if target is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(import_module(target), name)


def __dir__() -> list[str]:
    return __all__
