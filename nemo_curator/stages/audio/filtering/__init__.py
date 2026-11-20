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

"""Audio quality filtering stages.

Importing one filter must not import another's model stack (SIGMOS pulls onnxruntime and librosa). Concrete stages are therefore resolved only when requested.
"""

from __future__ import annotations

from importlib import import_module

_LAZY = {
    "BandFilterStage": "nemo_curator.stages.audio.filtering.band",
    "SIGMOSFilterStage": "nemo_curator.stages.audio.filtering.sigmos",
    "UTMOSFilterStage": "nemo_curator.stages.audio.filtering.utmos",
}

__all__ = ["BandFilterStage", "SIGMOSFilterStage", "UTMOSFilterStage"]


def __getattr__(name: str) -> object:
    target = _LAZY.get(name)
    if target is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(import_module(target), name)


def __dir__() -> list[str]:
    return __all__
