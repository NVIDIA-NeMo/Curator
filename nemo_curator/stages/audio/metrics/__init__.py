# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Audio quality and accuracy metrics stages (lazy imports)."""

import importlib
from typing import Any

_LAZY_IMPORTS = {
    "BandwidthEstimationStage": "nemo_curator.stages.audio.metrics.bandwidth",
    "ComputeWERStage": "nemo_curator.stages.audio.metrics.wer",
    "GetPairwiseWerStage": "nemo_curator.stages.audio.metrics.wer",
    "TorchSquimQualityMetricsStage": "nemo_curator.stages.audio.metrics.squim",
}

_cache: dict[str, Any] = {}


def __getattr__(name: str) -> Any:  # noqa: ANN401
    if name in _cache:
        return _cache[name]

    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        attr = getattr(module, name)
        _cache[name] = attr
        return attr

    msg = f"module 'nemo_curator.stages.audio.metrics' has no attribute '{name}'"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return list(_LAZY_IMPORTS.keys())


__all__ = list(_LAZY_IMPORTS.keys())
