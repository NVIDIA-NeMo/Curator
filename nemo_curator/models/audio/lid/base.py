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

"""Stage-adapter contract for spoken-language identification.

The Curator stage owns task I/O, waveform normalization, ensemble tags, resume
behavior, and manifest mutation. An ``AudioLIDAdapter`` owns provider-specific
weight resolution, model lifecycle, model batching, and conversion of raw model
outputs into a common language/confidence pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class AudioLIDResult:
    """Canonical language-identification result for one waveform.

    ``language`` is a normalized lowercase language code. An empty string with
    confidence ``0.0`` is a completed "no language" result, not a missing
    result; this distinction is important for ensemble resume semantics.
    """

    language: str
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.language, str):
            msg = f"AudioLIDResult.language must be a string, got {type(self.language).__name__}"
            raise TypeError(msg)
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            msg = f"AudioLIDResult.confidence must be finite and in [0, 1], got {self.confidence!r}"
            raise ValueError(msg)
        object.__setattr__(self, "language", self.language.strip().lower())
        object.__setattr__(self, "confidence", confidence)


@runtime_checkable
class AudioLIDAdapter(Protocol):
    """Structural protocol implemented by every spoken-LID adapter.

    Constructor contract: the generic stage constructs an adapter as
    ``cls(sample_rate=..., **adapter_kwargs)``. The stage's stable ensemble
    result key is deliberately not part of this protocol: provider model
    identifiers belong in adapter-specific fields such as ``source``,
    ``model_name``, or ``model_size``.

    ``identify_batch`` receives one dictionary per eligible task. Each item
    contains a contiguous mono 1-D float32 NumPy ``waveform`` at
    ``sample_rate``. The adapter must return exactly one result per item in the
    same order.
    """

    sample_rate: int

    def download_weights_on_node(self) -> None:
        """Cache model artifacts without retaining worker-local model state."""
        ...

    def load_model(self, *, num_gpus: int) -> None:
        """Load worker-local model state for the requested physical GPU count."""
        ...

    def unload_model(self) -> None:
        """Release worker-local model and accelerator state."""
        ...

    def identify_batch(self, items: list[dict[str, Any]]) -> list[AudioLIDResult]:
        """Return one canonical result per prepared waveform, in order."""
        ...


def _validate_sample_rate(sample_rate: object, *, owner: str) -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral) or sample_rate <= 0:
        msg = f"{owner}.sample_rate must be a positive integer, got {sample_rate!r}"
        raise ValueError(msg)
    return int(sample_rate)


def _validate_single_device_gpu_count(num_gpus: object, *, owner: str, gpu_required: bool = False) -> int:
    allowed = {1} if gpu_required else {0, 1}
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, Integral) or num_gpus not in allowed:
        expected = "exactly 1" if gpu_required else "0 or 1"
        msg = f"{owner} requires num_gpus to be {expected}, got {num_gpus!r}"
        raise ValueError(msg)
    return int(num_gpus)


def _waveform_from_item(item: dict[str, Any], *, owner: str) -> np.ndarray:
    if "waveform" not in item:
        msg = f"{owner} requires each item to contain 'waveform'"
        raise KeyError(msg)
    waveform = np.asarray(item["waveform"], dtype=np.float32)
    if waveform.ndim != 1:
        msg = f"{owner} expects a mono 1-D waveform, got shape {waveform.shape}"
        raise ValueError(msg)
    return np.ascontiguousarray(waveform, dtype=np.float32)
