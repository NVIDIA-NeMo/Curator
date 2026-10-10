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

"""Stage-adapter contract for voice-activity detection.

VAD stages own Curator task I/O and segment fan-out.  Adapters own the model
lifecycle, model-specific sample-rate handling, inference, and conversion to
the canonical time-in-seconds result below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Protocol, runtime_checkable


def _validate_detection_options(  # noqa: PLR0913
    *,
    threshold: float,
    min_duration_sec: float,
    max_duration_sec: float,
    min_interval_ms: int,
    speech_pad_ms: int,
    owner: str = "",
) -> None:
    prefix = f"{owner}." if owner else ""
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        msg = f"{prefix}threshold must be finite and in [0, 1], got {threshold!r}"
        raise ValueError(msg)
    minimum = float(min_duration_sec)
    maximum = float(max_duration_sec)
    if not math.isfinite(minimum) or minimum < 0:
        msg = f"{prefix}min_duration_sec must be finite and non-negative, got {min_duration_sec!r}"
        raise ValueError(msg)
    if not maximum > minimum:
        msg = f"{prefix}max_duration_sec must be greater than min_duration_sec, got {max_duration_sec!r}"
        raise ValueError(msg)
    for name, value in (("min_interval_ms", min_interval_ms), ("speech_pad_ms", speech_pad_ms)):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            msg = f"{prefix}{name} must be a non-negative integer, got {value!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class VADSegment:
    """One half-open speech interval in seconds."""

    start: float
    end: float


@dataclass
class VADResult:
    """Canonical VAD output for one input waveform.

    ``error`` is reserved for a failure attributable to this recording after a
    provider accepted the batch. Batch-wide inference failures should still be
    raised by the adapter because they cannot be assigned to one input safely.
    """

    segments: list[VADSegment]
    error: str | None = None


@runtime_checkable
class VADAdapter(Protocol):
    """Structural protocol implemented by voice-activity model adapters.

    ``detect_batch`` receives stage-normalized items in input order.  Every
    item contains a contiguous mono waveform and its original ``sample_rate``.
    The adapter must return exactly one result per item in the same order.
    """

    def download_weights_on_node(self) -> None:
        """Cache model assets without allocating worker-local model state."""
        ...

    def load_model(self, *, num_gpus: int) -> None:
        """Load worker-local state for the requested physical GPU count."""
        ...

    def unload_model(self) -> None:
        """Release worker-local model and accelerator state."""
        ...

    def detect_batch(self, items: list[dict[str, Any]]) -> list[VADResult]:
        """Return one canonical result per input item, preserving order."""
        ...
