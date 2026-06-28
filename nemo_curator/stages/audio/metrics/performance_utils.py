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

"""Pure post-processing helpers for the audio pipeline performance summary.

Owns everything that runs on top of the raw counters/samples collected by the
``performance.py`` accumulator: percentile computation, unit conversions, safe
ratio helpers, and audio-domain composites (``audio_hours_per_gpu_hour`` etc.).
Pure functions so they can be unit-tested and reused (CI checks, dashboards)
without the full accumulator.

NOTE: shadows ``nemo_curator.utils.performance_utils`` by filename only; the
import paths differ so there is no conflict. That module owns ``StagePerfStats``;
this one owns audio-specific post-processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

_MAX_PERCENTILE = 100.0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECONDS_PER_HOUR = 3600.0
BYTES_PER_MB = 1024.0 * 1024.0
BYTES_PER_GB = 1024.0 * 1024.0 * 1024.0
DEFAULT_PERCENTILES: tuple[int, ...] = (50, 95)


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------


def seconds_to_hours(seconds: float) -> float:
    """Convert seconds to hours."""
    return float(seconds) / SECONDS_PER_HOUR


def hours_to_seconds(hours: float) -> float:
    return float(hours) * SECONDS_PER_HOUR


def bytes_to_mb(b: float) -> float:
    return float(b) / BYTES_PER_MB


def bytes_to_gb(b: float) -> float:
    return float(b) / BYTES_PER_GB


# ---------------------------------------------------------------------------
# Safe ratio
# ---------------------------------------------------------------------------


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """Return numerator/denominator, or None if either input is non-positive.

    None (not 0/NaN) lets the consumer omit the field rather than carry a
    misleading zero downstream.
    """
    if numerator is None or denominator is None:
        return None
    if numerator <= 0 or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def add_ratio(entry: dict[str, Any], name: str, numerator: float, denominator: float) -> None:
    """Add ``entry[name] = numerator/denominator`` only when both > 0."""
    value = safe_ratio(numerator, denominator)
    if value is not None:
        entry[name] = value


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


def _percentile_sorted(sorted_values: list[float], p: float) -> float:
    """``p``-th percentile of an already-sorted, non-empty list (linear interp)."""
    if p < 0 or p > _MAX_PERCENTILE:
        msg = f"percentile p must be in [0, 100], got {p}"
        raise ValueError(msg)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    if lo == hi:
        return sorted_values[lo]
    frac = rank - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def percentile(values: Iterable[float], p: float) -> float | None:
    """Compute the p-th percentile of ``values`` with linear interpolation.

    Mirrors numpy's default but avoids importing numpy so it works in writer
    pods without it. Returns None when empty; ``p`` must be in ``[0, 100]``.
    """
    materialized = [float(v) for v in values]
    if not materialized:
        return None
    materialized.sort()
    return _percentile_sorted(materialized, p)


def summarize_samples(
    values: Iterable[float],
    name: str,
    percentiles: Iterable[int] = DEFAULT_PERCENTILES,
) -> dict[str, float]:
    """Return ``{f"{name}_p{P}": value}`` for each requested percentile.

    Empty samples -> empty dict. Sorts once, then indexes each percentile off it.
    """
    out: dict[str, float] = {}
    materialized = [float(v) for v in values]
    if not materialized:
        return out
    materialized.sort()
    for p in percentiles:
        out[f"{name}_p{p}"] = _percentile_sorted(materialized, p)
    return out


# ---------------------------------------------------------------------------
# Audio-domain composites
# ---------------------------------------------------------------------------


def audio_hours_per_gpu_hour(audio_seconds: float, gpu_seconds: float) -> float | None:
    """Hours of audio processed per GPU-hour spent.

    ``gpu_seconds`` is ``gpu_count * wallclock_s`` (caller-computed). Returns
    None when either input is non-positive.
    """
    return safe_ratio(seconds_to_hours(audio_seconds), seconds_to_hours(gpu_seconds))


def items_per_hour(items: float, wall_seconds: float) -> float | None:
    """Generic throughput in items / wallclock-hour."""
    if wall_seconds <= 0:
        return None
    return safe_ratio(items, seconds_to_hours(wall_seconds))


def estimate_wallclock_s(
    total_process_time_s: float,
    actor_count: float | None = None,
) -> float | None:
    """Best-effort stage wallclock estimate.

    ``process_time`` sums CPU time across an actor's invocations; true per-stage
    wall would need first/last timestamps the framework doesn't expose. So:
    divide by ``actor_count`` when positive (spread across concurrent actors),
    else use ``total_process_time_s``. No ``max(invocation_times)`` fallback --
    it reads optimistically under parallel actors, so we stay conservative.
    """
    if actor_count and actor_count > 0:
        return float(total_process_time_s) / float(actor_count)
    if total_process_time_s > 0:
        return float(total_process_time_s)
    return None
