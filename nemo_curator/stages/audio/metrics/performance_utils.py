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

The ``performance.py`` accumulator class collects raw counters and
per-invocation samples from each audio stage's ``_log_metrics`` output.
This module owns everything that runs *on top of* those raw values to
produce the published ``perf_summary.json`` shape: percentile
computation, unit conversions, safe ratio helpers, and audio-domain
composite throughput metrics (``audio_hours_per_gpu_hour`` etc.).

Keeping these as pure functions in a separate module means they can be
unit-tested in isolation and reused by anything that consumes the
audio perf data (e.g. CI throughput checks, dashboards, Kratos perf
harvesters) without dragging in the full
``AudioPerformanceSummary`` accumulator.

NOTE: This module shadows ``nemo_curator.utils.performance_utils`` by
filename only; the full import paths differ
(``nemo_curator.stages.audio.metrics.performance_utils`` vs
``nemo_curator.utils.performance_utils``) so there is no actual
conflict. The framework-level module owns ``StagePerfStats``; this
module owns audio-specific post-processing.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

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

    Returning None (rather than 0 or NaN) lets the consumer omit the
    field entirely from the published summary instead of carrying a
    misleading zero through downstream dashboards.
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

def percentile(values: Iterable[float], p: float) -> float | None:
    """Compute the p-th percentile of ``values`` with linear interpolation.

    Mirrors numpy's default behaviour (``np.percentile`` with
    ``interpolation='linear'``) but avoids importing numpy from this
    module so it can be used inside a writer pod that may not have
    numpy on the import path.

    Returns None when ``values`` is empty. ``p`` must be in ``[0, 100]``.
    """
    materialized = [float(v) for v in values]
    if not materialized:
        return None
    if p < 0 or p > 100:
        msg = f"percentile p must be in [0, 100], got {p}"
        raise ValueError(msg)
    materialized.sort()
    if len(materialized) == 1:
        return materialized[0]
    rank = (len(materialized) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(materialized) - 1)
    if lo == hi:
        return materialized[lo]
    frac = rank - lo
    return materialized[lo] + frac * (materialized[hi] - materialized[lo])


def summarize_samples(
    values: Iterable[float],
    name: str,
    percentiles: Iterable[int] = DEFAULT_PERCENTILES,
) -> dict[str, float]:
    """Return ``{f"{name}_p{P}": value}`` for each requested percentile.

    Empty samples -> empty dict (caller omits the field entirely).
    """
    out: dict[str, float] = {}
    materialized = [float(v) for v in values]
    if not materialized:
        return out
    for p in percentiles:
        v = percentile(materialized, p)
        if v is not None:
            out[f"{name}_p{p}"] = v
    return out


# ---------------------------------------------------------------------------
# Audio-domain composites
# ---------------------------------------------------------------------------

def audio_hours_per_gpu_hour(audio_seconds: float, gpu_seconds: float) -> float | None:
    """How many hours of audio the stage processes per GPU-hour spent.

    ``gpu_seconds`` is ``gpu_count * wallclock_s`` for the stage (the
    caller computes this from its cluster topology). Returns None when
    either input is non-positive.
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
    invocation_times: Iterable[float] | None = None,
) -> float | None:
    """Best-effort stage wallclock estimate.

    The framework's ``StagePerfStats.process_time`` sums CPU time across
    all invocations of an actor (and ``invocation_count`` is the number
    of dedup'd invocations across all actors). True per-stage wall would
    require first/last invocation timestamps, which the framework does
    not currently expose.

    Estimation order:
      1. If ``actor_count`` is positive: ``total_process_time_s / actor_count``.
      2. Else if ``invocation_times`` is non-empty: ``max(invocation_times)``
         (longest single invocation = lower bound on wall).
      3. Else: ``total_process_time_s`` (single-actor stages).
    """
    if actor_count and actor_count > 0:
        return float(total_process_time_s) / float(actor_count)
    if invocation_times is not None:
        materialized = [float(t) for t in invocation_times]
        if materialized:
            return max(materialized)
    if total_process_time_s > 0:
        return float(total_process_time_s)
    return None
