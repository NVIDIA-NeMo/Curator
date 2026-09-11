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

"""Resource calibration (1C.2): measured per-stage resources refine the plan.

A smoke run observes what each stage actually used; :func:`from_smoke` extracts
those measurements so the resource planner can raise a card's ``best_guess``
``resource`` facts when the smoke observed a larger peak. A bounded smoke cannot
prove the full run's maximum, so it never lowers the card/default estimate. This
is the plumbing — the *measurements themselves* come from a real (GPU) smoke; on
a CPU smoke there is no VRAM to read, so extraction is empty and the planner
falls back to card facts.

Nothing here is shared across users/sessions: a calibration is derived from *this*
smoke and applies to *this* machine (stamped with the machine fingerprint), i.e. an
operational, recomputable annotation — consistent with the no-memory stance.
"""

from __future__ import annotations

import math
from typing import Any

# Candidate per-stage perf-metric names a smoke may expose for each resource fact.
# (StagePerfStats keys vary by backend/instrumentation; we read whichever is present.)
_VRAM_KEYS = ("peak_vram_gb", "gpu_mem_gb", "vram_gb", "custom.peak_vram_gb", "custom.gpu_mem_gb")
_HOST_MEM_KEYS = ("peak_host_mem_gb", "host_mem_gb", "rss_gb", "custom.peak_host_mem_gb")
_THROUGHPUT_KEYS = ("throughput", "items_per_sec", "custom.throughput")


def _reported_machine_fingerprint(smoke_report: dict[str, Any]) -> str | None:
    """Recover a unique machine fingerprint already carried by a smoke report."""
    candidates: set[str] = set()

    def add(value: Any) -> None:  # noqa: ANN401
        if isinstance(value, str) and value.strip():
            candidates.add(value)

    add(smoke_report.get("machine_fingerprint"))
    resource_plan = smoke_report.get("resource_plan")
    if isinstance(resource_plan, dict):
        add(resource_plan.get("machine_fingerprint"))

    existing = smoke_report.get("calibration")
    if isinstance(existing, dict):
        add(existing.get("machine_fingerprint"))
        entries = existing.get("calibration", existing)
        if isinstance(entries, dict):
            for entry in entries.values():
                if isinstance(entry, dict):
                    add(entry.get("machine_fingerprint"))
    return next(iter(candidates)) if len(candidates) == 1 else None


def _valid_measurement(value: Any) -> bool:  # noqa: ANN401
    """Measured resources/rates must be finite, non-negative real numbers."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _read_metric(metrics: dict[str, Any], keys: tuple[str, ...], *, prefer: str = "max") -> float | None:
    """Read the first present metric among ``keys``.

    ``per_stage_metrics`` values are ``{sum, mean, min, max, count}`` aggregates,
    so resource peaks prefer ``max`` while throughput prefers ``mean``. Older
    reports without extrema fall back to mean -> sum. A bare number is taken as-is.
    """
    for k in keys:
        v = metrics.get(k)
        if isinstance(v, dict):
            for stat in (prefer, "mean", "sum"):
                if stat in v:
                    # An explicitly present but invalid preferred statistic makes
                    # this metric untrustworthy; do not silently substitute a less
                    # conservative aggregate from the same reading.
                    return float(v[stat]) if _valid_measurement(v[stat]) else None
        elif _valid_measurement(v):
            return float(v)
    return None


def from_smoke(smoke_report: dict[str, Any] | None, *, machine_fingerprint: str | None = None) -> dict[str, Any]:
    """Measured per-stage resource facts from a smoke report's ``per_stage_metrics``.

    Returns ``{stage: {gpu_mem_gb?, host_mem_gb?, throughput?, source: "measured",
    machine_fingerprint?}}`` for stages that carry a reading; empty when the perf
    stats have no resource measurements (e.g. a CPU smoke). Feed to
    ``run(..., calibration=...)`` so the planner conservatively takes the larger
    of the measured and card/default values.
    """
    report = smoke_report or {}
    per_stage = report.get("per_stage_metrics") or {}
    fingerprint = (
        machine_fingerprint
        if isinstance(machine_fingerprint, str) and machine_fingerprint.strip()
        else _reported_machine_fingerprint(report)
    )
    out: dict[str, Any] = {}
    for stage, metrics in per_stage.items():
        if not isinstance(metrics, dict):
            continue
        entry: dict[str, Any] = {}
        vram = _read_metric(metrics, _VRAM_KEYS)
        if vram is not None:
            entry["gpu_mem_gb"] = round(vram, 3)
        host = _read_metric(metrics, _HOST_MEM_KEYS)
        if host is not None:
            entry["host_mem_gb"] = round(host, 3)
        thr = _read_metric(metrics, _THROUGHPUT_KEYS, prefer="mean")
        if thr is not None:
            entry["throughput"] = round(thr, 3)
        if entry:
            entry["source"] = "measured"
            if fingerprint:
                entry["machine_fingerprint"] = fingerprint
            out[stage] = entry
    return out
