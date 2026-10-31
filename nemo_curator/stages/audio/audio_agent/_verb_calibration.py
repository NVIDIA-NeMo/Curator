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

"""Measured resource calibration."""

from __future__ import annotations

from typing import Any


def calibrate(smoke_report: dict[str, Any]) -> dict[str, Any]:
    """Extract measured per-stage resources from a smoke report (1C.2).

    Returns ``{calibration: {stage: {gpu_mem_gb?, host_mem_gb?, ...}}}`` to pass to
    ``run(..., calibration=...)`` so the planner can raise card ``best_guess``
    facts when the smoke observed a larger peak. A bounded smoke never lowers a
    card/default estimate. A CPU smoke still measures host RAM and throughput (it
    only lacks VRAM), and host RAM is what decides streaming vs batch.

    Calling this is optional: ``smoke`` already stores its measurements under the
    recipe's ``config_hash`` and ``run`` applies them when no calibration is passed.
    Extract them here to inspect, archive, or hand a run measurements it would not
    otherwise find (e.g. from a differently sized machine).
    """
    from nemo_curator.stages.audio.audio_agent import calibration as _cal

    existing = smoke_report.get("calibration")
    if isinstance(existing, dict):
        # ``smoke`` already extracted and machine-stamped these measurements.
        # Re-extracting from the surrounding report silently discarded that
        # binding and let measurements migrate to another machine.
        result: dict[str, Any] = {"calibration": existing}
        wrapper_fingerprint = smoke_report.get("machine_fingerprint")
        if not wrapper_fingerprint:
            fingerprints = {
                entry.get("machine_fingerprint")
                for entry in existing.values()
                if isinstance(entry, dict) and entry.get("machine_fingerprint")
            }
            if len(fingerprints) == 1:
                wrapper_fingerprint = fingerprints.pop()
        if isinstance(wrapper_fingerprint, str) and wrapper_fingerprint:
            result["machine_fingerprint"] = wrapper_fingerprint
        return result

    machine_fingerprint = smoke_report.get("machine_fingerprint")
    return {
        **(
            {"machine_fingerprint": machine_fingerprint}
            if isinstance(machine_fingerprint, str) and machine_fingerprint
            else {}
        ),
        "calibration": _cal.from_smoke(
            smoke_report,
            machine_fingerprint=(machine_fingerprint if isinstance(machine_fingerprint, str) else None),
        ),
    }
