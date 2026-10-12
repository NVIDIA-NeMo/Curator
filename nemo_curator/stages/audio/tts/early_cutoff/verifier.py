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

"""Frozen early cut-off MFA verifier.

Combines CTC endpoint-gate evidence with MFA alignment evidence. Every raw evidence value is mapped to its
percentile in the calibration's clean-control ``null_evidence`` distribution; the final score is compared against
the calibrated ``verifier_threshold`` by the resolver.
"""

from __future__ import annotations

import bisect
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

EVIDENCE_NAMES = (
    "ctc_age",
    "ctc_censor",
    "suffix_non_recovery",
    "low_last_phone_confidence",
    "eou",
    "whole_phone",
    "mfa_bad_local",
    "mfa_bad_alignment",
    "mfa_short_duration",
    "mfa_short_run",
)


def _finite(value: Any, default: float = 0.0) -> float:  # noqa: ANN401
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _mapping(value: Any) -> Mapping[str, Any]:  # noqa: ANN401
    return value if isinstance(value, dict) else {}


def raw_evidence(row: Mapping[str, Any]) -> dict[str, float]:
    """Raw evidence channels from ``row["prediction"]`` (gate annotation) and ``row["mfa"]`` (MFA evidence)."""
    prediction = _mapping(row.get("prediction"))
    components = _mapping(prediction.get("components"))
    mfa = _mapping(row.get("mfa"))
    return {
        "ctc_age": _finite(prediction.get("terminal_partial_ctc_log_likelihood_ratio"), -100.0),
        "ctc_censor": _finite(prediction.get("terminal_censor_log_odds"), -100.0),
        "suffix_non_recovery": _finite(components.get("suffix_non_recovery")),
        "low_last_phone_confidence": 1.0 - _finite(prediction.get("last_phone_confidence"), 1.0),
        "eou": _finite(prediction.get("eou_cutoff_score")),
        "whole_phone": _finite(prediction.get("whole_phone_cut_score")),
        "mfa_bad_local": -_finite(mfa.get("final_vs_local_loglike")),
        "mfa_bad_alignment": -_finite(mfa.get("final_vs_alignment_loglike")),
        "mfa_short_duration": -_finite(mfa.get("final_phone_duration_sec")),
        "mfa_short_run": _finite(mfa.get("max_running_short_interval")),
    }


def _percentile(value: float, null: Sequence[float]) -> float:
    if not null:
        msg = "verifier null evidence channel is empty"
        raise ValueError(msg)
    return bisect.bisect_left(null, value) / len(null)


def verifier_components(
    row: Mapping[str, Any],
    null_evidence: Mapping[str, Sequence[float]],
) -> dict[str, float | bool]:
    """Score one candidate; ``null_evidence`` lists must be sorted ascending (as stored in the calibration)."""
    raw = raw_evidence(row)
    evidence = {name: _percentile(raw[name], null_evidence[name]) for name in EVIDENCE_NAMES}
    ctc = sorted(
        (
            evidence["ctc_age"],
            evidence["ctc_censor"],
            evidence["suffix_non_recovery"],
            evidence["low_last_phone_confidence"],
            evidence["eou"],
        ),
        reverse=True,
    )
    mfa_values = sorted(
        (
            evidence["mfa_bad_local"],
            evidence["mfa_bad_alignment"],
            evidence["mfa_short_duration"],
            evidence["mfa_short_run"],
        ),
        reverse=True,
    )
    ctc_score = sum(ctc[:2]) / 2.0
    mfa_score = sum(mfa_values[:2]) / 2.0
    prediction = _mapping(row.get("prediction"))
    mfa = _mapping(row.get("mfa"))
    endpoint_quality = max(0.0, 1.0 - _finite(prediction.get("endpoint_bad_region_score")))
    valid_mfa_endpoint = bool(mfa.get("touches_eof")) and mfa.get("likelihood_status") == "AVAILABLE"
    partial_score = math.sqrt(max(ctc_score * mfa_score, 0.0)) * endpoint_quality if valid_mfa_endpoint else 0.0
    whole_score = math.sqrt(max(evidence["whole_phone"] * evidence["suffix_non_recovery"], 0.0)) * endpoint_quality
    return {
        "ctc_score": ctc_score,
        "mfa_score": mfa_score,
        "partial_score": partial_score,
        "whole_score": whole_score,
        "score": max(partial_score, whole_score),
        "valid_mfa_endpoint": valid_mfa_endpoint,
        "endpoint_quality": endpoint_quality,
    }
