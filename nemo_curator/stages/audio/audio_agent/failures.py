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

"""Failure classifier — map a raw error to an actionable FailureClass.

Deterministic signature matching over the versioned ``knowledge/failures.yaml``
taxonomy (symptom -> cause -> fix). Feeds both the Critic (structured directive
to re-plan) and the user-facing run report (triage).
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_FAILURES_PATH = os.path.join(_HERE, "knowledge", "failures.yaml")


@lru_cache(maxsize=1)
def _taxonomy() -> list[dict[str, Any]]:
    if not os.path.isfile(_FAILURES_PATH):
        return []
    try:
        import yaml

        with open(_FAILURES_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:  # noqa: BLE001
        return []
    if isinstance(data, dict):
        data = data.get("failures", [])
    return [d for d in (data or []) if isinstance(d, dict)]


def classify(error_text: str) -> dict[str, Any]:
    """Return the best-matching FailureClass for ``error_text`` (or an ``unknown``).

    Each ``symptom_signature`` is matched as a case-insensitive regex; a signature
    that is not a valid regex falls back to a case-insensitive substring test. We do
    NOT *additionally* substring-match a valid regex -- doing so would defeat the
    ``\\b`` anchors in the taxonomy that keep short tokens (e.g. ``\\bray\\b``) from
    firing on unrelated words like "array". Entries are tried in file order, which the
    taxonomy keeps specific-before-generic so the first match is the best match.
    """
    text = error_text or ""
    for entry in _taxonomy():
        for sig in entry.get("symptom_signature", []) or []:
            sig_str = str(sig)
            try:
                matched = re.search(sig_str, text, re.IGNORECASE) is not None
            except re.error:  # signature is not a valid regex -> literal substring
                matched = sig_str.lower() in text.lower()
            if matched:
                return {
                    "code": entry.get("code", "unknown"),
                    "likely_cause": entry.get("likely_cause", ""),
                    "layer": entry.get("layer", "unknown"),
                    "auto_fix": entry.get("auto_fix"),
                    "user_guidance": entry.get("user_guidance", ""),
                    "evidence": (error_text or "")[:500],
                }
    return {
        "code": "unknown_failure",
        "likely_cause": "unrecognized error signature",
        "layer": "unknown",
        "auto_fix": None,
        "user_guidance": "inspect the evidence; re-run with verbose logging",
        "evidence": (error_text or "")[:500],
    }
