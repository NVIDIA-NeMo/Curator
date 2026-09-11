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

"""Measured-resource store — carries a ``smoke``'s measurements to the next ``run``.

``smoke`` observes what each stage really used and returns a ``calibration`` block, but the
planner only ever saw it when the caller handed it back (``run --calibration``). Nothing
could enforce that: a run with no calibration is perfectly legitimate — a CPU smoke measures
no VRAM — so a forgotten flag is indistinguishable from having nothing to apply, and neither
warns. The planner then falls back to its conservative per-stage floor of 1 GB of host RAM,
which is the number that decides streaming vs batch, so a forgotten flag can put every stage
on the machine at once and OOM a run that a stored measurement would have serialized.

The smoke token cannot close that gap. It is an HMAC over the config hash — proof that a
smoke ran for this exact recipe, not an address where its measurements live. So the
measurements get a home of their own, keyed by the same config hash the token signs: edit the
recipe and the key changes with it, and a measurement carried to a different machine is
dropped by the planner's existing ``machine_fingerprint`` check rather than by anything here.

Files live under ``<run_store.runs_dir()>/calibration/``, so they honour
``AUDIO_AGENT_RUNS_DIR`` and are discarded by the same gesture that discards run records.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from nemo_curator.audio_agent.run_store import (
    _ensure_private_dir,
    _write_private_json,
    runs_dir,
)

# A config hash is a hex digest, but it arrives from a caller-supplied recipe. Anything that
# is not plainly filename-shaped is refused rather than sanitized: quietly rewriting a key
# would store measurements under a name no lookup could reproduce.
_SAFE_KEY = re.compile(r"\A[A-Za-z0-9_-]{1,128}\Z")


def store_dir() -> str:
    """The directory measured calibrations live in."""
    return os.path.join(runs_dir(), "calibration")


def path_for(config_hash: str | None) -> str | None:
    """The file a calibration for ``config_hash`` would occupy, or None if the key is unusable."""
    if not config_hash or not _SAFE_KEY.match(str(config_hash)):
        return None
    return os.path.join(store_dir(), f"{config_hash}.json")


def save(
    config_hash: str | None,
    calibration: dict[str, Any] | None,
    *,
    machine_fingerprint: str | None = None,
) -> str | None:
    """Persist measured per-stage resources for ``config_hash``; returns the path or None.

    Wraps the entries in the same envelope :func:`verbs.calibrate` returns, so a stored
    calibration and a hand-passed one are the same shape to the planner. A filesystem that
    refuses the write returns None: storing telemetry must never fail the smoke that produced
    it.
    """
    path = path_for(config_hash)
    if path is None or not calibration:
        return None
    payload = {
        "config_hash": str(config_hash),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "calibration": calibration,
    }
    if machine_fingerprint:
        payload["machine_fingerprint"] = machine_fingerprint
    try:
        _ensure_private_dir(runs_dir())
        _ensure_private_dir(store_dir())
        _write_private_json(path, payload)
    except OSError:
        return None
    return path


def load(config_hash: str | None) -> dict[str, Any] | None:
    """The stored calibration envelope for ``config_hash``, or None if there is none to apply.

    Returns None for a corrupt or hand-edited file for the same reason ``run_store.load``
    does: a plan built from conservative card facts is correct, just less informed, whereas a
    half-parsed measurement is neither.
    """
    path = path_for(config_hash)
    if path is None or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:  # noqa: BLE001 - a corrupt record must not break planning
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("calibration"), dict):
        return None
    return payload if payload["calibration"] else None
