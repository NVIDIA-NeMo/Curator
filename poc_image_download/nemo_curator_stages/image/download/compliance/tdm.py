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

"""TDMRep compliance checking.

Ported from cc-img-dl's compliance/tdm.py.
Changed from async httpx to sync requests.
"""

from __future__ import annotations

from fnmatch import fnmatch
from urllib.parse import urlparse

import requests

from .cache import ComplianceCache


def fetch_tdmrep(origin: str, session: requests.Session) -> dict | None:
    """Fetch TDMRep from .well-known/tdmrep.json."""
    url = f"{origin}/.well-known/tdmrep.json"
    try:
        resp = session.get(url, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                return data
    except (requests.RequestException, requests.Timeout, ValueError, KeyError):
        pass
    return None


def tdm_allowed(tdmrep: dict | None, url: str, policy: str = "conservative") -> bool:
    """Check if URL is allowed per TDMRep rules (first-match-wins glob matching)."""
    if tdmrep is None:
        return policy == "permissive"

    policy_rules = tdmrep.get("policy", [])
    if not policy_rules:
        return True

    path = urlparse(url).path or "/"
    for rule in policy_rules:
        location = rule.get("location", "*")
        permission = rule.get("permission", "allowed")
        if fnmatch(path, location):
            return permission == "allowed"

    return True


class TDMChecker:
    """Per-origin TDMRep checker with caching."""

    def __init__(self, cache: ComplianceCache, ttl: int, policy: str):
        self._cache = cache
        self._ttl = ttl
        self._policy = policy

    def allowed(self, url: str, session: requests.Session) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        cache_key = f"tdm:{origin}"

        cached = self._cache.get(cache_key)
        if cached is None:
            tdmrep = fetch_tdmrep(origin, session)
            self._cache.set(cache_key, tdmrep or {}, self._ttl)
            cached = tdmrep or {}

        tdmrep_data = cached if cached else None
        return tdm_allowed(tdmrep_data, url, self._policy)
