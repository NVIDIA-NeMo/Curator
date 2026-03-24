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

"""Robots.txt compliance checking.

Ported from cc-img-dl's compliance/robots.py.
Changed from async httpx to sync requests (Curator dependency).
"""

from __future__ import annotations

from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from loguru import logger

from .cache import ComplianceCache


def fetch_robots(origin: str, user_agent: str, session: requests.Session) -> str | None:
    """Fetch robots.txt from origin. Returns content or None if unreachable."""
    url = f"{origin}/robots.txt"
    try:
        resp = session.get(url, timeout=10.0, headers={"User-Agent": user_agent})
        if resp.status_code == 200:
            return resp.text
    except (requests.RequestException, requests.Timeout):
        pass
    return None


def can_fetch(robots_text: str | None, user_agent: str, url: str, policy: str = "conservative") -> bool:
    """Check if URL is allowed per robots.txt."""
    if robots_text is None:
        return policy == "permissive"
    parser = RobotFileParser()
    parser.parse(robots_text.splitlines())
    return parser.can_fetch(user_agent, url)


class RobotsChecker:
    """Per-origin robots.txt checker with caching."""

    def __init__(self, cache: ComplianceCache, user_agent: str, ttl: int, policy: str):
        self._cache = cache
        self._user_agent = user_agent
        self._ttl = ttl
        self._policy = policy

    def allowed(self, url: str, session: requests.Session) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        cache_key = f"robots:{origin}"

        robots_text = self._cache.get(cache_key)
        if robots_text is None:
            robots_text = fetch_robots(origin, self._user_agent, session)
            self._cache.set(cache_key, robots_text or "", self._ttl)

        if robots_text == "":
            robots_text = None

        return can_fetch(robots_text, self._user_agent, url, self._policy)
