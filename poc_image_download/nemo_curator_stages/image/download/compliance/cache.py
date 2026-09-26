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

"""Per-worker compliance cache with TTL expiry.

Ported directly from cc-img-dl's compliance/cache.py.
In Curator, each worker gets its own InMemoryCache instance via setup().
For cross-worker cache sharing at scale, swap in a Redis/DynamoDB-backed
implementation that conforms to the same get/set interface.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Protocol


class ComplianceCache(Protocol):
    """Cache protocol for compliance data (robots.txt, TDMRep)."""

    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: int = 86400) -> None: ...


class InMemoryCache:
    """Thread-safe in-memory cache with TTL expiry."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: int = 86400) -> None:
        with self._lock:
            self._store[key] = (value, time.monotonic() + ttl)
