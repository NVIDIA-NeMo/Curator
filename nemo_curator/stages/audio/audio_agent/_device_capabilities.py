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

"""Resolve configured device capabilities from Audio Agent cards."""

from __future__ import annotations

from typing import Any


def supported_devices(stage: Any, card: dict[str, Any] | None = None) -> frozenset[str] | None:  # noqa: ANN401
    """Return card-declared support for the configured adapter.

    Cards without an adapter map return ``None`` and retain their existing
    stage-level ``resource.gpu_optional`` behavior. Once a card declares an
    adapter map, an unknown or malformed adapter entry fails closed as an empty
    set rather than inheriting a capability belonging to another adapter.
    """
    capabilities = (card or {}).get("adapter_capabilities")
    if not isinstance(capabilities, dict):
        return None
    adapter_target = getattr(stage, "adapter_target", None)
    if not isinstance(adapter_target, str) or not adapter_target:
        return frozenset()
    entry = capabilities.get(adapter_target)
    if not isinstance(entry, dict):
        return frozenset()
    value = entry.get("supported_devices")
    try:
        devices = frozenset(str(device).strip().lower() for device in value)
    except TypeError:
        return frozenset()
    return devices if devices <= {"cpu", "cuda"} and devices else frozenset()


def gpu_optional(stage: Any, card: dict[str, Any] | None = None) -> bool | None:  # noqa: ANN401
    """Resolve CPU fallback support, preferring adapter-specific card facts."""
    devices = supported_devices(stage, card)
    if devices is not None:
        return "cpu" in devices
    raw = ((card or {}).get("resource") or {}).get("gpu_optional")
    return raw if isinstance(raw, bool) else None
