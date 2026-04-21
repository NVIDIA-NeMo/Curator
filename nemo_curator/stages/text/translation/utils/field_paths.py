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

"""Helpers for wildcard dot-path reads and writes."""

from __future__ import annotations

import copy
import json
from typing import Any


def extract_nested_fields(record: dict[str, Any], path: str) -> list[str]:
    """Extract string values matching a wildcard dot-path."""
    parts = path.split(".")

    def _find(obj: Any, remaining: list[str], collected: list[str]) -> None:
        if not remaining:
            return

        key = remaining[0]
        rest = remaining[1:]

        if key == "*":
            if isinstance(obj, list):
                for item in obj:
                    if not rest:
                        if isinstance(item, str):
                            collected.append(item)
                    else:
                        _find(item, rest, collected)
            return

        if isinstance(obj, dict) and key in obj:
            if not rest:
                value = obj[key]
                if isinstance(value, str):
                    collected.append(value)
            else:
                _find(obj[key], rest, collected)

    found: list[str] = []
    _find(record, parts, found)
    return found


def set_nested_fields(record: dict[str, Any], path: str, values: list[str]) -> dict[str, Any]:
    """Write values back to a wildcard dot-path in traversal order."""
    result = copy.deepcopy(record)
    parts = path.split(".")
    value_index = [0]

    def _set(obj: Any, remaining: list[str]) -> None:
        if not remaining:
            return

        key = remaining[0]
        rest = remaining[1:]

        if key == "*":
            if isinstance(obj, list):
                for item in obj:
                    if not rest:
                        pass
                    else:
                        _set(item, rest)
            return

        if isinstance(obj, dict) and key in obj:
            if not rest:
                if isinstance(obj[key], str) and value_index[0] < len(values):
                    obj[key] = values[value_index[0]]
                    value_index[0] += 1
            else:
                _set(obj[key], rest)

    _set(result, parts)

    if value_index[0] != len(values):
        from loguru import logger

        logger.warning(
            f"set_nested_fields: expected to set {len(values)} values for path "
            f"'{path}', but only set {value_index[0]}"
        )

    return result


def is_wildcard_path(path: str) -> bool:
    """Return whether the path contains ``*``."""
    return "*" in path


def normalize_text_field(text_field: str | list[str]) -> list[str]:
    """Normalize ``text_field`` to a list of field paths."""
    if isinstance(text_field, str):
        return [text_field]
    return list(text_field)


def parse_structured_value(value: Any) -> Any | None:
    """Return parsed dict/list data when the value looks like JSON."""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (json.JSONDecodeError, TypeError):
            return None
    return None
