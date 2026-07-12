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


def _append_if_string(value: object, collected: list[str]) -> None:
    """Append ``value`` when it is a string."""
    if isinstance(value, str):
        collected.append(value)


def _find_nested(obj: object, remaining: list[str], collected: list[str]) -> None:
    """Recursive worker for wildcard dot-path extraction."""
    if not remaining:
        return

    key = remaining[0]
    rest = remaining[1:]

    if key == "*":
        if isinstance(obj, list):
            for item in obj:
                if rest:
                    _find_nested(item, rest, collected)
                else:
                    _append_if_string(item, collected)
        return

    if isinstance(obj, dict) and key in obj:
        if rest:
            _find_nested(obj[key], rest, collected)
        else:
            _append_if_string(obj[key], collected)


def extract_nested_fields(record: dict[str, object], path: str) -> list[str]:
    """Extract string values matching a wildcard dot-path."""
    found: list[str] = []
    _find_nested(record, path.split("."), found)
    return found


def _next_value(values: list[str], value_index: list[int]) -> str | None:
    """Return the next replacement value, or ``None`` when exhausted."""
    if value_index[0] >= len(values):
        return None
    value = values[value_index[0]]
    value_index[0] += 1
    return value


def _set_dict_value(container: dict[str, object], key: str, values: list[str], value_index: list[int]) -> None:
    """Set a terminal dict value when the current value is a string."""
    if isinstance(container[key], str) and (value := _next_value(values, value_index)) is not None:
        container[key] = value


def _set_list_values(items: list[object], remaining: list[str], values: list[str], value_index: list[int]) -> None:
    """Set matching list items for wildcard traversal."""
    for idx, item in enumerate(items):
        if remaining:
            _set_nested(item, remaining, values, value_index)
        elif isinstance(item, str) and (value := _next_value(values, value_index)) is not None:
            items[idx] = value


def _set_nested(obj: object, remaining: list[str], values: list[str], value_index: list[int]) -> None:
    """Recursive worker for wildcard dot-path writes."""
    if not remaining:
        return

    key = remaining[0]
    rest = remaining[1:]

    if key == "*":
        if isinstance(obj, list):
            _set_list_values(obj, rest, values, value_index)
        return

    if isinstance(obj, dict) and key in obj:
        if rest:
            _set_nested(obj[key], rest, values, value_index)
        else:
            _set_dict_value(obj, key, values, value_index)


def set_nested_fields(record: dict[str, object], path: str, values: list[str]) -> dict[str, object]:
    """Write values back to a wildcard dot-path in traversal order."""
    result = copy.deepcopy(record)
    value_index = [0]
    _set_nested(result, path.split("."), values, value_index)

    if value_index[0] != len(values):
        from loguru import logger

        logger.warning(
            f"set_nested_fields: expected to set {len(values)} values for path '{path}', but only set {value_index[0]}"
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


def parse_structured_value(value: object) -> object | None:
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
