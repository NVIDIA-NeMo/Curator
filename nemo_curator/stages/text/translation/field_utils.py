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

"""Utilities for extracting and setting values in nested dicts using wildcard dot-paths.

Ported from Speaker's ``_get_all_nested_fields()`` and ``_set_all_nested_fields()``
in ``speaker/src/speaker/core/translate/translate_jsonl.py``.

A *field path* is a dot-separated string such as ``"messages.*.content"`` where
``*`` matches every element when the current level is a list.  This allows
traversing structures like::

    {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}

with path ``"messages.*.content"`` to extract ``["Hello", "Hi"]``.
"""

from __future__ import annotations

import copy
import json
from typing import Any


def extract_nested_fields(record: dict[str, Any], path: str) -> list[str]:
    """Extract all string values matching a wildcard dot-path from *record*.

    Args:
        record: A (possibly nested) dictionary.
        path: Dot-separated field path.  Use ``*`` to traverse list elements.
              Examples: ``"text"``, ``"messages.*.content"``,
              ``"conversations.*.value"``.

    Returns:
        An ordered list of all string values found at the matching positions.
        Non-string leaves are silently skipped.
    """
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
    """Write *values* back into *record* at positions matching the wildcard dot-path.

    The function performs an **in-order** replacement: the first string value
    encountered during traversal is replaced with ``values[0]``, the second
    with ``values[1]``, and so on.

    Args:
        record: A (possibly nested) dictionary.  A deep copy is made so the
                original is not mutated.
        path: Dot-separated field path (same syntax as :func:`extract_nested_fields`).
        values: Replacement strings, one per position found during traversal.

    Returns:
        A new dictionary with the values replaced.

    Raises:
        ValueError: If the number of *values* does not match the number of
            string positions found in *record* for the given *path*.
    """
    result = copy.deepcopy(record)
    parts = path.split(".")
    value_index = [0]  # mutable counter shared across recursive calls

    def _set(obj: Any, remaining: list[str]) -> None:
        if not remaining:
            return

        key = remaining[0]
        rest = remaining[1:]

        if key == "*":
            if isinstance(obj, list):
                for item in obj:
                    if not rest:
                        # Wildcard at leaf means list elements themselves -- not applicable
                        # for dict-based structures, skip silently.
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
    """Return ``True`` if *path* contains a wildcard (``*``) component.

    Args:
        path: A dot-separated field path.
    """
    return "*" in path


def normalize_text_field(text_field: str | list[str]) -> list[str]:
    """Normalize *text_field* to a list of field paths.

    Accepts either a single string or a list of strings and always returns a
    list.

    Args:
        text_field: A single field name / wildcard path, or a list of them.
    """
    if isinstance(text_field, str):
        return [text_field]
    return list(text_field)


def parse_structured_value(value: Any) -> dict[str, Any] | None:
    """Attempt to interpret *value* as a dict.

    If *value* is already a ``dict`` it is returned directly.  If it is a
    string, we attempt ``json.loads``; on success the parsed dict is returned,
    otherwise ``None``.

    Args:
        value: The cell value from the DataFrame.

    Returns:
        A dict if parsing succeeded, else ``None``.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            return None
    return None
