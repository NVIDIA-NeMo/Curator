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

"""Small helpers shared by LLM judge stages."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd


def stable_json_dumps(value: object) -> str:
    """Serialize a value as a stable JSON string for dataframe/Arrow columns."""
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def to_plain_dict(value: object) -> dict[str, Any] | None:
    """Convert dataclasses or mapping-like values to a plain dict."""
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return None


def is_missing_value(value: object) -> bool:
    """Return True for scalar missing values without treating containers as missing."""
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, set)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from an LLM response."""
    text = (raw or "").strip()
    if not text:
        msg = "empty response"
        raise ValueError(msg)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    last_error: json.JSONDecodeError | None = None
    starts = _json_object_starts(text)
    for start in starts:
        end = _find_json_object_end(text, start)
        if end < 0:
            continue
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(parsed, dict):
            msg = "parsed JSON is not an object"
            raise TypeError(msg)
        return parsed

    if last_error is not None:
        msg = "response does not contain a valid JSON object"
        raise ValueError(msg) from last_error

    if not starts:
        msg = "response does not contain a JSON object"
        raise ValueError(msg)
    msg = "unterminated JSON object"
    raise ValueError(msg)


def _json_object_starts(text: str) -> list[int]:
    """Return candidate object starts, ignoring braces inside JSON strings."""
    starts = []
    in_string = False
    escape = False
    for pos, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            starts.append(pos)
    return starts


def _find_json_object_end(text: str, start: int) -> int:
    """Return the balanced object end index from ``start``, or ``-1``."""
    depth = 0
    in_string = False
    escape = False
    for pos in range(start, len(text)):
        char = text[pos]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return pos
    return -1


def normalize_recommendation(value: object) -> list[str]:
    """Normalize recommendation fields to a stable list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    stripped = str(value).strip()
    return [stripped] if stripped else []


def coerce_float(value: object, key: str) -> float:
    """Coerce a score to float and raise a helpful error when it cannot be parsed."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        msg = f"dimension score {key!r} is not numeric"
        raise ValueError(msg) from exc
