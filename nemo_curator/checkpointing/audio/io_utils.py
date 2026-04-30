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

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def normalize_for_json(value: Any) -> Any:  # noqa: ANN401
    if value is None or isinstance(value, (str, int, float, bool)):
        normalized = value
    elif isinstance(value, dict):
        normalized = {str(key): normalize_for_json(item) for key, item in sorted(value.items())}
    elif isinstance(value, (list, tuple)):
        normalized = [normalize_for_json(item) for item in value]
    elif isinstance(value, set):
        normalized_items = [normalize_for_json(item) for item in value]
        normalized = sorted(normalized_items, key=repr)
    elif hasattr(value, "to_dict"):
        normalized = normalize_for_json(value.to_dict())
    elif hasattr(value, "__dict__"):
        normalized = normalize_for_json(vars(value))
    else:
        normalized = repr(value)
    return normalized


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temp:
        temp.write(text)
        temp.flush()
        temp_path = Path(temp.name)
    temp_path.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True))


def write_jsonl_atomic(path: Path, payloads: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(payload, sort_keys=True) for payload in payloads)
    if text:
        text += "\n"
    write_text_atomic(path, text)
