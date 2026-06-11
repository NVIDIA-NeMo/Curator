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

"""Small stdlib I/O helpers used by pipeline stages."""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: object) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_npy(path: str | Path, rows: list[list[float]]) -> Path:
    """Write a 2D float64 NumPy .npy file without importing numpy."""

    path = Path(path)
    ensure_dir(path.parent)
    n_rows = len(rows)
    n_cols = len(rows[0]) if rows else 0
    for row in rows:
        if len(row) != n_cols:
            raise ValueError("All rows must have the same length")

    header = {
        "descr": "<f8",
        "fortran_order": False,
        "shape": (n_rows, n_cols),
    }
    header_text = repr(header)
    header_bytes = header_text.encode("latin1")
    preamble_len = 10
    padding = 16 - ((preamble_len + len(header_bytes) + 1) % 16)
    full_header = header_bytes + b" " * padding + b"\n"

    with path.open("wb") as handle:
        handle.write(b"\x93NUMPY")
        handle.write(bytes([1, 0]))
        handle.write(struct.pack("<H", len(full_header)))
        handle.write(full_header)
        for row in rows:
            for value in row:
                handle.write(struct.pack("<d", float(value)))
    return path


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have matching dimensions")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def stable_relpath(path: str | Path, start: str | Path | None = None) -> str:
    path = Path(path)
    try:
        return os.path.relpath(path, start=start or Path.cwd())
    except ValueError:
        return str(path)
