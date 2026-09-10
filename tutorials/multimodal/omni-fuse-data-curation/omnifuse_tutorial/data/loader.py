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

"""Paired data-pool loader."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from omnifuse_tutorial.config.models import DataPoolConfig


def load_pool_records(pool: DataPoolConfig) -> list[dict[str, Any]]:
    mapping_path = pool.root_dir / pool.mapping_file
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    if mapping_path.suffix.lower() == ".csv":
        mapping_rows = _read_csv(mapping_path)
    else:
        mapping_rows = _read_jsonl(mapping_path)

    records = [_normalize_mapping_row(pool, row, index) for index, row in enumerate(mapping_rows)]
    records = [record for record in records if record is not None]
    if pool.shuffle:
        rng = random.Random(0)  # noqa: S311 - deterministic tutorial sampling, not security.
        rng.shuffle(records)
    if pool.n_samples is not None:
        records = records[: pool.n_samples]
    return records


def load_all_pools(pools: list[DataPoolConfig]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pool in pools:
        records.extend(load_pool_records(pool))
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _normalize_mapping_row(pool: DataPoolConfig, row: dict[str, Any], index: int) -> dict[str, Any] | None:
    raw_rel = row.get("data_path") or row.get("raw_path") or row.get("path")
    ann_rel = row.get("annotation_path") or row.get("caption_path") or row.get("label_path")
    annotation_text = row.get("annotation") or row.get("caption") or row.get("text")
    if not raw_rel:
        raise ValueError(f"Mapping row missing data_path/raw_path/path in pool {pool.name}: {row}")
    raw_path = _resolve(pool.root_dir, str(raw_rel))
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found for pool {pool.name}: {raw_path}")

    if pool.max_file_size_mb is not None and raw_path.exists():
        max_bytes = pool.max_file_size_mb * 1024 * 1024
        if raw_path.stat().st_size > max_bytes:
            return None

    annotation_path = _resolve(pool.root_dir, str(ann_rel)) if ann_rel else None
    if annotation_text is None and annotation_path:
        annotation_text = annotation_path.read_text(encoding="utf-8").strip()
    if annotation_text is None:
        raise ValueError(f"Mapping row has no annotation text/path in pool {pool.name}: {row}")

    raw_text = None
    if pool.modality == "text" and raw_path.exists():
        raw_text = raw_path.read_text(encoding="utf-8").strip()

    record_id = row.get("id") or _stable_id(pool.name, str(raw_rel), str(ann_rel or annotation_text))
    metadata = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "id",
            "data_path",
            "raw_path",
            "path",
            "annotation_path",
            "caption_path",
            "label_path",
            "annotation",
            "caption",
            "text",
        }
    }
    return {
        "pair_id": str(record_id),
        "pool": pool.name,
        "pool_index": index,
        "modality": pool.modality,
        "raw_path": str(raw_path),
        "annotation_path": str(annotation_path) if annotation_path else None,
        "annotation": str(annotation_text).strip(),
        "raw_text": raw_text,
        "metadata": metadata,
    }


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]
