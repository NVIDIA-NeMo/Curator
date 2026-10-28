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

"""Paired-record loading, artifact I/O, vector helpers, and NeMo Curator task helpers."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from omnifuse_tutorial.config import DataPoolConfig

# Mapping columns the loader consumes; every other column is kept as record metadata.
MAPPING_KEYS = {"id", "data_path", "annotation_path", "annotation"}


def make_batch(dataset_name: str, records: list[dict[str, Any]], metadata: dict[str, Any]) -> DocumentBatch:
    return DocumentBatch(dataset_name=dataset_name, data=pd.DataFrame.from_records(records), _metadata=metadata)


def with_records(task: DocumentBatch, records: list[dict[str, Any]], metadata: dict[str, Any]) -> DocumentBatch:
    task.data, task._metadata = pd.DataFrame.from_records(records), metadata
    return task


def task_records(task: DocumentBatch) -> list[dict[str, Any]]:
    return task.data.to_dict(orient="records")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def write_npy(path: Path, rows: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(rows, dtype=np.float64))
    return path


def expert_file(directory: Path, expert: str, kind: str) -> Path:
    """Per-expert array; `interleaved` rows alternate raw and annotation embeddings."""
    return directory / f"{expert.replace('-', '_')}_{kind}.npy"


def text_or_none(value: Any) -> str | None:
    return (value.strip() or None) if isinstance(value, str) else None


def raw_text(record: dict[str, Any]) -> str | None:
    return text_or_none(record.get("sns_raw_text")) or text_or_none(record.get("raw_text"))


def annotation_text(record: dict[str, Any]) -> str:
    return text_or_none(record.get("sns_annotation")) or text_or_none(record.get("annotation")) or ""


def unit(vector: Any, dim: int, dtype: type = np.float32) -> np.ndarray:
    """Zero-pad or truncate to `dim`, then L2-normalize (zero vectors stay zero)."""
    values = np.asarray(vector, dtype=dtype).reshape(-1)[:dim]
    values = np.pad(values, (0, dim - values.size))
    norm = np.linalg.norm(values)
    return values / norm if norm > 0 else values


def cosine(left: Any, right: Any) -> float:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    norm = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / norm) if norm else 0.0


def fingerprint(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            digest = hashlib.sha256(handle.read(65536)).hexdigest()[:24]
        return f"file:{path.suffix.lower()} size:{path.stat().st_size} sha256:{digest}"
    except OSError:
        return f"file:{path.suffix.lower()} unreadable"


def load_records(pools: list[DataPoolConfig]) -> list[dict[str, Any]]:
    """Read every pool's JSONL mapping file into normalized pair records."""
    records = []
    for pool in pools:
        mapping = pool.root_dir / pool.mapping_file
        if not mapping.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping}")
        rows = read_jsonl(mapping)
        pool_records = [record for index, row in enumerate(rows) if (record := _record(pool, row, index))]
        if pool.shuffle:
            random.Random(0).shuffle(pool_records)  # noqa: S311 - deterministic tutorial sampling, not security.
        records += pool_records if pool.n_samples is None else pool_records[: pool.n_samples]
    return records


def _record(pool: DataPoolConfig, row: dict[str, Any], index: int) -> dict[str, Any] | None:
    raw_rel, ann_rel, annotation = row.get("data_path"), row.get("annotation_path"), row.get("annotation") or None
    if not raw_rel:
        raise ValueError(f"Mapping row missing data_path in pool {pool.name}: {row}")
    raw_path = _resolve(pool.root_dir, raw_rel)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found for pool {pool.name}: {raw_path}")
    if pool.max_file_size_mb is not None and raw_path.stat().st_size > pool.max_file_size_mb * 1024 * 1024:
        return None
    annotation_path = _resolve(pool.root_dir, ann_rel) if ann_rel else None
    if annotation is None and annotation_path:
        annotation = annotation_path.read_text(encoding="utf-8").strip()
    if annotation is None:
        raise ValueError(f"Mapping row has no annotation text/path in pool {pool.name}: {row}")
    stable_id = "\x1f".join((pool.name, str(raw_rel), str(ann_rel or annotation)))
    return {
        "pair_id": str(row.get("id") or hashlib.sha256(stable_id.encode("utf-8")).hexdigest()[:16]),
        "pool": pool.name,
        "pool_index": index,
        "modality": pool.modality,
        "raw_path": str(raw_path),
        "annotation_path": str(annotation_path) if annotation_path else None,
        "annotation": str(annotation).strip(),
        "raw_text": raw_path.read_text(encoding="utf-8").strip() if pool.modality == "text" else None,
        "metadata": {key: value for key, value in row.items() if key not in MAPPING_KEYS},
    }


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()
