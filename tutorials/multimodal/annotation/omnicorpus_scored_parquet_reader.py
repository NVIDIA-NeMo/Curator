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

"""Read Parquet written by :mod:`omnicorpus_annotation_pipeline`.

:class:`OmnicorpusScoredParquetWriterStage` JSON-serializes CLIP per-text-position
score dicts so PyArrow can write the table. This module loads those files and
optionally restores CLIP columns to ``dict[int, float]`` cells (matching in-memory
score filter output).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs
from pyarrow.fs import FSSpecHandler, PyFileSystem

from nemo_curator.stages.interleaved.utils import resolve_storage_options
from nemo_curator.tasks import InterleavedBatch

CLIP_SCORES_SUFFIX = "_clip_scores_by_text_position"


def _is_clip_scores_column(name: str) -> bool:
    """Heuristic: columns produced by :class:`~InterleavedCLIPScoreFilterStage` for Parquet."""
    if name.endswith(CLIP_SCORES_SUFFIX):
        return True
    # Legacy name ``{stage.name}_clip_scores`` (no ``_by_text_position`` suffix).
    return name.endswith("clip_scores") and "clip_score_filter" in name


def _parse_clip_json_cell(v: Any) -> Any:
    """Turn a JSON string (writer output) or dict into ``dict[int, float]``; pass through nulls."""
    if v is None or v is pd.NA:
        return v
    if isinstance(v, dict):
        return {int(k): float(val) for k, val in v.items()}
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return pd.NA
        parsed = json.loads(s)
        if not isinstance(parsed, dict):
            return v
        return {int(k): float(val) for k, val in parsed.items()}
    if pd.api.types.is_scalar(v) and pd.isna(v):
        return pd.NA
    return v


def _restore_clip_columns(df: pd.DataFrame) -> pd.DataFrame:
    clip_cols = [c for c in df.columns if _is_clip_scores_column(c)]
    if not clip_cols:
        return df
    out = df.copy()
    for col in clip_cols:
        out[col] = out[col].map(_parse_clip_json_cell)
    return out


def read_omnicorpus_scored_parquet(
    paths: str | Path | list[str | Path],
    *,
    task_id: str = "omnicorpus_scored_read",
    dataset_name: str = "omnicorpus_scored",
    read_kwargs: dict[str, Any] | None = None,
    restore_clip_score_dicts: bool = True,
) -> InterleavedBatch:
    """Load one or more scored Parquet files into an :class:`~InterleavedBatch`.

    Args:
        paths: Single Parquet path/URI or a list of files (no directory glob).
        task_id: ``InterleavedBatch.task_id`` for the merged result.
        dataset_name: ``InterleavedBatch.dataset_name``.
        read_kwargs: Optional ``{"storage_options": {...}}`` for fsspec (cloud URIs).
        restore_clip_score_dicts: If True, parse CLIP score columns from JSON strings
            back to ``dict[int, float]`` per row.

    Returns:
        Batch with ``data`` as a :class:`pandas.DataFrame` (row order preserved).
    """
    path_list = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not path_list:
        msg = "read_omnicorpus_scored_parquet requires at least one path"
        raise ValueError(msg)

    storage = resolve_storage_options(io_kwargs=read_kwargs or {})
    frames: list[pd.DataFrame] = []
    source_files: list[str] = []

    for p in path_list:
        ps = str(p)
        source_files.append(ps)
        fs, _ = url_to_fs(ps, **(storage or {}))
        pa_fs = PyFileSystem(FSSpecHandler(fs))
        table = pq.read_table(ps, filesystem=pa_fs)
        frames.append(table.to_pandas())

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if restore_clip_score_dicts:
        df = _restore_clip_columns(df)

    metadata: dict[str, Any] = {"source_files": source_files}
    if storage:
        metadata["source_storage_options"] = storage

    return InterleavedBatch(
        task_id=task_id,
        dataset_name=dataset_name,
        data=df,
        _metadata=metadata,
    )


def read_omnicorpus_scored_parquet_pyarrow(
    paths: str | Path | list[str | Path],
    *,
    task_id: str = "omnicorpus_scored_read",
    dataset_name: str = "omnicorpus_scored",
    read_kwargs: dict[str, Any] | None = None,
    restore_clip_score_dicts: bool = True,
) -> InterleavedBatch:
    """Same as :func:`read_omnicorpus_scored_parquet` but ``task.data`` is a :class:`pyarrow.Table`."""
    batch = read_omnicorpus_scored_parquet(
        paths,
        task_id=task_id,
        dataset_name=dataset_name,
        read_kwargs=read_kwargs,
        restore_clip_score_dicts=restore_clip_score_dicts,
    )
    df = batch.to_pandas()
    table = pa.Table.from_pandas(df, preserve_index=False)
    return InterleavedBatch(
        task_id=batch.task_id,
        dataset_name=batch.dataset_name,
        data=table,
        _metadata=batch._metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Omnicorpus scored Parquet (CLIP JSON restored by default).")
    parser.add_argument("parquet", nargs="+", help="Parquet file path(s) or URI(s)")
    parser.add_argument("--dataset-name", default="omnicorpus_scored")
    parser.add_argument("--no-restore-clip", action="store_true", help="Keep CLIP columns as JSON strings")
    parser.add_argument(
        "--storage-options-json",
        default=None,
        help="JSON object passed as fsspec storage_options for remote paths",
    )
    args = parser.parse_args()
    read_kwargs: dict[str, Any] = {}
    if args.storage_options_json:
        read_kwargs["storage_options"] = json.loads(args.storage_options_json)
    batch = read_omnicorpus_scored_parquet(
        args.parquet,
        dataset_name=args.dataset_name,
        read_kwargs=read_kwargs or None,
        restore_clip_score_dicts=not args.no_restore_clip,
    )
    df = batch.to_pandas()
    print(f"rows={len(df)} cols={list(df.columns)}")
    clip_cols = [c for c in df.columns if _is_clip_scores_column(c)]
    if clip_cols:
        c0 = clip_cols[0]
        sample = df.loc[df["modality"] == "image", c0].dropna().head(1)
        if len(sample):
            v = sample.iloc[0]
            print(f"sample {c0!r} type={type(v).__name__}")
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
