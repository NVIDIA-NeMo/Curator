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

import io
import tarfile
from typing import NamedTuple

import fsspec
import pandas as pd
from fsspec.core import url_to_fs
from loguru import logger
from PIL import Image as _Image

from nemo_curator.tasks import InterleavedBatch

from .validation_utils import resolve_storage_options

_TAR_EXTENSIONS = (".tar", ".tar.gz", ".tgz")
# Exact columns added by InterleavedBatch.with_parsed_source_ref_columns(prefix="_src_").
# Drop only these — not everything starting with "_src_" — to preserve user passthrough columns.
_SRC_PARSE_COLS = ("_src_path", "_src_member", "_src_byte_offset", "_src_byte_size", "_src_frame_index")


class _ClassifiedRows(NamedTuple):
    tar_extract: dict[str, list[tuple[int, str, int | None]]]
    range_read: dict[str, list[tuple[int, str, int, int, int | None]]]
    direct_read: dict[str, list[int]]
    missing: list[int]


def _get_frame_index(df: pd.DataFrame, idx: int) -> int | None:
    if "_src_frame_index" not in df.columns:
        return None
    val = df.loc[idx, "_src_frame_index"]
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return int(val)


def _classify_rows(
    df: pd.DataFrame,
    image_mask: pd.Series,
) -> _ClassifiedRows:
    """Partition pending image rows into three I/O strategy groups.

    - tar_extract: has member name but no byte_offset (must open tar and extractfile)
    - range_read: has member + byte_offset + byte_size (can use fs.cat_ranges)
    - direct_read: no member (path is the file itself)
    - missing: path is None/NaN
    """
    tar_extract: dict[str, list[tuple[int, str, int | None]]] = {}
    range_read: dict[str, list[tuple[int, str, int, int, int | None]]] = {}
    direct_read: dict[str, list[int]] = {}
    missing: list[int] = []

    for idx in df[image_mask].index:
        path = df.loc[idx, "_src_path"]
        if path is None or (isinstance(path, float) and pd.isna(path)) or path == "":
            missing.append(idx)
            continue

        path_str = str(path)
        raw_member = df.loc[idx, "_src_member"]
        has_member = raw_member not in (None, "") and pd.notna(raw_member)

        if not has_member:
            direct_read.setdefault(path_str, []).append(idx)
            continue

        member_str = str(raw_member)
        frame_idx = _get_frame_index(df, idx)
        raw_offset = df.loc[idx, "_src_byte_offset"]
        raw_size = df.loc[idx, "_src_byte_size"]
        has_range = raw_offset is not None and raw_size is not None and pd.notna(raw_offset) and pd.notna(raw_size)

        if has_range and int(raw_size) > 0:
            range_read.setdefault(path_str, []).append((idx, member_str, int(raw_offset), int(raw_size), frame_idx))
        else:
            tar_extract.setdefault(path_str, []).append((idx, member_str, frame_idx))

    return _ClassifiedRows(tar_extract=tar_extract, range_read=range_read, direct_read=direct_read, missing=missing)


def _extract_tiff_frame(tiff_bytes: bytes, frame_index: int) -> bytes | None:
    """Extract a single frame from a multi-frame TIFF, returning it as a single-frame TIFF.

    Returns the raw bytes unchanged if the data is not a TIFF.
    """
    try:
        with _Image.open(io.BytesIO(tiff_bytes)) as img:
            if img.format != "TIFF":
                return tiff_bytes
            if frame_index >= getattr(img, "n_frames", 1):
                return None
            img.seek(frame_index)
            # Copy forces a full pixel decode before the context manager closes the source.
            # We intentionally do NOT reuse the source compression (e.g. JPEG-in-TIFF) because
            # lossy codecs (JPEG) do not support alpha channels and corrupt RGBA frames.
            frame = img.copy()
        buf = io.BytesIO()
        frame.save(buf, format="TIFF")
        return buf.getvalue()
    except (OSError, SyntaxError, ValueError):
        return None


def _fill_tar_extract_rows(
    groups: dict[str, list[tuple[int, str, int | None]]],
    storage_options: dict[str, object],
    binary_values: list[object],
    error_values: list[str | None],
) -> None:
    """Open each tar once and extract all needed members sequentially."""
    for path, keyed_rows in groups.items():
        key_cache: dict[str, bytes | None] = {}
        try:
            with fsspec.open(path, mode="rb", **storage_options) as fobj, tarfile.open(fileobj=fobj, mode="r:*") as tf:
                for idx, member, frame_idx in keyed_rows:
                    if member not in key_cache:
                        try:
                            extracted = tf.extractfile(member)
                        except KeyError:
                            extracted = None
                        key_cache[member] = extracted.read() if extracted is not None else None

                    payload = key_cache[member]
                    if payload is None:
                        error_values[idx] = f"missing member '{member}'"
                        continue

                    if frame_idx is not None:
                        payload = _extract_tiff_frame(payload, frame_idx)
                        if payload is None:
                            error_values[idx] = f"failed to extract frame {frame_idx} from '{member}'"
                            continue

                    binary_values[idx] = payload
                    error_values[idx] = None
        except (OSError, tarfile.TarError):
            for idx, *_ in keyed_rows:
                error_values[idx] = "failed to read path"


def _scatter_range_blobs(
    blobs: list[object],
    range_keys: list[tuple[str, int, int]],
    unique_ranges: dict[tuple[str, int, int], list[tuple[int, str, int | None]]],
    binary_values: list[object],
    error_values: list[str | None],
) -> None:
    """Distribute deduplicated range-read results, extracting TIFF frames as needed."""
    for key, blob in zip(range_keys, blobs, strict=True):
        if isinstance(blob, Exception):
            for idx, member, _fi in unique_ranges[key]:
                error_values[idx] = f"range read error for member '{member}'"
        elif blob is None or len(blob) == 0:
            for idx, member, _fi in unique_ranges[key]:
                error_values[idx] = f"empty range read for member '{member}'"
        else:
            raw = bytes(blob) if not isinstance(blob, bytes) else blob
            for idx, member, frame_idx in unique_ranges[key]:
                payload = _extract_tiff_frame(raw, frame_idx) if frame_idx is not None else raw
                if payload is None:
                    error_values[idx] = f"failed to extract frame {frame_idx} from '{member}'"
                else:
                    binary_values[idx] = payload
                    error_values[idx] = None


def _build_global_range_index(
    groups: dict[str, list[tuple[int, str, int, int, int | None]]],
    storage_options: dict[str, object],
    error_values: list[str | None],
) -> list[tuple[object, dict[tuple[str, int, int], list[tuple[int, str, int | None]]]]]:
    """Resolve filesystem paths and build per-filesystem deduplicated range indices.

    Returns a list of ``(fs, unique_ranges)`` pairs — one entry per distinct
    filesystem.  *unique_ranges* maps ``(fs_path, offset, size)`` to the rows
    that need that byte range.  Returns an empty list if no valid paths exist.

    Paths are grouped by filesystem instance (fsspec caches instances per
    protocol and storage options), so ``cat_ranges`` is called once per
    connection pool.  Batches that span multiple storage backends (e.g.
    local + S3) are handled correctly — each backend gets its own call.
    """
    if not groups:
        return []

    # id(fs) -> (fs, unique_ranges); fsspec caches fs instances so same-backend
    # paths naturally share the same id.
    fs_groups: dict[int, tuple[object, dict[tuple[str, int, int], list[tuple[int, str, int | None]]]]] = {}

    for path, entries in groups.items():
        try:
            fs, fs_path = url_to_fs(path, **storage_options)
        except (ValueError, OSError) as exc:
            logger.warning("Failed to resolve filesystem for path {!r}: {}", path, exc)
            for idx, *_ in entries:
                error_values[idx] = "failed to resolve filesystem"
            continue

        fs_id = id(fs)
        if fs_id not in fs_groups:
            fs_groups[fs_id] = (fs, {})

        _, unique_ranges = fs_groups[fs_id]
        for idx, member, offset, size, frame_idx in entries:
            unique_ranges.setdefault((fs_path, offset, size), []).append((idx, member, frame_idx))

    return list(fs_groups.values())


def _fill_range_read_rows(
    groups: dict[str, list[tuple[int, str, int, int, int | None]]],
    storage_options: dict[str, object],
    binary_values: list[object],
    error_values: list[str | None],
) -> None:
    """Batch byte-range reads grouped by filesystem, calling cat_ranges once per backend.

    Deduplicates identical (path, offset, size) tuples globally so the same
    byte range is only fetched once even if referenced by multiple rows.
    Correctly handles batches that span multiple storage backends (e.g. local
    files mixed with remote files) by dispatching each group to its own fs.
    """
    for fs, unique_ranges in _build_global_range_index(groups, storage_options, error_values):
        if not unique_ranges:
            continue

        range_keys = list(unique_ranges.keys())
        cat_paths = [fp for fp, _, _ in range_keys]
        cat_starts = [off for _, off, _ in range_keys]
        cat_ends = [off + sz for _, off, sz in range_keys]

        try:
            blobs = fs.cat_ranges(cat_paths, cat_starts, cat_ends)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("cat_ranges failed ({} ranges): {}", len(range_keys), exc)
            for entries in unique_ranges.values():
                for idx, *_ in entries:
                    error_values[idx] = "cat_ranges failed"
            continue

        _scatter_range_blobs(blobs, range_keys, unique_ranges, binary_values, error_values)


def _fill_direct_read_rows(
    groups: dict[str, list[int]],
    storage_options: dict[str, object],
    binary_values: list[object],
    error_values: list[str | None],
) -> None:
    """Read each direct file once, share bytes across all rows referencing it."""
    for path, row_idxs in groups.items():
        payload = _read_direct_file(path, storage_options)
        for idx in row_idxs:
            if payload is not None:
                binary_values[idx] = payload
                error_values[idx] = None
            else:
                error_values[idx] = "failed to read path"


def _read_direct_file(path: str, storage_options: dict[str, object]) -> bytes | None:
    try:
        with fsspec.open(path, mode="rb", **storage_options) as fobj:
            return fobj.read()
    except (OSError, RuntimeError, ValueError):
        return None


def _fill_materialized_bytes(
    df: pd.DataFrame,
    image_mask: pd.Series,
    *,
    storage_options: dict[str, object],
    binary_values: list[object],
    error_values: list[str | None],
) -> None:
    classified = _classify_rows(df, image_mask)

    for idx in classified.missing:
        error_values[idx] = "missing path"

    _fill_tar_extract_rows(classified.tar_extract, storage_options, binary_values, error_values)
    _fill_range_read_rows(classified.range_read, storage_options, binary_values, error_values)
    _fill_direct_read_rows(classified.direct_read, storage_options, binary_values, error_values)


def _init_materialization_buffers(df: pd.DataFrame) -> tuple[list[object], list[str | None]]:
    error_values = (
        df["materialize_error"].astype("object").tolist() if "materialize_error" in df.columns else [None] * len(df)
    )
    binary_values = (
        df["binary_content"].astype("object").tolist() if "binary_content" in df.columns else [None] * len(df)
    )
    return binary_values, error_values


def _build_image_mask(
    df: pd.DataFrame,
    *,
    only_missing_binary: bool,
    image_content_types: tuple[str, ...] | None,
) -> pd.Series:
    image_mask = (
        (df["modality"] == "image") if "modality" in df.columns else pd.Series(False, index=df.index, dtype=bool)
    )
    if image_content_types is not None and "content_type" in df.columns:
        image_mask &= df["content_type"].isin(image_content_types)
    if only_missing_binary and "binary_content" in df.columns:
        image_mask &= df["binary_content"].isna()
    return image_mask


def _task_with_dataframe(task: InterleavedBatch, df: pd.DataFrame) -> InterleavedBatch:
    return InterleavedBatch(
        task_id=task.task_id,
        dataset_name=task.dataset_name,
        data=df,
        _metadata=task._metadata,
        _stage_perf=task._stage_perf,
    )


def materialize_task_binary_content(
    task: InterleavedBatch,
    *,
    io_kwargs: dict[str, object] | None = None,
    only_missing_binary: bool = True,
    image_content_types: tuple[str, ...] | None = None,
) -> InterleavedBatch:
    """Return a task with image-row binary content materialized from source_ref.

    Dispatches to three I/O strategies based on source_ref contents:
    - range_read: byte_offset + byte_size present -> batched fs.cat_ranges()
    - tar_extract: member present, no byte range -> open tar + extractfile
    - direct_read: no member -> read file directly
    """
    df = task.with_parsed_source_ref_columns(prefix="_src_").reset_index(drop=True)
    if df.empty:
        return task

    binary_values, error_values = _init_materialization_buffers(df)
    image_mask = _build_image_mask(
        df,
        only_missing_binary=only_missing_binary,
        image_content_types=image_content_types,
    )
    if not image_mask.any():
        out = df.drop(columns=[c for c in _SRC_PARSE_COLS if c in df.columns])
        return _task_with_dataframe(task, out)

    storage_options = resolve_storage_options(task=task, io_kwargs=io_kwargs)
    _fill_materialized_bytes(
        df,
        image_mask,
        storage_options=storage_options,
        binary_values=binary_values,
        error_values=error_values,
    )

    out = df.drop(columns=[c for c in _SRC_PARSE_COLS if c in df.columns])
    out["binary_content"] = pd.Series(binary_values, dtype="object")
    out["materialize_error"] = pd.Series(error_values, dtype="object")
    return _task_with_dataframe(task, out)
