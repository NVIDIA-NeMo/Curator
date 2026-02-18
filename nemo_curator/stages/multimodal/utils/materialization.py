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

import tarfile

import fsspec
import pandas as pd

from nemo_curator.tasks import MultiBatchTask

from .validation_utils import resolve_storage_options


def load_bytes_from_content_reference(
    content_path: str | None,
    content_key: str | None,
    storage_options: dict[str, object],
    byte_cache: dict[tuple[str, str], bytes | None],
) -> bytes | None:
    if not content_path:
        return None

    cache_key = (str(content_path), str(content_key or ""))
    if cache_key in byte_cache:
        return byte_cache[cache_key]

    try:
        with fsspec.open(str(content_path), mode="rb", **storage_options) as fobj:
            if content_key:
                with tarfile.open(fileobj=fobj, mode="r:*") as tf:
                    try:
                        extracted = tf.extractfile(content_key)
                    except KeyError:
                        extracted = None
                    payload = extracted.read() if extracted is not None else None
                    byte_cache[cache_key] = payload
                    return payload
            payload = fobj.read()
            byte_cache[cache_key] = payload
            return payload
    except Exception:  # noqa: BLE001
        byte_cache[cache_key] = None
        return None


def load_bytes_from_metadata_source(
    source_value: str | None,
    storage_options: dict[str, object],
    byte_cache: dict[tuple[str, str], bytes | None],
) -> bytes | None:
    source = MultiBatchTask.parse_metadata_source(source_value)
    return load_bytes_from_content_reference(
        content_path=source.get("content_path"),
        content_key=source.get("content_key"),
        storage_options=storage_options,
        byte_cache=byte_cache,
    )


def _init_materialization_buffers(df: pd.DataFrame) -> tuple[list[object], list[str | None]]:
    error_values = (
        df["materialize_error"].astype("object").tolist() if "materialize_error" in df.columns else [None] * len(df)
    )
    binary_values = df["binary_content"].astype("object").tolist() if "binary_content" in df.columns else [None] * len(df)
    return binary_values, error_values


def _build_image_mask(
    df: pd.DataFrame,
    *,
    only_missing_binary: bool,
    image_content_types: tuple[str, ...] | None,
) -> pd.Series:
    image_mask = (df["modality"] == "image") if "modality" in df.columns else pd.Series(False, index=df.index, dtype=bool)
    if image_content_types is not None and "content_type" in df.columns:
        image_mask &= df["content_type"].isin(image_content_types)
    if only_missing_binary and "binary_content" in df.columns:
        image_mask &= df["binary_content"].isna()
    return image_mask


def _fill_materialized_bytes(
    df: pd.DataFrame,
    image_mask: pd.Series,
    *,
    storage_options: dict[str, object],
    binary_values: list[object],
    error_values: list[str | None],
) -> None:
    pending = df[image_mask]
    for content_path, idxs in pending.groupby("_src_content_path").groups.items():
        if content_path is None or pd.isna(content_path):
            for idx in idxs:
                error_values[idx] = "missing content_path"
            continue
        keyed_rows: list[tuple[int, str]] = []
        direct_rows: list[int] = []
        for idx in idxs:
            raw_key = df.loc[idx, "_src_content_key"]
            if raw_key not in (None, "") and pd.notna(raw_key):
                keyed_rows.append((idx, str(raw_key)))
            else:
                direct_rows.append(idx)
        content_path_str = str(content_path)
        _fill_group_keyed_rows(content_path_str, keyed_rows, storage_options, binary_values, error_values)
        _fill_group_direct_rows(content_path_str, direct_rows, storage_options, binary_values, error_values)


def _fill_group_keyed_rows(
    content_path: str,
    keyed_rows: list[tuple[int, str]],
    storage_options: dict[str, object],
    binary_values: list[object],
    error_values: list[str | None],
) -> None:
    if not keyed_rows:
        return
    key_cache: dict[str, bytes | None] = {}
    try:
        with fsspec.open(content_path, mode="rb", **storage_options) as fobj, tarfile.open(
            fileobj=fobj, mode="r:*"
        ) as tf:
            for idx, content_key in keyed_rows:
                if content_key not in key_cache:
                    try:
                        extracted = tf.extractfile(content_key)
                    except KeyError:
                        extracted = None
                    key_cache[content_key] = extracted.read() if extracted is not None else None
                payload = key_cache[content_key]
                if payload is None:
                    error_values[idx] = f"missing content_key '{content_key}'"
                    continue
                binary_values[idx] = payload
                error_values[idx] = None
    except Exception:  # noqa: BLE001
        for idx, _ in keyed_rows:
            error_values[idx] = "failed to read content_path"


def _fill_group_direct_rows(
    content_path: str,
    direct_rows: list[int],
    storage_options: dict[str, object],
    binary_values: list[object],
    error_values: list[str | None],
) -> None:
    if not direct_rows:
        return
    try:
        with fsspec.open(content_path, mode="rb", **storage_options) as fobj:
            payload = fobj.read()
        for idx in direct_rows:
            binary_values[idx] = payload
            error_values[idx] = None
    except Exception:  # noqa: BLE001
        for idx in direct_rows:
            error_values[idx] = "failed to read content_path"


def _task_with_dataframe(task: MultiBatchTask, df: pd.DataFrame) -> MultiBatchTask:
    return MultiBatchTask(
        task_id=task.task_id,
        dataset_name=task.dataset_name,
        data=df,
        _metadata=task._metadata,
        _stage_perf=task._stage_perf,
    )


def materialize_task_binary_content(
    task: MultiBatchTask,
    *,
    io_kwargs: dict[str, object] | None = None,
    only_missing_binary: bool = True,
    image_content_types: tuple[str, ...] | None = None,
) -> MultiBatchTask:
    """Return a task with image-row binary content materialized from metadata_source."""
    df = task.with_parsed_source_columns(prefix="_src_").reset_index(drop=True)
    if df.empty:
        return task
    binary_values, error_values = _init_materialization_buffers(df)
    image_mask = _build_image_mask(
        df,
        only_missing_binary=only_missing_binary,
        image_content_types=image_content_types,
    )
    if not image_mask.any():
        out = df.drop(columns=[c for c in df.columns if c.startswith("_src_")], errors="ignore")
        return _task_with_dataframe(task, out)

    storage_options = resolve_storage_options(task=task, io_kwargs=io_kwargs)
    _fill_materialized_bytes(
        df,
        image_mask,
        storage_options=storage_options,
        binary_values=binary_values,
        error_values=error_values,
    )

    out = df.drop(columns=[c for c in df.columns if c.startswith("_src_")], errors="ignore")
    out["binary_content"] = pd.Series(binary_values, dtype="object")
    out["materialize_error"] = pd.Series(error_values, dtype="object")
    return _task_with_dataframe(task, out)
