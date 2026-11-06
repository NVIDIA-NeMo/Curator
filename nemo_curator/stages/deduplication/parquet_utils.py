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

import glob
from typing import TYPE_CHECKING, Any

import fsspec
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Sequence


def _get_sources(
    filepath: str | list[str],
    storage_options: dict[str, Any] | None,
    filesystem: fsspec.AbstractFileSystem | None = None,
) -> tuple[fsspec.AbstractFileSystem, str | list[str], list[str], bool]:
    paths = [filepath] if isinstance(filepath, str) else filepath
    if not paths:
        msg = "At least one Parquet path is required"
        raise ValueError(msg)

    if filesystem is None:
        filesystem, first_path = fsspec.core.url_to_fs(paths[0], **(storage_options or {}))
    else:
        first_path = filesystem._strip_protocol(paths[0])
    resolved_paths = [first_path, *(filesystem._strip_protocol(path) for path in paths[1:])]

    if isinstance(filepath, str) and filesystem.isdir(first_path):
        parquet_paths = [path for path in filesystem.find(first_path) if not _is_metadata_file(path)]
        return filesystem, first_path, parquet_paths, True

    if isinstance(filepath, str) and glob.has_magic(first_path):
        expanded_paths = filesystem.glob(first_path)
        return filesystem, expanded_paths, expanded_paths, False

    expanded_paths: list[str] = []
    for path in resolved_paths:
        if filesystem.isdir(path):
            expanded_paths.extend(item for item in filesystem.find(path) if not _is_metadata_file(item))
        elif glob.has_magic(path):
            expanded_paths.extend(filesystem.glob(path))
        else:
            expanded_paths.append(path)

    source: str | list[str] = resolved_paths[0] if isinstance(filepath, str) else resolved_paths
    return filesystem, source, expanded_paths, False


def _is_metadata_file(path: str) -> bool:
    name = path.rsplit("/", maxsplit=1)[-1]
    return name.startswith(("_", "."))


def _get_pyarrow_filesystem_paths(filepath: str | list[str], filesystem: pafs.FileSystem) -> list[str]:
    paths = [filepath] if isinstance(filepath, str) else filepath
    parquet_paths: list[str] = []
    for path in paths:
        if glob.has_magic(path):
            return []

        info = filesystem.get_file_info(path)
        if info.type == pafs.FileType.Directory:
            entries = filesystem.get_file_info(pafs.FileSelector(info.path, recursive=True))
            parquet_paths.extend(
                entry.path
                for entry in entries
                if entry.type == pafs.FileType.File and not _is_metadata_file(entry.path)
            )
        else:
            parquet_paths.append(info.path)

    return parquet_paths


def _column_is_selected(path: str, columns: Sequence[str] | None) -> bool:
    if columns is None:
        return True
    return any(path == column or path.startswith(f"{column}.") for column in columns)


def _metadata_has_page_indexed_delta_byte_array(metadata: pq.FileMetaData, columns: Sequence[str] | None) -> bool:
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            if (
                "DELTA_BYTE_ARRAY" in column.encodings
                and column.has_column_index
                and column.has_offset_index
                and _column_is_selected(column.path_in_schema, columns)
            ):
                return True
    return False


def has_page_indexed_delta_byte_array(
    filepath: str | list[str],
    columns: Sequence[str] | None = None,
    storage_options: dict[str, Any] | None = None,
    filesystem: fsspec.AbstractFileSystem | pafs.FileSystem | None = None,
) -> bool:
    """Return whether a selected Parquet column has the affected encoding and page indexes."""
    if columns is not None and not columns:
        return False

    if isinstance(filesystem, pafs.FileSystem):
        filesystem_paths = _get_pyarrow_filesystem_paths(filepath, filesystem)
        for path in filesystem_paths:
            with filesystem.open_input_file(path) as file_obj:
                metadata = pq.ParquetFile(file_obj).metadata
            if _metadata_has_page_indexed_delta_byte_array(metadata, columns):
                return True
        return False

    filesystem, _, paths, _ = _get_sources(filepath, storage_options, filesystem)
    for path in paths:
        with filesystem.open(path, "rb") as file_obj:
            metadata = pq.ParquetFile(file_obj).metadata
        if _metadata_has_page_indexed_delta_byte_array(metadata, columns):
            return True

    return False


def read_parquet_with_pyarrow(
    filepath: str | list[str],
    *,
    storage_options: dict[str, Any] | None = None,
    allow_mismatched_pq_schemas: bool = False,
    **kwargs,
) -> pa.Table:
    """Read Parquet data through fsspec and PyArrow, preserving selected columns and filters."""
    provided_filesystem = kwargs.pop("filesystem", None)
    if isinstance(provided_filesystem, pafs.FileSystem):
        if (
            allow_mismatched_pq_schemas
            and isinstance(filepath, list)
            and len(filepath) > 1
            and "schema" not in kwargs
            and all(provided_filesystem.get_file_info(path).type == pafs.FileType.File for path in filepath)
        ):
            schemas = []
            for path in filepath:
                with provided_filesystem.open_input_file(path) as file_obj:
                    schemas.append(pq.ParquetFile(file_obj).schema_arrow)
            kwargs["schema"] = pa.unify_schemas(schemas)
        return pq.read_table(filepath, filesystem=provided_filesystem, **kwargs)

    fsspec_filesystem = provided_filesystem if isinstance(provided_filesystem, fsspec.AbstractFileSystem) else None
    filesystem, source, paths, is_directory = _get_sources(filepath, storage_options, fsspec_filesystem)
    if provided_filesystem is None:
        arrow_filesystem = pafs.PyFileSystem(pafs.FSSpecHandler(filesystem))
    elif isinstance(provided_filesystem, fsspec.AbstractFileSystem):
        arrow_filesystem = pafs.PyFileSystem(pafs.FSSpecHandler(provided_filesystem))
    else:
        arrow_filesystem = provided_filesystem

    if allow_mismatched_pq_schemas and len(paths) > 1 and not is_directory and "schema" not in kwargs:
        schemas = []
        for path in paths:
            with filesystem.open(path, "rb") as file_obj:
                schemas.append(pq.ParquetFile(file_obj).schema_arrow)
        kwargs["schema"] = pa.unify_schemas(schemas)

    return pq.read_table(source, filesystem=arrow_filesystem, **kwargs)
