# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import cudf
    import cupy as cp

import pyarrow.parquet as pq
from fsspec.parquet import open_parquet_files
from loguru import logger

CUDF_COLUMN_SIZE_LIMIT = 2_000_000_000
_FOOTER_BATCH_SIZE = 64


@dataclass(frozen=True)
class ParquetFileInfo:
    path: str
    num_rows: int
    metadata_bytes: int
    footer: Any | None = None


def get_array_from_df(df: "cudf.DataFrame", embedding_col: str) -> "cp.ndarray":
    """
    Convert a column of lists to a 2D array.
    """
    return df[embedding_col].list.leaves.values.reshape(len(df), -1)


def _root_column(path: str | list[str]) -> str:
    return path[0] if isinstance(path, list) else path.split(".", maxsplit=1)[0]


def read_parquet_file_info(  # noqa: C901
    files: list[str],
    *,
    retained_columns: list[str] | None = None,
    storage_options: dict[str, Any] | None = None,
) -> list[ParquetFileInfo]:
    """Read exact row counts and projected metadata sizes in input order."""
    if not files:
        return []

    retained = set(retained_columns or [])
    try:
        if storage_options:
            result = []
            for start in range(0, len(files), _FOOTER_BATCH_SIZE):
                batch = files[start : start + _FOOTER_BATCH_SIZE]
                parquet_files = open_parquet_files(batch, storage_options=storage_options, row_groups=[])
                for path, parquet_file in zip(batch, parquet_files, strict=True):
                    metadata = pq.read_metadata(parquet_file)
                    metadata_bytes = 0
                    for row_group_index in range(metadata.num_row_groups):
                        row_group = metadata.row_group(row_group_index)
                        for column_index in range(row_group.num_columns):
                            column = row_group.column(column_index)
                            if _root_column(column.path_in_schema) in retained:
                                metadata_bytes += column.total_uncompressed_size
                    result.append(ParquetFileInfo(path, metadata.num_rows, metadata_bytes))
            return result

        import pylibcudf as plc

        result = []
        for start in range(0, len(files), _FOOTER_BATCH_SIZE):
            batch = files[start : start + _FOOTER_BATCH_SIZE]
            footers = plc.io.parquet_metadata.read_parquet_footers(plc.io.SourceInfo(batch))
            for path, footer in zip(batch, footers, strict=True):
                metadata_bytes = sum(
                    column.meta_data.total_uncompressed_size
                    for row_group in footer.row_groups
                    for column in row_group.columns
                    if _root_column(column.meta_data.path_in_schema) in retained
                )
                result.append(ParquetFileInfo(path, footer.num_rows, metadata_bytes, footer))
        return result  # noqa: TRY300
    except Exception as error:
        msg = f"Failed to read Parquet footer metadata for one of {len(files)} files"
        raise RuntimeError(msg) from error


def iter_parquet_chunks(
    files: list[str],
    *,
    columns: list[str],
    footers: list[Any] | None = None,
    chunk_read_limit: int | None = None,
    pass_read_limit: int | None = None,
) -> Iterator["cudf.DataFrame"]:
    """Yield zero-copy cuDF frames from pylibcudf's bounded Parquet reader."""
    if not files:
        return

    import cudf
    import cupy as cp
    import pylibcudf as plc

    if chunk_read_limit is None:
        # One output frame can coexist with its two Parquet encoding buffers
        # and the next frame while reads and writes overlap.
        chunk_read_limit = cp.cuda.runtime.memGetInfo()[0] // 4
    if pass_read_limit is None:
        # Match libcudf's derived pass limit when only an output limit is supplied.
        pass_read_limit = chunk_read_limit + chunk_read_limit // 2

    source = plc.io.SourceInfo(files)
    options = plc.io.parquet.ParquetReaderOptions.builder(source).allow_mismatched_pq_schemas(True).build()
    options.set_column_names(columns)
    reader = plc.io.parquet.ChunkedParquetReader(
        options,
        chunk_read_limit=chunk_read_limit,
        pass_read_limit=pass_read_limit,
        parquet_metadatas=footers,
    )
    while reader.has_next():
        yield cudf.DataFrame.from_pylibcudf(reader.read_chunk())


def break_parquet_partition_into_groups(
    files: list[str],
    embedding_dim: int | None = None,
    storage_options: dict[str, Any] | None = None,
    *,
    file_info: list[ParquetFileInfo] | None = None,
) -> list[list[str]]:
    """Group complete files below cuDF's nested-column element limit."""
    if not files:
        return []
    if embedding_dim is None:
        embedding_dim = 1024
    if embedding_dim <= 0:
        msg = f"embedding_dim must be positive, got {embedding_dim}"
        raise ValueError(msg)

    max_rows = CUDF_COLUMN_SIZE_LIMIT // embedding_dim
    infos = file_info or read_parquet_file_info(files, storage_options=storage_options)
    rows_by_file = {info.path: info.num_rows for info in infos}
    subgroups: list[list[str]] = []
    subgroup: list[str] = []
    subgroup_rows = 0
    for path in files:
        rows = rows_by_file[path]
        if rows > max_rows:
            msg = f"Parquet file {path!r} has {rows} rows, exceeding the per-read limit of {max_rows}"
            raise ValueError(msg)
        if subgroup and subgroup_rows + rows > max_rows:
            subgroups.append(subgroup)
            subgroup = []
            subgroup_rows = 0
        subgroup.append(path)
        subgroup_rows += rows
    if subgroup:
        subgroups.append(subgroup)
    if len(subgroups) > 1:
        logger.debug(f"Broke {len(files)} files into {len(subgroups)} exact row-bounded subgroups")
    return subgroups
