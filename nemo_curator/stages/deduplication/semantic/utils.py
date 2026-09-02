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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import cudf
    import cupy as cp

import pyarrow.parquet as pq
from fsspec.parquet import open_parquet_files
from fsspec.utils import get_protocol
from loguru import logger

CUDF_COLUMN_SIZE_LIMIT = 2_000_000_000
# Footer scans open one file handle or object stream per input. Batching prevents large datasets
# from exhausting file descriptors or creating an unbounded number of concurrent remote reads.
_FOOTER_BATCH_SIZE = 64


@dataclass(frozen=True)
class ParquetFileInfo:
    path: str
    num_rows: int
    metadata_bytes: int
    embedding_elements: int = 0


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
    embedding_column: str | None = None,
    storage_options: dict[str, Any] | None = None,
) -> list[ParquetFileInfo]:
    """Read exact row, embedding-element, and projected metadata sizes in input order."""
    if not files:
        return []

    retained = set(retained_columns or [])
    try:
        if storage_options or get_protocol(files[0]) != "file":
            result = []
            for start in range(0, len(files), _FOOTER_BATCH_SIZE):
                batch = files[start : start + _FOOTER_BATCH_SIZE]
                parquet_files = open_parquet_files(batch, storage_options=storage_options, row_groups=[])
                for path, parquet_file in zip(batch, parquet_files, strict=True):
                    metadata = pq.read_metadata(parquet_file)
                    metadata_bytes = 0
                    embedding_elements = 0
                    for row_group_index in range(metadata.num_row_groups):
                        row_group = metadata.row_group(row_group_index)
                        for column_index in range(row_group.num_columns):
                            column = row_group.column(column_index)
                            if _root_column(column.path_in_schema) in retained:
                                metadata_bytes += column.total_uncompressed_size
                            if _root_column(column.path_in_schema) == embedding_column:
                                embedding_elements += column.num_values
                    result.append(
                        ParquetFileInfo(
                            path,
                            metadata.num_rows,
                            metadata_bytes,
                            embedding_elements=embedding_elements,
                        )
                    )
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
                embedding_elements = sum(
                    column.meta_data.num_values
                    for row_group in footer.row_groups
                    for column in row_group.columns
                    if _root_column(column.meta_data.path_in_schema) == embedding_column
                )
                result.append(ParquetFileInfo(path, footer.num_rows, metadata_bytes, embedding_elements))
        return result  # noqa: TRY300
    except Exception as error:
        msg = f"Failed to read Parquet footer metadata for one of {len(files)} files"
        raise RuntimeError(msg) from error


def break_parquet_partition_into_groups(
    files: list[str],
    *,
    file_info: list[ParquetFileInfo],
) -> list[list[str]]:
    """Group complete files below cuDF's nested-column element limit."""
    if not files:
        return []
    elements_by_file = {info.path: info.embedding_elements for info in file_info}
    subgroups: list[list[str]] = []
    subgroup: list[str] = []
    subgroup_elements = 0
    for path in files:
        elements = elements_by_file[path]
        if elements >= CUDF_COLUMN_SIZE_LIMIT:
            # Whole files are the smallest read unit here. Supporting an individually oversized file
            # requires switching this path to cuDF's ChunkedParquetReader instead of grouping files.
            msg = f"Parquet file {path!r} has {elements} embedding elements, exceeding cuDF's column-size limit"
            raise ValueError(msg)
        if subgroup and subgroup_elements + elements >= CUDF_COLUMN_SIZE_LIMIT:
            subgroups.append(subgroup)
            subgroup = []
            subgroup_elements = 0
        subgroup.append(path)
        subgroup_elements += elements
    if subgroup:
        subgroups.append(subgroup)
    if len(subgroups) > 1:
        logger.debug(f"Broke {len(files)} files into {len(subgroups)} exact element-bounded subgroups")
    return subgroups
