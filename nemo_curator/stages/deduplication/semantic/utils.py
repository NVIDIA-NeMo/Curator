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

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cudf
    import cupy as cp

from typing import Any

import pyarrow.parquet as pq
from fsspec.parquet import open_parquet_file
from loguru import logger

PARQUET_FOOTER_BATCH_SIZE = 512
PARQUET_FOOTER_READ_WORKERS = 8


def get_array_from_df(df: "cudf.DataFrame", embedding_col: str) -> "cp.ndarray":
    """
    Convert a column of lists to a 2D array.
    """
    return df[embedding_col].list.leaves.values.reshape(len(df), -1)


def read_parquet_file_row_counts(files: list[str], storage_options: dict[str, Any] | None = None) -> dict[str, int]:
    """Read ordered per-file row counts from Parquet footer metadata."""
    if not files:
        return {}

    if storage_options:

        def read_num_rows(file: str) -> int:
            with open_parquet_file(file, storage_options=storage_options) as parquet_file:
                return pq.read_metadata(parquet_file).num_rows

        try:
            with ThreadPoolExecutor(max_workers=min(PARQUET_FOOTER_READ_WORKERS, len(files))) as executor:
                return dict(zip(files, executor.map(read_num_rows, files), strict=True))
        except Exception as error:
            msg = f"Failed to read Parquet footer metadata, starting with {files[0]!r}"
            raise RuntimeError(msg) from error

    import pylibcudf as plc

    row_counts = {}
    for offset in range(0, len(files), PARQUET_FOOTER_BATCH_SIZE):
        batch = files[offset : offset + PARQUET_FOOTER_BATCH_SIZE]
        try:
            footers = plc.io.parquet_metadata.read_parquet_footers(plc.io.SourceInfo(batch))
        except Exception as error:
            msg = f"Failed to read Parquet footer metadata for batch starting with {batch[0]!r}"
            raise RuntimeError(msg) from error

        if len(footers) != len(batch):
            msg = f"Expected {len(batch)} Parquet footers, got {len(footers)}"
            raise RuntimeError(msg)
        row_counts.update((file, int(footer.num_rows)) for file, footer in zip(batch, footers, strict=True))
    return row_counts


def break_parquet_partition_into_groups(  # noqa: C901
    files: list[str],
    embedding_dim: int | None = None,
    storage_options: dict[str, Any] | None = None,
    *,
    row_counts: dict[str, int] | None = None,
) -> list[list[str]]:
    """Break parquet files into groups to avoid cudf 2bn row limit."""
    if not files:
        return []
    if embedding_dim is None:
        # Default aggressive assumption of 1024 dimensional embedding
        embedding_dim = 1024
    if embedding_dim <= 0:
        msg = f"embedding_dim must be positive, got {embedding_dim}"
        raise ValueError(msg)

    cudf_max_num_rows = 2_000_000_000  # cudf only allows 2bn rows
    max_rows_per_group = cudf_max_num_rows // embedding_dim
    if max_rows_per_group == 0:
        msg = f"embedding_dim {embedding_dim} exceeds the cuDF list-child element limit"
        raise ValueError(msg)

    if row_counts is None:
        row_counts = read_parquet_file_row_counts(files, storage_options=storage_options)
    subgroups: list[list[str]] = []
    subgroup: list[str] = []
    subgroup_rows = 0
    for file in files:
        if file not in row_counts:
            msg = f"Missing Parquet row count for {file!r}"
            raise ValueError(msg)
        file_rows = row_counts[file]
        if file_rows < 0:
            msg = f"Parquet row count for {file!r} must be non-negative, got {file_rows}"
            raise ValueError(msg)
        if file_rows > max_rows_per_group:
            msg = f"Parquet file {file!r} has {file_rows} rows, exceeding the per-group limit of {max_rows_per_group}"
            raise ValueError(msg)
        if subgroup and subgroup_rows + file_rows > max_rows_per_group:
            subgroups.append(subgroup)
            subgroup = []
            subgroup_rows = 0
        subgroup.append(file)
        subgroup_rows += file_rows
    if subgroup:
        subgroups.append(subgroup)

    if len(subgroups) > 1:
        logger.debug(f"Broke {len(files)} files into {len(subgroups)} row-bounded subgroups")
    return subgroups
