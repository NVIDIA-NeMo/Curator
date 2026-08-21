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

from typing import TYPE_CHECKING, Any, Literal

import cupy as cp

if TYPE_CHECKING:
    import cudf

import pyarrow.parquet as pq
from fsspec.parquet import open_parquet_file
from loguru import logger

EmbeddingStorageDtype = Literal["auto", "float16", "float32"]


def get_array_from_df(df: "cudf.DataFrame", embedding_col: str) -> "cp.ndarray":
    """
    Convert a column of lists to a 2D array.
    """
    return df[embedding_col].list.leaves.values.reshape(len(df), -1)


def decode_embedding_array(
    df: "cudf.DataFrame",
    embedding_col: str,
    storage_dtype: EmbeddingStorageDtype = "auto",
) -> "cp.ndarray":
    """Decode embedding storage to FP16 or FP32 independently of compute policy."""
    if storage_dtype not in {"auto", "float16", "float32"}:
        msg = f"Unsupported embedding storage dtype: {storage_dtype}"
        raise ValueError(msg)

    embeddings = get_array_from_df(df, embedding_col)
    if storage_dtype == "auto":
        if embeddings.dtype == cp.uint16:
            storage_dtype = "float16"
        elif embeddings.dtype == cp.float32:
            return embeddings
        elif embeddings.dtype == cp.float64:
            # Pairwise historically accepted Python-float list columns, whose
            # leaves cuDF represents as FP64. Keep that input compatible while
            # normalizing it to the supported FP32 compute/storage contract.
            return embeddings.astype(cp.float32, copy=False)
        else:
            msg = f"Expected uint16 or float32 embedding storage, got {embeddings.dtype}"
            raise TypeError(msg)

    if storage_dtype == "float16":
        if embeddings.dtype != cp.uint16:
            msg = f"Expected uint16 bit storage for float16 embeddings, got {embeddings.dtype}"
            raise TypeError(msg)
        return embeddings.view(cp.float16)

    if embeddings.dtype != cp.float32:
        msg = f"Expected float32 embedding storage, got {embeddings.dtype}"
        raise TypeError(msg)
    return embeddings


def break_parquet_partition_into_groups(
    files: list[str], embedding_dim: int | None = None, storage_options: dict[str, Any] | None = None
) -> list[list[str]]:
    """Break parquet files into groups to avoid cudf 2bn row limit."""
    if embedding_dim is None:
        # Default aggressive assumption of 1024 dimensional embedding
        embedding_dim = 1024

    cudf_max_num_rows = 2_000_000_000  # cudf only allows 2bn rows
    cudf_max_num_elements = cudf_max_num_rows / embedding_dim  # cudf considers each element in an array to be a row

    # Load the first file and get the number of rows to estimate
    with open_parquet_file(files[0], storage_options=storage_options) as f:
        # Multiply by 1.5 to adjust for skew
        avg_num_rows = pq.read_metadata(f).num_rows * 1.5

    max_files_per_subgroup = int(cudf_max_num_elements / avg_num_rows)
    max_files_per_subgroup = max(1, max_files_per_subgroup)  # Ensure at least 1 file per subgroup

    # Break files into subgroups
    subgroups = [files[i : i + max_files_per_subgroup] for i in range(0, len(files), max_files_per_subgroup)]
    if len(subgroups) > 1:
        logger.debug(
            f"Broke {len(files)} files into {len(subgroups)} subgroups with max {max_files_per_subgroup} files per subgroup"
        )
    return subgroups
