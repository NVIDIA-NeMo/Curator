# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Tuple

import numba
import numpy as np

from nemo_curator._compat import DASK_SHUFFLE_METHOD_ARG


def get_agg_text_bytes_df(
    df: dask_cudf.DataFrame,
    agg_column: str,
    bytes_column: str,
    n_partitions: int,
    shuffle: bool = False,
) -> Tuple[dask_cudf.DataFrame, int]:
    """
    Groupby bucket and calculate total bytes for a bucket.
    """
    shuffle_arg = "shuffle_method" if DASK_SHUFFLE_METHOD_ARG else "shuffle"
    agg_df = (
        df[[agg_column, bytes_column]]
        .groupby([agg_column])
        .agg({bytes_column: "sum"}, split_out=n_partitions, **{shuffle_arg: shuffle})
    )
    agg_df = agg_df.reset_index(drop=False)
    # Doing a per partition sort
    # seems to cause issues with
    # jaccard shuffle  (Overflow errors)
    # which  are caught and then
    # retried with text_bytes_aware_merge
    agg_df = agg_df.persist()
    agg_df = agg_df.sort_values(by=[bytes_column], ascending=False, ignore_index=True)
    agg_df = agg_df.persist()
    # Added length to force computation
    # after persist
    agg_df_len = len(agg_df)

    return agg_df, agg_df_len


# next-fit-descending bin packing
# https://en.wikipedia.org/wiki/Next-fit-decreasing_bin_packing
@numba.jit(nopython=True)
def build_partition(sizes: np.ndarray, max_size: int) -> np.ndarray:
    """
    Given an array of items and a max bin size this method
    attempts to return a grouping of items such that no group exceeds
    the max bin size using the Next-fit-decreasing bin packing approach.
    """
    i: int = 0
    count: int = 0
    current: int = 0
    size: int = 0
    partition = np.empty(sizes.shape, dtype=np.int32)
    for i in range(len(sizes)):
        size = sizes[i]
        if current + size < max_size:
            partition[i] = count
            current += size
        else:
            count += 1
            current = size
            partition[i] = count
    return partition
