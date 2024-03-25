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

import cudf
import dask_cudf
import numba
import numpy as np

from nemo_curator._compat import DASK_SHUFFLE_METHOD_ARG


# next-fit-descending bin packing
# https://en.wikipedia.org/wiki/Next-fit-decreasing_bin_packing
@numba.jit(nopython=True)
def build_partition(sizes: np.ndarray, max_size):
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


def update_id(df, lower_bound):
    df["output_partition_id"] += lower_bound
    return df


def get_output_part_ids_with_approx_equal_sum(
    bucket_text_bytes_df, max_text_bytes_per_part: int
):
    """'
    Create a output_series that maps the ser.index into `nparts`
    so that the total sum of bucket_val_counts_df
    for each output id are all most equal and
    less than max_text_bytes_per_part
    This is used downstream for creating equal output_ids
    """
    sizes = bucket_text_bytes_df["bucket_text_bytes"].values
    bucket_output_ar = build_partition(sizes.get(), max_text_bytes_per_part)
    df = cudf.DataFrame()
    df["bucket"] = bucket_text_bytes_df["bucket"]
    df["output_partition_id"] = bucket_output_ar
    return df


def get_agg_text_bytes_df(df, agg_column, n_partitions, shuffle=False):
    shuffle_arg = "shuffle_method" if DASK_SHUFFLE_METHOD_ARG else "shuffle"
    agg_df = (
        df[[agg_column, "text_bytes"]]
        .groupby([agg_column])
        .agg({"text_bytes": "sum"}, split_out=n_partitions, **{shuffle_arg: shuffle})
    )
    agg_df = agg_df.rename(columns={"text_bytes": f"{agg_column}_text_bytes"})
    agg_df = agg_df.reset_index(drop=False)
    # Doing a per partition sort
    # seems to cause issues with
    # jaccard shuffle  (Overflow errors)
    # which  are caught and then
    # retried with text_bytes_aware_merge
    agg_df = agg_df.persist()
    agg_df = agg_df.sort_values(
        by=[f"{agg_column}_text_bytes"], ascending=False, ignore_index=True
    )
    agg_df = agg_df.persist()
    # Added length to force computation
    # after persist
    print(f"Agg_df computed of length = {len(agg_df)}", flush=True)
    return agg_df


def get_output_map_from_text_bytes_per_bucket(ddf_bk_text_bytes):
    # String bytes limit for cuDF
    max_text_bytes_per_part = int(np.iinfo(np.int32).max // 1.2)
    print(f"max_text_bytes_per_part = {max_text_bytes_per_part}")

    # Increasing in an attempt to prevent hitting
    # ulimits
    output_map_df_meta = cudf.DataFrame({"bucket": [0], "output_partition_id": [1]})
    output_map_df_meta["bucket"] = output_map_df_meta["bucket"].astype(np.uint64)
    output_map_df_meta["output_partition_id"] = output_map_df_meta[
        "output_partition_id"
    ].astype(np.int32)
    output_map_df = ddf_bk_text_bytes.map_partitions(
        get_output_part_ids_with_approx_equal_sum,
        max_text_bytes_per_part,
        meta=output_map_df_meta,
    )
    output_map_df = output_map_df.persist()
    print(f"Step 1 of output_map_df of len: {len(output_map_df)} computed")
    lower_bounds = (
        output_map_df["output_partition_id"]
        .map_partitions(lambda s: (s.max() + 1))
        .compute()
    )
    lower_bounds = np.cumsum(lower_bounds)

    updated_parts = [
        output_map_df.get_partition(i).map_partitions(update_id, lower_bounds[i - 1])
        for i in range(1, len(lower_bounds))
    ]
    updated_parts.append(output_map_df.get_partition(0))
    output_map_df = dask_cudf.concat(updated_parts)
    output_map_df = output_map_df.persist()
    print(f"All steps of output_map_df of len: {len(output_map_df)} computed")
    return output_map_df


def get_output_map_based_on_str_bytes(ddf_bk, ddf_text):
    """
    Add output_partition_id to ddf_bk
    """
    print("Getting text bytes", flush=True)
    ddf_text["text_bytes"] = ddf_text["text"].map_partitions(
        lambda s: s.str.byte_count()
    )
    n_partitions = ddf_bk.npartitions
    ddf_text = ddf_text.drop(columns=["text"]).repartition(npartitions=n_partitions)
    ddf_bk = ddf_bk.merge(ddf_text).repartition(npartitions=n_partitions)
    del ddf_text
    ddf_bk_text_bytes = get_agg_text_bytes_df(
        ddf_bk,
        agg_column="bucket",
        n_partitions=n_partitions,
        shuffle=True,
    )
    del ddf_bk
    output_map_df = get_output_map_from_text_bytes_per_bucket(ddf_bk_text_bytes)
    return output_map_df
