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

import math
from operator import getitem

import numpy as np
import pandas as pd
from dask.base import tokenize
from dask.dataframe.core import new_dd_object
from dask.dataframe.shuffle import partitioning_index
from dask.highlevelgraph import HighLevelGraph
from dask.utils import M

from nemo_curator._compat import DASK_SHUFFLE_CAST_DTYPE


def _split_part(part, nsplits):
    total_rows = len(part)
    stride = math.ceil(total_rows / nsplits)
    return {
        i: part.iloc[offset : offset + stride]
        for i, offset in enumerate(range(0, total_rows, stride))
    }


def text_bytes_aware_merge(text_df, right_df, broadcast=True, *, on):
    if not isinstance(on, list):
        on = [on]

    string_size = text_df[on + ["text"]]
    string_size["text"] = string_size["text"].map_partitions(
        lambda s: s.str.byte_count()
    )
    post_merge_size = (
        string_size.merge(right_df[on], on=on, how="inner", broadcast=broadcast)["text"]
        .map_partitions(lambda s: s.sum())
        .compute()
    )
    limit = 2**30
    splits = np.ceil(post_merge_size.divide(limit)).astype("int")

    # Check if we need to split any partitions
    if (splits <= 1).all():
        # No need to split
        left = text_df
        right = right_df
    else:
        # Split partitions of "large" DataFrame.
        # The split functionality requires us to target
        # the input collection with more partitions.
        if text_df.npartitions >= right_df.npartitions:
            large_size = "left"
            large_frame = text_df
        else:
            large_size = "right"
            large_frame = right_df

        # Build graph to split large partitions
        dsk = {}
        npartitions = 0
        token = tokenize(large_frame, splits)
        name = f"get-split-{token}"
        split_name = f"split-large-{token}"
        for i, nsplits in enumerate(splits):
            input_partition = (large_frame._name, i)
            if nsplits > 1:
                dsk[(split_name, i)] = (_split_part, input_partition, nsplits)
                for _i in range(nsplits):
                    dsk[(name, npartitions)] = (getitem, (split_name, i), _i)
                    npartitions += 1
            else:
                dsk[(name, npartitions)] = input_partition
                npartitions += 1
        divisions = (None,) * (npartitions + 1)
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[large_frame])

        if large_size == "left":
            left = new_dd_object(graph, name, large_frame._meta, divisions)
            right = right_df
        else:
            left = text_df
            right = new_dd_object(graph, name, large_frame._meta, divisions)

    return left.merge(right, on=on, how="inner", broadcast=broadcast)


def blockwise_merge(left, right, on, how="inner"):
    return left.map_partitions(
        M.merge,
        right,
        on=on,
        how=how,
        meta=left._meta.merge(right._meta, on=on, how=how),
        enforce_metadata=False,
        transform_divisions=False,
        align_dataframes=False,
    )


def apply_bk_mapping(part, bk_map):
    # Need "sort" to preserve order after merge
    part["sort"] = range(len(part))
    return part.merge(bk_map, on="file_id", how="left").sort_values(
        "sort", ignore_index=True
    )["part_id"]


def extract_partitioning_index(
    left_df,
    merge_on,
    bk_mapping,
    parts_per_bucket_batch,
    total_bucket_partitions,
):
    # We know `right_df` is already shuffled by `merge_on`.
    # It is easy to calculate the partitioning index that each
    # row of `left_df` would need to move to for us to apply
    # a partition-wise merge between `left_df` and `right_df`.
    # We call this `global_partitioning_index`:

    if DASK_SHUFFLE_CAST_DTYPE:
        # Need to use the same type-casting logic as `shuffle`
        dtypes = {}
        if not isinstance(merge_on, list):
            merge_on = [merge_on]
        for col, dtype in left_df[merge_on].dtypes.items():
            if pd.api.types.is_numeric_dtype(dtype):
                dtypes[col] = np.float64
        if not dtypes:
            dtypes = None
        cast_dtype = {"cast_dtype": dtypes}
    else:
        # `cast_dtype` argument doesn't exist yet
        cast_dtype = {}

    num_bucket_files = bk_mapping.file_id.max() + 1
    global_partitioning_index = left_df[merge_on].map_partitions(
        partitioning_index,
        npartitions=num_bucket_files,
        meta=left_df._meta._constructor_sliced([0]),
        enforce_metadata=False,
        transform_divisions=False,
        align_dataframes=False,
        **cast_dtype,
    )

    if total_bucket_partitions < num_bucket_files:
        # Our bucket-map files were aggregated.
        # Use bk_mapping to adjust `global_partitioning_index`
        global_partitioning_index = global_partitioning_index.to_frame(
            name="file_id"
        ).map_partitions(
            apply_bk_mapping,
            bk_mapping,
            meta=global_partitioning_index._meta,
            enforce_metadata=False,
            transform_divisions=False,
            align_dataframes=False,
        )

    # Since we are iterating over `right_df`, we do not really
    # want to send the rows of `left_df` to the partition
    # indices encoded in `global_partitioning_index`. Instead, we
    # need to take a modulus with `parts_per_bucket_batch` to
    # define a `"_partitions"` column.
    left_df["_partitions"] = global_partitioning_index % parts_per_bucket_batch

    return left_df, global_partitioning_index


def filter_text_rows_by_bucket_batch(
    left_df,
    global_partitioning_index,
    bucket_part_offset,
    bucket_part_end_offset,
    total_bucket_partitions,
):
    # Drop rows that don't apply to this "bucket" batch.
    # Since we are not using ALL of `right_df`, we do not
    # need to bother transferring data that will not align
    # with the target partition
    if (bucket_part_end_offset - bucket_part_offset) < total_bucket_partitions:
        return left_df[
            global_partitioning_index.isin(
                list(
                    range(
                        bucket_part_offset,
                        bucket_part_end_offset,
                    )
                )
            )
        ]
    else:
        # No need to filter left_df
        return left_df


def merge_left_to_shuffled_right(
    subset_text_df,
    subset_bucket_df,
    merge_on,
):
    from nemo_curator.utils.fuzzy_dedup_utils.shuffle_utils import (
        rearange_by_column_direct,
    )

    # We are merging an unshuffled batch of "left" partitions
    # with a shuffled batch of "right" partitions. To minimize
    # data movement, we can manaully rerrange the "left" batch
    # by the "_partitions" column, and perform a partition-wise
    # inner merge.
    shuffled_text_df = rearange_by_column_direct(
        subset_text_df,
        col="_partitions",
        npartitions=subset_bucket_df.npartitions,
        ignore_index=True,
    ).drop(columns=["_partitions"])

    return blockwise_merge(
        shuffled_text_df,
        subset_bucket_df,
        on=merge_on,
    )
