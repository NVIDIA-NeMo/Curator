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
import dask_cuda
import numpy as np
from dask import config
from dask.dataframe.shuffle import rearrange_by_column
from dask_cuda.explicit_comms.dataframe.shuffle import shuffle as explicit_comms_shuffle
from packaging.version import Version

from nemo_curator.gpu_deduplication.jaccard_utils.get_output_map_utils import (
    build_partition,
    get_agg_text_bytes_df,
)

USE_EXCOMMS = Version(dask_cuda.__version__) >= Version("23.10")


def rearange_by_column_direct(
    df,
    col,
    npartitions,
    ignore_index,
    excomms_default=USE_EXCOMMS,
):
    # Execute a "direct" shuffle operation without staging
    if config.get("explicit-comms", excomms_default):
        # Use explicit comms unless the user has
        # disabled it with the dask config system,
        # or we are using an older version of dask-cuda
        return explicit_comms_shuffle(
            df,
            [col],
            npartitions=npartitions,
            ignore_index=ignore_index,
        )
    else:
        return rearrange_by_column(
            df,
            col=col,
            shuffle="tasks",
            # Prevent staged shuffling by setting max_branch
            # to the number of input partitions + 1
            max_branch=npartitions + 1,
            npartitions=npartitions,
            ignore_index=ignore_index,
        )


def get_shuffle_part_ids_df(agg_df, partition_on, num_workers=None):
    sizes = agg_df[f"{partition_on}_text_bytes"].values
    max_text_bytes_per_part = int(np.iinfo(np.int32).max // 1.2)

    # Adjust max_text_bytes_per_part if the number of output
    # partitions is small compared to the number of workers.
    # Sometimes we just have very few output partitions to
    # deal with, and just need a larger batch
    npartitions_min = int(num_workers * 0.8)
    while True:
        output_ar = build_partition(sizes.get(), max_text_bytes_per_part)
        if output_ar.max() > npartitions_min or max_text_bytes_per_part < 2**24:
            break
        max_text_bytes_per_part = int(max_text_bytes_per_part // 2.0)

    df = cudf.DataFrame()
    df[partition_on] = agg_df[partition_on]
    df["_partitions"] = output_ar
    return df


def get_shuffle_partition_info(df, partition_on, num_workers=None):
    df["text_bytes"] = df["text"].map_partitions(lambda s: s.str.byte_count())
    agg_df = get_agg_text_bytes_df(df, partition_on, 1)
    del df

    agg_df = agg_df.reset_index(drop=True)
    shuffle_part_ids = agg_df.map_partitions(
        get_shuffle_part_ids_df, partition_on, num_workers=num_workers
    ).persist()
    return shuffle_part_ids


def text_bytes_aware_shuffle(df, partition_on, num_workers=None):
    """
    This shuffle takes into account the text bytes of each partition
    and tries to make sure that the output partitions do not exceed
    the char limit of cuDF

    Args:
        df: dask_cudf dataframe
        partition_on: column name to partition on


    Returns:
        dask_cudf dataframe with _partitions columns
    """
    print("Starting text bytes aware shuffle", flush=True)
    df = df.persist()
    shuffle_part_ids = get_shuffle_partition_info(
        df, partition_on, num_workers=num_workers
    )
    n_output_partitions = shuffle_part_ids["_partitions"].max().compute() + 1
    n_output_partitions = int(n_output_partitions)
    df = df.merge(shuffle_part_ids, on=partition_on, how="inner").persist()

    df = (
        rearange_by_column_direct(
            df,
            col="_partitions",
            npartitions=n_output_partitions,
            ignore_index=True,
            excomms_default=True,
        )
        .drop(columns=["_partitions"])
        .persist()
    )
    print(f"Will write {len(df)} rows to disk", flush=True)
    return df
