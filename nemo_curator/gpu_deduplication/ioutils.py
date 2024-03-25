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

import os
from typing import Sequence

import cudf
import dask_cudf
from dask import dataframe as dd
from tqdm import tqdm


# TODO:
# Combine this with
# nemo_curator.distributed_utils.read_cudf_jsonl
def read_json_func(files, engine="cudf", include_path_column=False, columns=None):
    """
    Reads multiple Json Lines files into a cuDF
    dataframe with an additional `path` column denoting the path
    of the input file.
    """
    if not include_path_column:
        if columns:
            return cudf.read_json(files, engine="cudf", lines=True)[columns]
        else:
            return cudf.read_json(files, engine="cudf", lines=True)

    dfs = []
    for file in files:
        if columns:
            df = cudf.read_json(file, engine=engine, lines=True)[columns]
        else:
            df = cudf.read_json(file, engine=engine, lines=True)
        df["path"] = file
        dfs.append(df)
    return cudf.concat(dfs, ignore_index=True)


def bucketed_read(files, func=read_json_func, b_size=2, meta=None, **kwargs):
    """
    Read files with `b_size` number of files per bucket.
    Users can specify their own read
    """
    filepaths = [
        files[i : i + b_size] for i in range(0, len(files), b_size)  # noqa: E203
    ]
    if meta:
        return dd.from_map(func, filepaths, meta=meta, **kwargs)
    else:
        return dd.from_map(func, filepaths, **kwargs)


# TODO: Remove this function
def regular_read_json(files, include_path_column=False):
    return dask_cudf.read_json(
        files, engine="cudf", lines=True, include_path_column=include_path_column
    )


def batched_writing(
    dask_df: dask_cudf.DataFrame,
    output_path: str,
    partition_on: Sequence[str],
    parts_ber_batch: int = 32,
):
    """
    Write a dask dataframe to parquet in batches.
    This allows us to do batched exectution and prevent OOMs
    Args:
        dask_df: dask dataframe to write
        output_path: path to write to
        partition_on: columns to partition on
        parts_ber_batch: number of partitions per batch
    """

    total_partitions = dask_df.npartitions
    for batch_id, part_offset in tqdm(
        enumerate(range(0, dask_df.npartitions, parts_ber_batch))
    ):
        print(f"\nStarted processing batch in = {batch_id}", flush=True)
        df = dask_df.partitions[part_offset : part_offset + parts_ber_batch]
        if partition_on:
            df.to_parquet(
                output_path,
                partition_on=partition_on,
                name_function=lambda x: f"batch_{batch_id}_part_{x}.parquet",
                write_metadata_file=False,
            )
        else:
            df.to_parquet(
                output_path,
                name_function=lambda x: f"batch_{batch_id}_part_{x}.parquet",
                write_metadata_file=False,
            )
        print(
            f"Part {part_offset+parts_ber_batch}/{total_partitions} completed",
            flush=True,
        )


def strip_trailing_sep(path: str):
    """
    Strips a path string of trailing path seperators like `/` if any.
    """
    return path.rstrip(os.path.sep)
