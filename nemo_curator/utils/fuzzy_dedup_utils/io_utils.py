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
from glob import glob

import cudf
import dask_cudf
import numpy as np
from dask import dataframe as dd

from nemo_curator.utils.fuzzy_dedup_utils.id_mapping import convert_str_id_to_int


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


def get_text_ddf_from_json_path_with_blocksize(
    input_data_paths, num_files, blocksize, id_column, text_column
):
    data_paths = [
        entry.path for data_path in input_data_paths for entry in os.scandir(data_path)
    ]
    data_paths = [f for f in data_paths if f.endswith(".jsonl")]
    data_paths.sort()
    if num_files != -1:
        data_paths = data_paths[:num_files]
    meta_df = cudf.DataFrame(
        {
            text_column: ["x"],
            id_column: ["x"],
        }
    )
    print(
        f"Number of files being read for jaccard calculation = {len(data_paths)}",
        flush=True,
    )
    filepaths_ls = chunk_files(data_paths, blocksize)
    text_ddf = dd.from_map(
        read_json_func, filepaths_ls, columns=list(meta_df.columns), meta=meta_df
    )
    text_ddf = text_ddf.map_partitions(
        convert_str_id_to_int,
        id_column=id_column,
        meta=cudf.DataFrame(
            {text_column: ["a"], "doc_id": [0], "dataset_id": np.uint32(1)}
        ),
    )
    return text_ddf


def get_bucket_ddf_from_parquet_path(input_bucket_path, num_workers):
    # Read parquet-formatted parquet files
    ddf_bk = dask_cudf.read_parquet(
        input_bucket_path,
        blocksize="512MiB",
        aggregate_files=True,
    )
    # Repartition to ensure we at least have num_workers partitions
    npartitions = max(ddf_bk.npartitions, num_workers)
    ddf_bk = ddf_bk.repartition(npartitions=npartitions)
    print(f"Number of ddf_bk partitions = {ddf_bk.npartitions}", flush=True)
    return ddf_bk


def aggregated_anchor_docs_with_bk_read(path, blocksize):
    from dask.utils import natural_sort_key
    from pyarrow.dataset import dataset

    ds = dataset(
        sorted(glob(f"{path}/*.parquet"), key=natural_sort_key),
        format="parquet",
    )
    chunks = chunk_files(ds.get_fragments(), blocksize)

    # Record mapping between file indices and partition indices.
    # We need to do this, because our anchor_docs_with_bk data
    # should be shuffled on disk.
    assert len(chunks)
    part_id = np.repeat(
        np.arange(len(chunks), dtype="int32"),
        np.fromiter(map(len, chunks), dtype="int32"),
    )
    file_id = np.arange(len(part_id), dtype="int32")
    mapping_df = cudf.DataFrame({"file_id": file_id, "part_id": part_id})

    meta = cudf.DataFrame.from_arrow(ds.schema.empty_table())
    return dd.from_map(cudf.read_parquet, chunks, meta=meta), mapping_df


def get_restart_offsets(output_path):
    bucket_offset, text_offset = 0, 0
    fn = f"{output_path}/_restart_offset.txt"
    if os.path.exists(fn):
        with open(fn, "r") as f:
            offsets = f.readline().strip("\n").split(",")
            bucket_offset = int(offsets[0])
            text_offset = int(offsets[1])
    return bucket_offset, text_offset


def update_restart_offsets(output_path, bucket_offset, text_offset):
    with open(f"{output_path}/_restart_offset.txt", "w") as f:
        f.write(f"{bucket_offset},{text_offset}\n")


def chunk_files(file_list, max_size_mb):
    """
    Chunk files into lists of files that are less than max_size_mb
    """

    max_size_bytes = max_size_mb * 1024 * 1024
    chunks = []
    current_chunk = []
    current_size = 0

    for frag_or_path in file_list:
        if isinstance(frag_or_path, str):
            file_path = frag_or_path
            file_size = get_file_size(file_path)
        else:
            file_path = frag_or_path.path
            file_size = get_frag_size(frag_or_path)

        if current_size + file_size <= max_size_bytes:
            current_chunk.append(file_path)
            current_size += file_size
        else:
            # Handle case when the first
            # file is larger than max_size_mb
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [file_path]
            current_size = file_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def get_frag_size(frag):
    # Pyarrow dataset fragment
    return sum(rg.total_byte_size for rg in frag.row_groups)


def get_file_size(file_path):
    return os.path.getsize(file_path)


def strip_trailing_sep(path: str):
    """
    Strips a path string of trailing path seperators like `/` if any.
    """
    return path.rstrip(os.path.sep)
