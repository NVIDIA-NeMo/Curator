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
from time import time

import cudf
import cugraph
import cugraph.dask as dcg
import cugraph.dask.comms.comms as Comms
import cupy
import dask_cudf
import numpy as np
from dask.dataframe.shuffle import shuffle as dd_shuffle
from dask.utils import M

from nemo_curator.gpu_deduplication.jaccard_utils.doc_id_mapping import (
    convert_str_pair_adlr_ids_to_int,
)
from nemo_curator.gpu_deduplication.utils import (
    enable_spilling,
    get_client,
    get_num_workers,
    parse_nc_args,
    timer,
)


def sort_adlr_id(df):
    x = df[["adlr_id_x", "adlr_id_y"]].values
    x = cupy.sort(x, axis=1)
    df["adlr_id_x"] = x[:, 0]
    df["adlr_id_y"] = x[:, 1]
    for i in ["adlr_id_x", "adlr_id_y"]:
        df[i] = df[i].astype("uint64")
    return df


def thresholding(df, threshold=0.8):
    mask = df.jaccard > threshold
    df.loc[mask, "jaccard"] = np.int8(1)
    df.loc[~mask, "jaccard"] = np.int8(0)
    return df


@timer
def run_connected_components(jaccard_pairs_path, adlr_id_path, output_path):
    Comms.initialize(p2p=True)
    df = dask_cudf.read_parquet(
        jaccard_pairs_path, blocksize="1GB", aggregate_files=True
    )
    df = df[df["jaccard"] == 1].reset_index(drop=True)

    labels_df = dask_cudf.read_parquet(adlr_id_path)
    num_nodes = len(labels_df)

    self_edge_df = labels_df[["uid"]].rename(columns={"uid": "adlr_id_x"})
    self_edge_df["adlr_id_y"] = self_edge_df["adlr_id_x"]

    df = df[["adlr_id_x", "adlr_id_y"]].astype(np.int64)
    df = dask_cudf.concat([df, self_edge_df])

    G = cugraph.MultiGraph(directed=False)
    G.from_dask_cudf_edgelist(
        df, source="adlr_id_x", destination="adlr_id_y", renumber=False
    )
    result = dcg.weakly_connected_components(G)
    del G
    max_partitions = min(32, result.npartitions)
    n_components = len(result[["labels"]].drop_duplicates(split_out=max_partitions))
    num_labels = len(result)
    print("# of groups", n_components)
    print("# of docs removed", num_labels - n_components)
    labels_df = labels_df.merge(
        result, left_on=["uid"], right_on=["vertex"], how="inner"
    )
    labels_df = labels_df[["dataset_id", "doc_id", "labels"]]
    labels_df = labels_df.rename(columns={"labels": "group"})
    labels_df = labels_df.persist()
    # Doing an inner merge above
    # should not change any rows

    assert num_nodes == len(labels_df)
    print(f"assert num_nodes:{num_nodes}==labels_df:{len(labels_df)} passed")
    labels_df.to_parquet(output_path, write_index=False)
    Comms.destroy()


def attach_args(parser=None):
    description = """Computes connected component"""
    if not parser:
        parser = parse_nc_args(description=description)

    parser.add_argument(
        "--jaccard-pairs-path",
        type=str,
        help="The directory containing the jaccard results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="The output directory to write results to",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="The cache directory to write intermediate results to",
    )
    return parser


def delete_cache_data(path):
    if "cache" not in path:
        return
    cmd = f"rm -rf {path}"
    print(cmd)
    os.system(cmd)


def write_output(ddf, output_path):
    if not isinstance(output_path, str):
        assert TypeError(f"output_path should be str. got {type(output_path)}")
    print(f"write {output_path} ...")
    ddf.to_parquet(output_path, write_index=False)


def get_unique_ids_per_partition(df):
    unique_df_ls = []
    for tag in ["x", "y"]:
        subset_df = df[[f"dataset_id_{tag}", f"doc_id_{tag}"]].drop_duplicates()
        subset_df = subset_df.rename(
            columns={f"dataset_id_{tag}": "dataset_id", f"doc_id_{tag}": "doc_id"}
        )
        unique_df_ls.append(subset_df)
    unique_df = cudf.concat(unique_df_ls, ignore_index=True)
    unique_df = unique_df.drop_duplicates()
    return unique_df


@timer
def write_dedup_parsed_adlr_id(args):
    dedup_parsed_adlr_id_path = f"{args.cache_dir}/dedup_parsed_adlr_id.parquet"
    ddf = dask_cudf.read_parquet(
        args.jaccard_pairs_path,
        columns=["adlr_id_x", "adlr_id_y"],
        blocksize="1GB",
        aggregate_files=True,
    )
    ddf = ddf.map_partitions(
        convert_str_pair_adlr_ids_to_int,
        meta={
            "dataset_id_x": "uint32",
            "doc_id_x": "int64",
            "dataset_id_y": "uint32",
            "doc_id_y": "int64",
        },
    )

    unique_docs = ddf.map_partitions(get_unique_ids_per_partition)
    unique_docs = unique_docs.drop_duplicates(split_out=ddf.npartitions // 4)
    unique_docs["uid"] = np.uint64(1)
    unique_docs["uid"] = unique_docs["uid"].cumsum()
    unique_docs["uid"] = unique_docs["uid"] - 1
    write_output(unique_docs, dedup_parsed_adlr_id_path)
    return dedup_parsed_adlr_id_path


def batched_merge_and_write(ddf, ddf_adlr_id, output_path, batch_size=32):
    total_batches = (ddf.npartitions + batch_size - 1) // batch_size
    for batch_id, offset in enumerate(range(0, ddf.npartitions, batch_size)):
        st = time()
        subset_ddf = ddf.partitions[offset : offset + batch_size]
        for tag in ["x", "y"]:
            subset_ddf = subset_ddf.merge(
                ddf_adlr_id,
                left_on=[f"dataset_id_{tag}", f"doc_id_{tag}"],
                right_on=["dataset_id", "doc_id"],
                how="inner",
                broadcast=True,
            )
            subset_ddf = subset_ddf.rename(columns={"uid": f"adlr_id_{tag}"})
            subset_ddf = subset_ddf.drop(columns=[f"dataset_id_{tag}", f"doc_id_{tag}"])

        subset_ddf = subset_ddf[["adlr_id_x", "adlr_id_y", "jaccard"]]
        output_batch_path = os.path.join(output_path, f"{batch_id}.parquet")
        subset_ddf.to_parquet(output_batch_path, write_index=False)

        et = time()
        print(f"batch_id = {batch_id}/{total_batches}, time = {et - st}", flush=True)


@timer
def write_encoded_jaccard_pair(args, client):
    dedup_parsed_adlr_id_path = f"{args.cache_dir}/dedup_parsed_adlr_id.parquet"
    output_path = f"{args.cache_dir}/encoded_jaccard_pair/"
    ddf_adlr_id = dask_cudf.read_parquet(
        dedup_parsed_adlr_id_path, blocksize="2GB", aggregate_files=True
    )
    ddf_adlr_id = ddf_adlr_id.persist()
    len(ddf_adlr_id)
    ddf = dask_cudf.read_parquet(
        args.jaccard_pairs_path,
        blocksize="256MB",
        aggregate_files=True,
    )
    ddf = ddf.map_partitions(
        convert_str_pair_adlr_ids_to_int,
        meta={
            "jaccard": "float32",
            "dataset_id_x": "uint32",
            "doc_id_x": "int64",
            "dataset_id_y": "uint32",
            "doc_id_y": "int64",
        },
    )
    num_workers = get_num_workers(client)
    batched_merge_and_write(ddf, ddf_adlr_id, output_path, num_workers)


@timer
def write_dedup_encoded_jaccard_pair(args, client):
    input_path = f"{args.cache_dir}/encoded_jaccard_pair"
    output_path = f"{args.cache_dir}/final_dedup_encoded_jaccard_pair.parquet"

    ddf = dask_cudf.read_parquet(input_path, blocksize="512MB", aggregate_files=True)
    meta = {"adlr_id_x": "uint64", "adlr_id_y": "uint64", "jaccard": "float32"}
    ddf = ddf.map_partitions(sort_adlr_id, meta=meta)
    ddf = ddf.map_partitions(thresholding, meta=meta)
    ddf = ddf.map_partitions(
        M.drop_duplicates,
        meta=ddf._meta,
        enforce_metadata=False,
        transform_divisions=False,
        align_dataframes=False,
    )
    ddf = dd_shuffle(
        ddf,
        ["adlr_id_x", "doc_id"],
        ignore_index=True,
        shuffle="tasks",
    )
    ddf = ddf.map_partitions(
        M.drop_duplicates,
        meta=ddf._meta,
        enforce_metadata=False,
        transform_divisions=False,
        align_dataframes=False,
    )

    write_output(ddf, output_path)
    return output_path


def main(args):
    description = """Takes a dataset consisting of document pairs
    and their corresponding jaccard similarity to compute connected
    components of docuements across pairs to find similar docuemnt
    after applying a given threshold. The result is a dataset
    consisting of all documents that are similar (above the threshold)
    and the component they belong to."""
    start = time()
    output_path = os.path.join(args.output_dir, "connected_components.parquet")

    client = get_client(args)
    enable_spilling()
    client.run(enable_spilling)
    adlr_id_path = write_dedup_parsed_adlr_id(args)
    write_encoded_jaccard_pair(args, client)
    jaccard_pairs_path = write_dedup_encoded_jaccard_pair(args, client)
    run_connected_components(jaccard_pairs_path, adlr_id_path, output_path)
    print(f"All done in {time()-start:.1f} seconds")


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
