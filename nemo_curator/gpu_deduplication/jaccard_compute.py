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
import time

import cudf
import dask.dataframe as dd
import numpy as np

from nemo_curator.gpu_deduplication.jaccard_utils.jaccard_similarity_utils import (
    compute_jaccard_and_create_pair_df,
)
from nemo_curator.gpu_deduplication.utils import (
    enable_spilling,
    get_client,
    get_num_workers,
    parse_nc_args,
)


def create_bins(path_dicts, max_size):
    path_dicts.sort(key=lambda x: x["str_bytes"], reverse=True)
    bins, bin_sizes = [], []
    for path_d in path_dicts:
        new_path, new_size = path_d["path"], path_d["str_bytes"]
        for i, bin_size in enumerate(bin_sizes):
            if bin_size + new_size <= max_size:
                bins[i].append(new_path)
                bin_sizes[i] += new_size
                new_size = 0
                break
        if new_size:
            bins.append([new_path])
            bin_sizes.append(new_size)
    return bins


def get_anchor_docs_and_string_size(path):
    df = cudf.read_parquet(path)
    str_bytes = df["text"].str.byte_count().sum()
    is_anchor_flag = (df["adlr_id"] == df["anchor_1_adlr_id"]) | (
        df["adlr_id"] == df["anchor_0_adlr_id"]
    )
    anchor_df = df[is_anchor_flag].reset_index(drop=True)
    return anchor_df, {"path": path, "str_bytes": str_bytes}


def compute_jaccard_on_1_partition(path):
    try:
        df = cudf.read_parquet(path)
        pair_df = compute_jaccard_and_create_pair_df(df)
    except OverflowError:
        paths = [entry.path for entry in os.scandir(os.path.join(path))]
        anchor_df_str_size_ls = [
            get_anchor_docs_and_string_size(path) for path in paths
        ]
        anchor_df = cudf.concat(
            [anchor_doc for anchor_doc, _ in anchor_df_str_size_ls], ignore_index=True
        ).drop_duplicates()
        df_str_size = [str_size for _, str_size in anchor_df_str_size_ls]
        paths = create_bins(df_str_size, np.iinfo(np.int32).max // 10)
        pair_dfs = []
        for path in paths:
            print(path)
            df = cudf.read_parquet(path).reset_index(drop=True)
            df = cudf.concat([df, anchor_df], ignore_index=True)
            pair_df = compute_jaccard_and_create_pair_df(df)
            pair_dfs.append(pair_df)
        pair_df = cudf.concat(pair_dfs, ignore_index=True)
    return pair_df


def run_jaccard_compute(shuffled_docs_path, output_final_results_path):
    print("Starting Jaccard Computation", flush=True)
    st = time.time()
    paths = [
        entry.path
        for entry in os.scandir(shuffled_docs_path)
        if not entry.path.endswith(".txt")
    ]
    meta_df = cudf.DataFrame(
        {
            "adlr_id_x": ["x"],
            "adlr_id_y": ["y"],
            "jaccard": np.float32([0.0]),
        }
    )
    result_df = dd.from_map(
        compute_jaccard_on_1_partition, paths, meta=meta_df
    ).reset_index(drop=True)

    result_df.to_parquet(
        output_final_results_path,
        write_index=False,
        write_metadata_file=False,
    )
    print(f"Jaccard Computing+Writing time: {time.time() - st:.1f} seconds")


def main(args):
    description = """Computes the Jaccard similarity between document pairs
    from partitioned parquet dataset. Result is a parquet dataset consiting of
    document id pair along with their Jaccard similarity score.
    """
    OUTPUT_PATH = args.output_dir
    shuffled_docs_path = args.shuffled_docs_path
    output_final_results_path = os.path.join(OUTPUT_PATH, "dedup_final_results.parquet")
    client = get_client(args)
    enable_spilling()
    client.run(enable_spilling)
    print(f"Num Workers = {get_num_workers(client)}", flush=True)
    print("Connected to dask cluster", flush=True)
    print("Running jaccard compute script", flush=True)

    # Run actual computation
    run_jaccard_compute(shuffled_docs_path, output_final_results_path)


def attach_args(parser=None):
    description = """Computes  jaccard similarity"""
    if not parser:
        parser = parse_nc_args(description=description)

    parser.add_argument(
        "--shuffled-docs-path",
        type=str,
        help="The directory containing the shuffled documents",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="The output directory to write results to",
    )
    return parser


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
