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

import argparse
import os
import time

import dask.dataframe as dd
from dask.distributed import Client

from nemo_curator.modules.fuzzy_dedup._mapbuckets import _MapBuckets
from nemo_curator.utils.distributed_utils import get_client, get_num_workers
from nemo_curator.utils.fuzzy_dedup_utils.io_utils import (
    get_bucket_ddf_from_parquet_path,
    get_text_ddf_from_json_path_with_blocksize,
)
from nemo_curator.utils.script_utils import ArgumentHelper


def get_anchor_and_output_map_info(  # noqa: PLR0913
    input_data_paths: list[str],
    input_bucket_path: str,
    text_ddf_blocksize: int,
    num_files: int,
    num_workers: int,
    shuffle_type: str,
    input_bucket_field: str,
    input_id_field: str,
    input_text_field: str,
    input_meta: str,
) -> dd.DataFrame:
    """
    Get anchor docs with bucket info
    Args:
        input_data_paths: list of paths to input data
        input_bucket_path: path to input buckets
        text_ddf_blocksize: blocksize for text ddf
        num_files: number of files to read
        num_workers: number of workers
        shuffle_type: type of shuffle to use
    Returns:
        ddf_anchor_docs_with_bk
    """
    ddf_text = get_text_ddf_from_json_path_with_blocksize(
        input_data_paths=input_data_paths,
        num_files=num_files,
        blocksize=text_ddf_blocksize,
        id_column=input_id_field,
        text_column=input_text_field,
        input_meta=input_meta,
    )
    ddf_bk = get_bucket_ddf_from_parquet_path(input_bucket_path=input_bucket_path, num_workers=num_workers)
    map_buckets = _MapBuckets(
        id_fields=["dataset_id", "doc_id"],
        bucket_field=input_bucket_field,
        text_field=input_text_field,
    )
    return map_buckets.map_buckets_with_anchors(documents_df=ddf_text, buckets_df=ddf_bk, shuffle_type=shuffle_type)


def attach_args() -> argparse.ArgumentParser:
    description = """
    Takes the buckets generated from minhashes and uses
    document length information to create a coarse mapping of mapping multiple
    buckets to a logical partition by using a modified bin packing algorithm.
    """
    parser = argparse.ArgumentParser(
        description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    arg_helper = ArgumentHelper(parser)

    arg_helper.parse_gpu_dedup_args()

    arg_helper.add_arg_input_meta()
    arg_helper.add_arg_output_dir()
    arg_helper.add_arg_text_ddf_blocksize()
    parser.add_argument(
        "--input-bucket-dir",
        type=str,
        help="The directory containing bucket information files.",
    )
    parser.add_argument(
        "--input-bucket-field",
        type=str,
        default="_bucket_id",
        help="Name of the column containing minhashes.",
    )
    parser.add_argument(
        "--shuffle-type",
        type=str,
        default="tasks",
        help="Type of shuffle to use before writing to Parquet.",
    )

    return parser


def jaccard_get_output_map_workflow(  # noqa: PLR0913
    client: Client,
    input_data_paths: list[str],
    input_bucket_path: str,
    output_anchor_docs_with_bk_path: str,
    text_ddf_blocksize: int,
    num_files: int,
    shuffle_type: str,
    input_bucket_field: str,
    input_id_field: str,
    input_text_field: str,
    input_meta: str,
) -> None:
    """
    Workflow for jaccard shuffle
    Args:
        client: dask client
        input_data_paths: list of paths to input data
        input_bucket_path: path to input buckets
        output_anchor_docs_with_bk_path: path to save anchor docs with bucket info
        text_ddf_blocksize: blocksize for text ddf
        num_files: number of files to read
        parts_per_worker: number of parts per worker
        shuffle_type: type of shuffle to use before writing to parquet
    """
    num_workers = get_num_workers(client)
    ddf_anchor_docs_with_bk = get_anchor_and_output_map_info(
        input_data_paths,
        input_bucket_path,
        text_ddf_blocksize,
        num_files,
        num_workers,
        shuffle_type,
        input_bucket_field,
        input_id_field,
        input_text_field,
        input_meta=input_meta,
    )
    ddf_anchor_docs_with_bk.to_parquet(
        output_anchor_docs_with_bk_path,
        write_index=False,
    )


def main(args: argparse.Namespace) -> None:
    input_data_paths = args.input_data_dirs
    input_bucket_path = args.input_bucket_dir
    output_anchor_docs_with_bk_path = os.path.join(args.output_dir, "anchor_docs_with_bk.parquet")
    client = get_client(**ArgumentHelper.parse_client_args(args))
    print(f"Num Workers = {get_num_workers(client)}", flush=True)
    print("Connected to dask cluster", flush=True)
    print("Running jaccard map buckets script", flush=True)
    print(f"Args = {args}")
    st = time.time()
    jaccard_get_output_map_workflow(
        client,
        input_data_paths,
        input_bucket_path,
        output_anchor_docs_with_bk_path,
        args.text_ddf_blocksize,
        args.num_files,
        args.shuffle_type,
        args.input_bucket_field,
        args.input_json_id_field,
        args.input_json_text_field,
        args.input_meta,
    )
    et = time.time()
    print(f"Bucket Mapping time taken = {et - st} s")


def console_script() -> None:
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
