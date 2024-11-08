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

import dask_cudf

from nemo_curator import BucketsToEdges
from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.utils.distributed_utils import get_client, get_num_workers
from nemo_curator.utils.script_utils import ArgumentHelper


def attach_args():
    description = """
    Takes the buckets generated from minhashes and converts
    them into an edge list for the connected components algorithm. This is done by
    assuming all documents in the same bucket are similar.
    """
    parser = argparse.ArgumentParser(
        description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argumentHelper = ArgumentHelper(parser)

    argumentHelper.parse_gpu_dedup_args()
    parser.add_argument(
        "--input-bucket-dir",
        type=str,
        help="The directory containing anchor_docs_with_bk files.",
    )
    parser.add_argument(
        "--input-bucket-field",
        type=str,
        default="_bucket_id",
        help="Name of the column containing the bucket ID.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory to write results.",
    )
    return parser


def main(args):
    logger = create_logger(
        rank=0,
        log_file=os.path.join(args.log_dir, "rank_000.log"),
        name="buckets_to_cc_log",
    )

    input_bucket_path = args.input_bucket_dir
    OUTPUT_PATH = args.output_dir

    client = get_client(**ArgumentHelper.parse_client_args(args))
    logger.info(f"Client Created {client}")
    logger.info(f"Num Workers = {get_num_workers(client)}")
    logger.info(
        "Running buckets -> EdgeList for CC",
    )

    buckets_to_edges = BucketsToEdges(
        cache_dir=OUTPUT_PATH,
        id_fields=["dataset_id", "doc_id"],
        str_id_name=args.input_json_id_field,
        bucket_field=args.input_bucket_field,
        logger=logger,
    )
    st = time.time()
    buckets_df = DocumentDataset(
        dask_cudf.read_parquet(input_bucket_path, split_row_groups=False)
    )
    _ = buckets_to_edges(buckets_df)
    et = time.time()
    logger.info(f"Bucket to Edges conversion took = {et-st} s")


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
