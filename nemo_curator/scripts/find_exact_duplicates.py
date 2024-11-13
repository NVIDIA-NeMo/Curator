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

from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.modules import ExactDuplicates
from nemo_curator.utils.distributed_utils import get_client, read_data
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.utils.fuzzy_dedup_utils.io_utils import strip_trailing_sep
from nemo_curator.utils.script_utils import ArgumentHelper


def pre_imports():
    import cudf  # noqa: F401


def main(args):
    logger = create_logger(
        rank=0, log_file=os.path.join(args.log_dir, "rank_000.log"), name="exact_dedup"
    )
    logger.info(f"Starting workflow with args:\n {args}")

    assert args.hash_method == "md5", "Currently only md5 hash is supported"
    client = get_client(**ArgumentHelper.parse_client_args(args))
    logger.info(f"Client Created {client}")
    if args.device == "gpu":
        client.run(pre_imports)
        logger.info("Pre imports complete")

    data_paths = args.input_data_dirs
    id_field = args.input_json_id_field
    text_field = args.input_json_text_field
    num_files = args.num_files
    t0 = time.time()
    dfs = []
    for data_path in data_paths:
        data_path = strip_trailing_sep(data_path)
        if num_files is not None and num_files <= 0:
            logger.info(f"Processed {num_files}... quitting")
            break
        files = get_all_files_paths_under(root=data_path, recurse_subdirectories=False)
        files = [f for f in files if f.endswith(".jsonl")]
        df = read_data(
            files[:num_files] if num_files else files,
            file_type="jsonl",
            backend="pandas" if args.device != "gpu" else "cudf",
            files_per_partition=args.files_per_partition,
            add_filename=False,
        )[[id_field, text_field]]
        if num_files is not None:
            num_files -= len(files)
        dfs.append(df)
        logger.info(f"Lazy read complete for {dfs[-1].npartitions} partitions")

    input_df = dask_cudf.concat(dfs, ignore_unknown_divisions=True)
    exact_dups = ExactDuplicates(
        logger=logger,
        id_field=id_field,
        text_field=text_field,
        hash_method=args.hash_method,
        profile_dir=args.profile_path,
        cache_dir=args.output_dir,
    )
    exact_dups(dataset=DocumentDataset(input_df))
    logger.info(
        f"Exact deduplication computation across datasets took {time.time() - t0}s complete at {args.output_dir}"  # noqa:E501
    )


def attach_args():
    description = "Compute exact duplicates in a given dataset."
    parser = argparse.ArgumentParser(
        description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argumentHelper = ArgumentHelper(parser)

    argumentHelper.parse_gpu_dedup_args()

    argumentHelper.add_arg_output_dir(
        help="Output directory where duplicate docs will be written. "
        "Each file is a Pickle file that contains a dictionary of NumPy arrays. "
        "The keys are the document IDs and the values are the duplicate documents.",
        required=True,
    )
    parser.add_argument(
        "--hash-method",
        type=str,
        default="md5",
        help="Hash method to use for exact deduplication.",
    )

    return parser


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
