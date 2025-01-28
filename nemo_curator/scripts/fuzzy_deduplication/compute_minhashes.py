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

from nemo_curator import MinHash
from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.utils.distributed_utils import (
    get_client,
    performance_report_if,
    read_data,
)
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.utils.fuzzy_dedup_utils.io_utils import strip_trailing_sep
from nemo_curator.utils.script_utils import ArgumentHelper


def pre_imports():
    import cudf  # noqa: F401


def main(args):
    logger = create_logger(
        rank=0, log_file=os.path.join(args.log_dir, "rank_000.log"), name="minhash_log"
    )
    logger.info(f"Starting workflow with args:\n {args}")

    assert args.hash_bytes in {4, 8}, "Currently only 32bit/64bit hashes are supported"
    assert args.device == "gpu"

    client = get_client(**ArgumentHelper.parse_client_args(args))
    logger.info(f"Client Created {client}")
    client.run(pre_imports)
    logger.info("Pre imports complete")

    data_paths = args.input_data_dirs
    id_field = args.input_json_id_field
    text_field = args.input_json_text_field
    num_files = args.num_files

    minhasher = MinHash(
        seed=args.seed,
        num_hashes=args.minhash_length,
        char_ngrams=args.char_ngram,
        use_64bit_hash=False if args.hash_bytes == 4 else True,
        logger=logger,
        id_field=id_field,
        text_field=text_field,
    )

    t0 = time.time()
    for data_path in data_paths:
        print(f"Computing minhashes for {data_path}", flush=True)
        data_path = strip_trailing_sep(data_path)
        if num_files is not None and num_files <= 0:
            print(f"Processed {args.num_files}... quitting")
            break

        files = get_all_files_paths_under(root=data_path, recurse_subdirectories=False)
        files = [f for f in files if f.endswith(".jsonl")]
        df = read_data(
            files[:num_files] if num_files else files,
            file_type="jsonl",
            backend="cudf",
            files_per_partition=args.files_per_partition,
            add_filename=False,
            input_meta=args.input_meta,
        )[[id_field, text_field]]

        if num_files is not None:
            num_files -= len(files)

        res = minhasher(DocumentDataset(df)).df
        logger.info(
            f"Lazy minhash generation complete for {res.npartitions} partitions"
        )
        logger.info(f"Starting execution for {data_path}")
        write_path = os.path.join(
            args.output_minhash_dir, os.path.basename(data_path), "minhashes.parquet"
        )

        t1 = time.time()
        with performance_report_if(
            args.profile_path, f"{os.path.basename(data_path)}-minhash-profile.html"
        ):
            res.to_parquet(write_path, write_index=False)
        logger.info(
            f"Minhash computation for f{data_path} took {time.time() - t1}s complete at {write_path}"  # noqa:E501
        )
    logger.info(
        f"Minhash computation across datasets took {time.time() - t0}s complete at {args.output_minhash_dir}"  # noqa:E501
    )


def attach_args():
    description = """
    Computes minhash signatures from an input directory of documents
    contained within JSONL files. For each document, a DataFrame of document IDs
    and minhash signatures is created. This DataFrame is written to file after processing.
    """
    parser = argparse.ArgumentParser(
        description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argumentHelper = ArgumentHelper(parser)

    argumentHelper.parse_gpu_dedup_args()

    argumentHelper.add_arg_minhash_length()
    argumentHelper.add_arg_seed(
        help="Random seed used for intializing the hash "
        "functions used to compute the minhashes."
    )
    argumentHelper.add_arg_input_meta()
    parser.add_argument(
        "--char-ngram",
        type=int,
        default=24,
        help="The number of consecutive characters to include in a sliding "
        "window when creating the document shingles for computing "
        "minhash signatures.",
    )
    parser.add_argument(
        "--hash-bytes",
        type=int,
        default=4,
        help="Number of bytes per computed minhash. "
        "Default is an unsigned 32-bit integer.",
    )
    parser.add_argument(
        "--output-minhash-dir",
        type=str,
        required=True,
        help="Output directory where minhashes will be written. "
        "Each file is a Parquet file that contains two series, the document IDs, "
        "and a series of lists, each list denoting the minhash signature for that document ID.",
    )

    return parser


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
