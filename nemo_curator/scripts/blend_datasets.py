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

import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client, read_data, write_to_disk
from nemo_curator.utils.file_utils import (
    expand_outdir_and_mkdir,
    get_all_files_paths_under,
)
from nemo_curator.utils.script_utils import add_distributed_args, attach_bool_arg


def main(args):
    client = get_client(args, args.device)

    out_dir = expand_outdir_and_mkdir(args.output_data_dir)

    input_dirs = args.input_data_dirs.split(",")
    weights = [float(weight) for weight in args.weights.split(",")]

    datasets = [
        DocumentDataset(
            read_data(
                get_all_files_paths_under(path),
                file_type=args.input_file_type,
                backend="pandas",
            )
        )
        for path in input_dirs
    ]

    output_dataset = nc.blend_datasets(args.target_samples, datasets, weights)

    if args.shuffle:
        shuffle = nc.Shuffle(seed=args.seed)
        output_dataset = shuffle(output_dataset)

    write_to_disk(output_dataset.df, out_dir, output_type=args.output_file_type)

    client.close()


def attach_args(
    parser=argparse.ArgumentParser(
        """
Blends a collection of datasets together based on certain weights.

It takes as input a comma-separated list of dataset directories, the
corresponding weights that should be associated with each datatset,
and the target number of samples to aggregate from across all the datasets.
The file shards of the resulting dataset are not guaranteed to be even
or reflect the original dataset(s).

A blend is created from these datasets and saved to the specified output directory.
Optionally, the user can choose to shuffle this dataset as well.
  """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
):
    parser.add_argument(
        "--input-data-dirs",
        type=str,
        default=None,
        help="Comma-separated list of directories consisting of dataset "
        "files that are accessible to all nodes.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Comma-separated list of floating-point weights corresponding "
        "to each dataset passed in --input-data-dirs",
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=None,
        help="The output directory to where the blended dataset is"
        "retained during filtering will be written. If this argument "
        "is not specified, then the document scores from the "
        "filter(s) will be written to the document meta data in place",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=10000,
        help="The number of samples to be included in the output dataset."
        " There may be more samples in order to accurately reflect the "
        "weight balance, but there will never be less",
    )
    attach_bool_arg(
        parser,
        "shuffle",
        default=False,
        help_str="Shuffles the dataset after blending",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If specified, the random seed used for shuffling.",
    )
    parser.add_argument(
        "--input-file-type",
        type=str,
        default="jsonl",
        help="File type of the dataset to be read in. Supported file formats"
        " include 'jsonl' (default), 'pickle', or 'parquet'.",
    )
    parser.add_argument(
        "--output-file-type",
        type=str,
        default="jsonl",
        help="File type the dataset will be written to. Supported file formats"
        " include 'jsonl' (default), 'pickle', or 'parquet'.",
    )

    parser = add_distributed_args(parser)

    return parser


def console_script():
    main(attach_args().parse_args())
