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
import random

import nemo_curator
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client, read_data, write_to_disk
from nemo_curator.utils.file_utils import (
    expand_outdir_and_mkdir,
    get_all_files_paths_under,
)
from nemo_curator.utils.script_utils import add_distributed_args, attach_bool_arg


def main(args):
    client = get_client(args, args.device)

    output_dir = expand_outdir_and_mkdir(args.output_data_dir)
    files = get_all_files_paths_under(args.input_data_dir)
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(files)

    dataset = DocumentDataset(
        read_data(
            files, file_type=args.input_file_type, backend="pandas", add_filename=True
        )
    )
    add_id = nemo_curator.AddId(
        args.id_field_name, id_prefix=args.id_prefix, start_index=args.starting_index
    )
    id_dataset = add_id(dataset)

    write_to_disk(
        id_dataset.df,
        output_dir,
        write_to_filename=True,
        output_type=args.output_file_type,
    )


def attach_args(
    parser=argparse.ArgumentParser(
        """
Adds unique identifiers to each document in the dataset.
Creates a new ID field with name specified by the argument
"--id-field-name" within each json.

This script essentially works by counting the total
number of documents within the dataset and then, in parallel
assigns unique sequential ids to each document in the dataset.

If a document identifier does not already exist for each document, then
these ids must be added prior to performing fuzzy/exact deduplication
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
):
    parser.add_argument(
        "--input-data-dir",
        type=str,
        default=None,
        help="Input directory consisting of .jsonl files that are accessible "
        "to all nodes. Use this for a distributed file system",
    )
    parser.add_argument(
        "--starting-index",
        type=int,
        default=None,
        help="If supplied, determines the starting index from which to start "
        "indexing the documents. By default, it is unspecified, and uses an id"
        " scheme that is fast to calculate and is not guaranteed to be ordered.",
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=None,
        help="The output directory to where the jsonl "
        "files with ids will be written. If not specified, the ids will "
        "be written in-place",
    )
    parser.add_argument(
        "--id-field-name",
        type=str,
        default="adlr_id",
        help="The name of the field that will contain the id value. "
        "Default is 'adlr_id'",
    )
    parser.add_argument(
        "--id-prefix",
        type=str,
        default="doc_id",
        help="The prefix to the id number that will be assigned to the "
        "document. When performing deduplication jointly with different"
        "datasets, it is helpful to provide a prefix that denotes that a "
        "document belongs to a particular dataset (e.g., wiki for documents"
        "that come from the wikipedia dataset)",
    )
    attach_bool_arg(
        parser,
        "shuffle",
        help_str="Shuffle the order of files before assigning IDs."
        "Useful for creating a copy dataset with different IDs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="If shuffling is specified, use this random seed to "
        "perform the random shuffling",
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
