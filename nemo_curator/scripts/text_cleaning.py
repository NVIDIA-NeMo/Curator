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

import nemo_curator
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers import UnicodeReformatter
from nemo_curator.utils.distributed_utils import get_client, read_data, write_to_disk
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir, get_batched_files
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):
    client = get_client(**ArgumentHelper.parse_client_args(args))

    # Make the output directories
    output_clean_dir = expand_outdir_and_mkdir(args.output_clean_dir)

    cleaner = nemo_curator.Modify(
        UnicodeReformatter(), text_field=args.input_text_field
    )

    for files in get_batched_files(
        args.input_data_dir,
        output_clean_dir,
        args.input_file_type,
        batch_size=args.batch_size,
    ):
        dataset = DocumentDataset(
            read_data(
                files,
                file_type=args.input_file_type,
                backend="pandas",
                add_filename=True,
            )
        )
        cleaned_dataset = cleaner(dataset)
        write_to_disk(
            cleaned_dataset.df,
            output_clean_dir,
            write_to_filename=True,
            output_type=args.output_file_type,
        )
        print(f"Finished reformatting {len(files)} files")

    print("Finished reformatting all files")


def attach_args(
    parser=argparse.ArgumentParser(
        """
Text cleaning and language filtering.

Takes as input a directory consisting of .jsonl files with one
document per line and outputs to a separate directory the text
with fixed unicode. Also performs language filtering using
the \"language\" field within each JSON object.
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
):
    argumentHelper = ArgumentHelper(parser)

    argumentHelper.add_arg_batch_size()
    argumentHelper.add_arg_input_data_dir()
    argumentHelper.add_arg_input_file_type()
    argumentHelper.add_arg_input_text_field()
    argumentHelper.add_arg_output_file_type()
    argumentHelper.add_distributed_args()
    parser.add_argument(
        "--output-clean-dir",
        type=str,
        required=True,
        help="The output directory to where the cleaned JSONL files will be written.",
    )

    return parser


def console_script():
    main(attach_args().parse_args())
