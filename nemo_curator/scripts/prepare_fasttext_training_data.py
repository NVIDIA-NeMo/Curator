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
import csv

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers import FastTextLabelModifier
from nemo_curator.modules import Modify
from nemo_curator.utils.distributed_utils import get_client, read_data
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.utils.script_utils import ArgumentHelper


def sample_rows(df, n, seed):
    samples = df.sample(frac=n / len(df) + 0.05, random_state=seed)

    return samples.head(n=n, compute=False)


def main(args):
    client = get_client(**ArgumentHelper.parse_client_args(args))
    # Get local path
    files = list(get_all_files_paths_under(args.input_data_dir))
    raw_data = read_data(files, file_type="jsonl", backend="pandas")
    dataset = DocumentDataset(raw_data)
    text_field = args.input_json_field

    # fastText requires each document to be prepended with a special label for training
    preprocessing = Modify(FastTextLabelModifier(args.label), text_field=text_field)
    labeled_data = preprocessing(dataset)

    samples = sample_rows(labeled_data.df, args.output_num_samples, args.seed)

    samples[text_field].to_csv(
        args.output_train_file,
        single_file=True,
        encoding="utf-8",
        header=False,
        index=False,
        quoting=csv.QUOTE_NONE,
        sep="\n",
    )
    client.close()


def attach_args(
    parser=argparse.ArgumentParser(
        """
Prepare data for training skip-gram classifier with FastText.

Takes as input a directory of .jsonl files, and writes an output
file of samples prepared in order to train a skip-gram classifier
with FastText.
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
):
    argumentHelper = ArgumentHelper(parser)

    argumentHelper.add_arg_input_data_dir()
    argumentHelper.add_arg_log_dir(default="./log/prepare_filter_data")
    argumentHelper.add_arg_output_train_file(
        help="The output file containing prepared samples to train a "
        "skip-gram classifier with FastText."
    )
    argumentHelper.add_arg_seed(
        help="The random seed to use for sampling from the dataset."
    )
    argumentHelper.add_distributed_args()
    parser.add_argument(
        "--input-json-field",
        type=str,
        default="text",
        help="The input field within each JSON object on which the filter will "
        'operate. By default, the filter will operate on the "text" '
        'field, but other fields can be specified, such as "url" or "id".',
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        required=True,
        help="The label to be used at the beginning of each sample "
        'in the output file. For example, "__label__hq" could be '
        "used for the high-quality (positive) samples.",
    )

    return parser


def console_script():
    main(attach_args().parse_args())
