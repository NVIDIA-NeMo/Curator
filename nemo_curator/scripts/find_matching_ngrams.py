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
import pickle

import nemo_curator
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client, read_data
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.utils.script_utils import add_distributed_args


def main(args):
    client = get_client(args, args.device)

    # Each rank read in the task data
    with open(args.input_task_ngrams, "rb") as fp:
        task_ngrams = pickle.load(fp)

    decontaminator = nemo_curator.TaskDecontamination(
        [], text_field=args.input_text_field, max_ngram_size=args.max_ngram_size
    )

    files = get_all_files_paths_under(args.input_data_dir)
    dataset = DocumentDataset(
        read_data(files, file_type=args.input_file_type, backend="pandas")
    )

    result = decontaminator.find_matching_ngrams(task_ngrams, dataset).compute()
    print(f"Found a total of {len(result['matched-ngrams'])} matching n-grams")

    output = {
        "matched-ngrams": result["matched-ngrams"],
        "ngrams-freq": result["ngrams-freq"],
        "max-ngram-size": args.max_ngram_size,
        "min-ngram-size": args.min_ngram_size,
    }
    with open(args.output_matched_ngram_data, "wb") as fp:
        pickle.dump(output, fp)


def attach_args(
    parser=argparse.ArgumentParser(
        """
    Searches for matching task n-grams in the input dataset
    and writes out a list of n-grams that were found.
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
        "--input-text-field",
        type=str,
        default="text",
        help="The name of the field within each datapoint object of the input "
        "file that contains the text.",
    )
    parser.add_argument(
        "--output-matched-ngram-data",
        type=str,
        default=None,
        help="Output dictionary that contains the output matched n-grams "
        "and the frequency of their matches, min-ngram size, max-ngram "
        "size and the frequencies of n-gram sizes. All of these data will be "
        "used by remove_matching_grams for which this program is a prequisite",
    )
    parser.add_argument(
        "--input-task-ngrams",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--max-ngram-size",
        type=int,
        default=13,
        help="The maximum n-gram size to consider within the dataset",
    )
    parser.add_argument(
        "--min-ngram-size",
        type=int,
        default=8,
        help="The minimum n-gram size to consider within the datset",
    )
    parser.add_argument(
        "--input-file-type",
        type=str,
        default="jsonl",
        help="File type of the dataset to be read in. Supported file formats"
        " include 'jsonl' (default), 'pickle', or 'parquet'.",
    )

    parser = add_distributed_args(parser)

    return parser


def console_script():
    main(attach_args().parse_args())
