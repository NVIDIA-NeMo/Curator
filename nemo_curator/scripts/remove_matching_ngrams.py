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
from nemo_curator.utils.distributed_utils import get_client, read_data, write_to_disk
from nemo_curator.utils.file_utils import (
    expand_outdir_and_mkdir,
    get_all_files_paths_under,
    get_batched_files,
)
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):
    client = get_client(**ArgumentHelper.parse_client_args(args))

    output_tdd_dir = expand_outdir_and_mkdir(args.output_task_deduped_dir)
    output_rm_doc_dir = None
    if args.output_removed_doc_dir is not None:
        output_rm_doc_dir = expand_outdir_and_mkdir(args.output_removed_doc_dir)

    # Each rank read in the task data
    print(f"Reading in matched n-grams from {args.input_matched_ngrams}")
    with open(args.input_matched_ngrams, "rb") as fp:
        matched_ngram_data = pickle.load(fp)

    # Unpack the results from find_matched_ngrams
    matched_ngrams = matched_ngram_data["matched-ngrams"]
    ngrams_freq = matched_ngram_data["ngrams-freq"]
    max_ngram_size = matched_ngram_data["max-ngram-size"]

    decontaminator = nemo_curator.TaskDecontamination(
        [],
        text_field=args.input_text_field,
        max_ngram_size=max_ngram_size,
        max_matches=args.match_threshold,
        max_splits=args.max_document_splits,
        removed_dir=output_rm_doc_dir,
    )

    files = list(get_all_files_paths_under(args.input_data_dir))
    for files in get_batched_files(
        args.input_data_dir,
        output_tdd_dir,
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
        decontaminated_dataset = decontaminator.remove_matching_ngrams(
            matched_ngrams, ngrams_freq, dataset
        )
        write_to_disk(
            decontaminated_dataset.df,
            output_tdd_dir,
            write_to_filename=True,
            output_type=args.output_file_type,
        )
        print(f"Finished decontaminating {len(files)} files")

    print("Finished decontaminating all files")


def attach_args(
    parser=argparse.ArgumentParser(
        """
 Using the matching n-grams find by
 nemo_curator/scripts/find_matching_ngrams.py
 (provided by the argument --input-matched-ngrams),
 passes over all documents and removes matching n-grams from the corpus by
 splitting documents containing the match. If a document is split more than
 --max-splits times, it is removed from the corpus.
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
        "--input-matched-ngrams",
        type=str,
        default=None,
        required=True,
        help="Input dictionary (.pkl file) that contains matched "
        "n-gram data from the find_matching_ngrams code.",
    )
    parser.add_argument(
        "--match-threshold",
        type=int,
        default=10,
        help="A threshold that determines if a matched n-gram will be "
        "considered for removal in remove_matching_ngrams. N-grams that "
        "exceed this number of matches in the training dataset will not be "
        "considered during the removal stage.",
    )
    parser.add_argument(
        "--max-document-splits",
        type=int,
        default=10,
        help="A threshold used to determine if a document should be removed "
        "from the corpus if it is split more than "
        "--max-document-splits number of times.",
    )
    parser.add_argument(
        "--output-removed-doc-dir",
        type=str,
        default=None,
        help="Output directory to where removed documents will be written. "
        "Documents will be removed from the corpus if they are split more "
        "than --max-document-splits number of times, or if the user specifies "
        "that they be removed via the flag --remove-split-docs.",
    )
    parser.add_argument(
        "--output-task-deduped-dir",
        type=str,
        default=None,
        required=True,
        help="Output directory to where task-deduplicated (split) "
        "documents will be written.",
    )

    return parser


def console_script():
    main(attach_args().parse_args())
