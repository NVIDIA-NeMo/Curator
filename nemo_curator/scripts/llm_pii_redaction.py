# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import time
from pathlib import Path

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.llm_pii_modifier import LLMPiiModifier
from nemo_curator.modules.modify import Modify
from nemo_curator.utils.distributed_utils import get_client, read_data, write_to_disk
from nemo_curator.utils.file_utils import get_batched_files
from nemo_curator.utils.llm_pii_utils import PII_LABELS, SYSTEM_PROMPT
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):
    """Main function that performs LLM-based PII de-identification given a batch of files"""

    print("Beginning PII job")
    start_time = time.time()
    Path(args.output_data_dir).mkdir(parents=True, exist_ok=True)

    modifier = LLMPiiModifier(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        system_prompt=args.system_prompt,
        pii_labels=args.pii_labels,
        language=args.language,
    )

    for file_names in get_batched_files(
        args.input_data_dir, args.output_data_dir, args.input_file_type, args.n_workers
    ):
        source_data = read_data(
            file_names,
            file_type=args.input_file_type,
            backend="pandas",
            add_filename=True,
        )
        dataset = DocumentDataset(source_data)

        modify = Modify(modifier, text_field=args.text_field)
        modified_dataset = modify(dataset)
        write_to_disk(
            modified_dataset.df,
            args.output_data_dir,
            write_to_filename=True,
            output_type=args.output_file_type,
        )

    end_time = time.time()
    print("Total time taken for PII job: %0.3f seconds" % (end_time - start_time))


def attach_args(
    parser=argparse.ArgumentParser(
        """
        Main driver script for applying LLM-based PII redaction on documents. Inputs are in the input-data-dir directory.
        This script will then perform PII detection and de-identification on each document within the corpus.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
):
    argumentHelper = ArgumentHelper(parser)

    argumentHelper.add_arg_input_data_dir(help="Directory containing the input files.")
    argumentHelper.add_arg_input_file_type()
    argumentHelper.add_arg_output_data_dir(
        help="The output directory to where redacted documents will be written."
    )
    argumentHelper.add_arg_output_file_type()
    argumentHelper.add_distributed_args()

    parser.add_argument(
        "--base_url",
        type=str,
        required=True,
        help="The base URL for the user's NIM",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="The API key for the user's NIM, if needed.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta/llama-3.1-70b-instruct",
        help="The model to use for the LLM.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=SYSTEM_PROMPT,
        help="The system prompt to feed into the LLM.",
    )
    parser.add_argument(
        "--pii_labels",
        type=str,
        default=PII_LABELS,
        help="Comma separated list of PII entity types. None implies all supported types.",
    )
    argumentHelper.add_arg_language(help="Language of input documents.")

    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="The input field within each JSONL or CSV object on which the PII redactor will "
        'operate. By default, the redactor will operate on the "text" '
        'field but other fields can be specified such as "url" or "id".',
    )

    return parser


def console_script():
    parser = attach_args()
    args = parser.parse_args()
    client = get_client(**ArgumentHelper.parse_client_args(args))
    if not args.n_workers:
        args.n_workers = len(client.scheduler_info()["workers"])
    main(args)


if __name__ == "__main__":
    console_script()
