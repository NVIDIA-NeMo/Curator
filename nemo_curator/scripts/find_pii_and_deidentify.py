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
import logging
import time
from pathlib import Path

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modules.modify import Modify

# from nemo_curator.pii.algorithm import DEFAULT_LANGUAGE
from nemo_curator.utils.distributed_utils import get_client, read_data, write_to_disk
from nemo_curator.utils.file_utils import get_batched_files
from nemo_curator.utils.script_utils import add_distributed_args


def main(args):
    """Main function that performs PII de-identifcation given a batch of files"""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        level=logging.DEBUG,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.debug("Beginning PII job")
    start_time = time.time()
    Path(args.output_data_dir).mkdir(parents=True, exist_ok=True)

    supported_entities = (
        args.supported_entities.split(",") if args.supported_entities else None
    )

    modifier = PiiModifier(
        language=args.language,
        supported_entities=supported_entities,
        anonymize_action=args.anonymize_action,
        hash_type=args.hash_type,
        chars_to_mask=args.chars_to_mask,
        masking_char=args.masking_char,
        new_value=args.new_value,
        batch_size=args.batch_size,
        device=args.device,
    )

    for file_names in get_batched_files(
        args.input_data_dir, args.output_data_dir, args.input_file_type, args.n_workers
    ):
        logging.info("Reading input files....")
        source_data = read_data(
            file_names,
            file_type=args.input_file_type,
            backend="pandas",
            add_filename=True,
        )
        dataset = DocumentDataset(source_data)
        logging.debug(f"Dataset has {source_data.npartitions} partitions")

        modify = Modify(modifier)
        modified_dataset = modify(dataset)
        write_to_disk(
            modified_dataset.df,
            args.output_data_dir,
            write_to_filename=True,
            output_type=args.output_file_type,
        )

    end_time = time.time()
    logging.debug(
        "Total time taken in PII job: %0.3f seconds" % (end_time - start_time)
    )


def attach_args(
    parser=argparse.ArgumentParser(
        """
        Main driver script for applying PII redaction on documents. Inputs are in the input-data-dir directory.
        This script will then perform PII detection and de-identification on each document within the corpus.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
):

    parser.add_argument(
        "--language",
        type=str,
        default="en",
        required=False,
        help="Language of input documents",
    )

    parser.add_argument(
        "--supported-entities",
        type=str,
        default=None,
        required=False,
        help="Comma separated list of PII entity types. None implies all supported types",
    )

    parser.add_argument(
        "--anonymize-action",
        type=str,
        default="replace",
        required=False,
        help="Anonymization action. Choose from among: redact, hash, mask and replace",
    )

    parser.add_argument(
        "--hash-type",
        type=str,
        default=None,
        required=False,
        help="The hash type. Choose from among: sha256, sha512 or md5",
    )

    parser.add_argument(
        "--chars-to-mask",
        type=int,
        default=100,
        required=False,
        help="The number of characters to mask. Only applicable if anonymize action is mask",
    )

    parser.add_argument(
        "--masking-char",
        type=str,
        default="*",
        required=False,
        help="The masking character. Only applicable if anonymize action is mask",
    )

    parser.add_argument(
        "--new-value",
        type=str,
        default=None,
        required=False,
        help="The new value to replace with. Only applicable if anonymize action is replace",
    )

    parser.add_argument(
        "--input-data-dir",
        type=str,
        default=None,
        required=True,
        help="Directory containing the input files",
    )

    parser.add_argument(
        "--input-file-type",
        type=str,
        default="jsonl",
        required=True,
        choices=["jsonl", "csv", "text"],
        help="The input file type (only jsonl is currently supported)",
    )

    parser.add_argument(
        "--output-file-type",
        type=str,
        default="jsonl",
        required=True,
        choices=["jsonl", "csv", "text"],
        help="The output file type (only jsonl is currently supported)",
    )

    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="The input field within each JSONL or CSV object on which the PII redactor will "
        "operate. By default, the redactor will operate on the 'text' "
        "field but other fields can be specified such as 'url' or 'id'.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="The batch size for processing multiple texts together.",
    )

    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=None,
        required=True,
        help="The output directory to where redacted documents will be written.",
    )

    return parser


def console_script():
    arguments = add_distributed_args(attach_args()).parse_args()
    client = get_client(arguments, arguments.device)
    if not arguments.n_workers:
        arguments.n_workers = len(client.scheduler_info()["workers"])
    main(arguments)


if __name__ == "__main__":
    console_script()
