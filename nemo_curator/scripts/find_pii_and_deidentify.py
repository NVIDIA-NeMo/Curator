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
from nemo_curator.utils.distributed_utils import get_client, read_data, write_to_disk
from nemo_curator.utils.file_utils import get_batched_files
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):
    """Main function that performs PII de-identification given a batch of files"""
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
        "Total time taken for PII job: %0.3f seconds" % (end_time - start_time)
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
    argumentHelper = ArgumentHelper(parser)

    argumentHelper.add_arg_batch_size(
        default=2000, help="The batch size for processing multiple texts together."
    )
    argumentHelper.add_arg_input_data_dir(help="Directory containing the input files.")
    argumentHelper.add_arg_input_file_type()
    argumentHelper.add_arg_language(help="Language of input documents.")
    argumentHelper.add_arg_output_data_dir(
        help="The output directory to where redacted documents will be written."
    )
    argumentHelper.add_arg_output_file_type()
    argumentHelper.add_distributed_args()

    parser.add_argument(
        "--anonymize-action",
        type=str,
        default="replace",
        help="Anonymization action. Choose from among: redact, hash, mask, and replace.",
    )
    parser.add_argument(
        "--chars-to-mask",
        type=int,
        default=100,
        help="The number of characters to mask. Only applicable if anonymize action is mask.",
    )
    parser.add_argument(
        "--hash-type",
        type=str,
        default=None,
        help="The hash type. Choose from among: sha256, sha512, or md5.",
    )
    parser.add_argument(
        "--masking-char",
        type=str,
        default="*",
        help="The masking character. Only applicable if anonymize action is mask.",
    )
    parser.add_argument(
        "--new-value",
        type=str,
        default=None,
        help="The new value to replace with. Only applicable if anonymize action is replace.",
    )
    parser.add_argument(
        "--supported-entities",
        type=str,
        default=None,
        help="Comma separated list of PII entity types. None implies all supported types.",
    )
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
