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
import json
import os
from functools import partial
from typing import Any

from docbuilder import EmailsDownloader, EmailsIterator
from filters import FilterEmailsWithLongBody, FilterEmptyEmails
from modifiers import AddPeriod, AddSystemPrompt

from nemo_curator import ScoreFilter, Sequential
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modifiers.unicode_reformatter import UnicodeReformatter
from nemo_curator.modules.modify import Modify
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR_PATH, "data")
DATASET_URL = "https://huggingface.co/datasets/neelblabla/enron_labeled_emails_with_subjects-llama2-7b_finetuning/raw/main/prompts_train.csv"


def download_and_convert_to_jsonl() -> str:
    """
    Downloads the emails dataset and converts it to JSONL format.

    Returns:
        str: The path to the JSONL file.
    """

    # Download the dataset in raw format and convert it to JSONL.
    downloader = EmailsDownloader(DATA_DIR)
    output_path = os.path.join(DATA_DIR, "emails.jsonl")
    raw_fp = downloader.download(DATASET_URL)

    iterator = EmailsIterator()

    # Parse the raw data and write it to a JSONL file.
    with open(output_path, "w") as f:
        for record in iterator.iterate(raw_fp):
            json_record = json.dumps(record, ensure_ascii=False)
            f.write(json_record + "\n")

    return output_path


def redact_pii(dataset: DocumentDataset, text_field) -> DocumentDataset:
    """
    Redacts personally identifiable information (PII) from a given dataset.

    Args:
        dataset (DocumentDataset): The dataset containing documents with PII.

    Returns:
        DocumentDataset: The redacted dataset with PII replaced by a generic value.
    """
    redactor = Modify(
        PiiModifier(
            supported_entities=[
                "ADDRESS",
                "EMAIL_ADDRESS",
                "LOCATION",
                "PERSON",
                "URL",
                "PHONE_NUMBER",
            ],
            anonymize_action="replace",
            device="cpu",
        ),
        text_field=text_field,
    )
    return redactor(dataset)


def run_curation_pipeline(args: Any, jsonl_fp: str) -> str:
    """
    Run the curation pipeline on the dataset.

    Args:
        args (Any): Command-line arguments.
        jsonl_fp (str): The path to the uncurated JSONL file.

    Returns:
        str: The path to the curated JSONL file.
    """
    client = get_client(**ArgumentHelper.parse_client_args(args))
    print(f"    Running the curation pipeline on '{jsonl_fp}'...")
    orig_dataset = DocumentDataset.read_json(jsonl_fp, add_filename=True)
    dataset = orig_dataset

    redact_pii_subject = partial(redact_pii, text_field="subject")
    redact_pii_body = partial(redact_pii, text_field="body")

    curation_steps = Sequential(
        [
            #
            # Unify the text encoding to Unicode.
            #
            Modify(UnicodeReformatter(), text_field="subject"),
            Modify(UnicodeReformatter(), text_field="body"),
            Modify(UnicodeReformatter(), text_field="category"),
            #
            # Filtering
            #
            # Filter out empty emails.
            ScoreFilter(
                FilterEmptyEmails(), text_field="subject", score_type=bool, invert=True
            ),
            ScoreFilter(
                FilterEmptyEmails(), text_field="body", score_type=bool, invert=True
            ),
            ScoreFilter(
                FilterEmptyEmails(), text_field="category", score_type=bool, invert=True
            ),
            # Filter out emails that are too long.
            ScoreFilter(FilterEmailsWithLongBody(), text_field="body", score_type=bool),
            #
            # Redact personally identifiable information (PII).
            #
            redact_pii_subject,
            redact_pii_body,
            #
            # Final modifications.
            #
            # Add system prompts to every email, which helps the model focus on the task.
            Modify(AddSystemPrompt(), text_field="body"),
            # Add a period to the end of each email category, which makes PEFT easier.
            Modify(AddPeriod(), text_field="category"),
        ]
    )

    dataset = curation_steps(dataset)
    dataset = dataset.persist()

    print(f"    Original dataset length: {len(orig_dataset.df)}")
    print(f"    After running the curation pipeline: {len(dataset.df)}")
    print(f"    Writing to '{jsonl_fp}'...")
    out_path = os.path.join(
        os.path.dirname(jsonl_fp),
        "curated",
    )
    os.makedirs(out_path, exist_ok=True)
    dataset.to_json(out_path, write_to_filename=True)
    client.close()
    return os.path.join(out_path, os.path.basename(jsonl_fp))


def main():
    parser = argparse.ArgumentParser()
    args = ArgumentHelper(parser).add_distributed_args().parse_args()
    # Limit the total number of workers to ensure we don't run out of memory.
    args.n_workers = min(args.n_workers, 8)

    # Prepare the download and JSONL directories.
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    jsonl_fp = download_and_convert_to_jsonl()
    run_curation_pipeline(args, jsonl_fp)


if __name__ == "__main__":
    main()
