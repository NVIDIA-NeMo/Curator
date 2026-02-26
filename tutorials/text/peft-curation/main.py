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
import os
import time

from stages import (
    AddPeriod,
    AddTokenCount,
    ApplyChatTemplate,
    EnronEmailsDownloadExtractStage,
    FilterEmailsWithLongBody,
    FilterEmptyEmails,
)
from transformers import AutoTokenizer

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.filters import ScoreFilter
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modifiers import Modify
from nemo_curator.stages.text.modifiers.unicode import UnicodeReformatter


def main(args: argparse.Namespace) -> None:
    # Initialize and start the Ray client
    ray_client = RayClient()
    ray_client.start()

    raw_dir = os.path.join(args.data_root, "raw")
    curated_dir = os.path.join(args.data_root, "curated")
    # Initialize the directories
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(curated_dir, exist_ok=True)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print("Running the TinyStories curation pipeline")
    print(f"    The dataset will be downloaded to '{raw_dir}'")
    print(f"    The curated dataset will be written to '{curated_dir}'")

    # Define the processing stages
    stages = [
        # Download and conversion to a DocumentBatch
        EnronEmailsDownloadExtractStage(raw_dir),
        # If the email is too long, discard
        ScoreFilter(
            filter_obj=FilterEmailsWithLongBody(),
            text_field="body",
        ),
        # Detects empty emails (either empty body, or labeled as empty)
        # We check the subject, body, and category
        # If the email is empty, then discard
        ScoreFilter(
            filter_obj=[FilterEmptyEmails(), FilterEmptyEmails(), FilterEmptyEmails()],
            text_field=["subject", "body", "category"],
            invert=True,
        ),
        # Uses the ftfy library to correct broken Unicode text
        # We check the subject, body, and category
        Modify(
            modifier_fn=[UnicodeReformatter(), UnicodeReformatter(), UnicodeReformatter()],
            input_fields=["subject", "body", "category"],
            output_fields=["subject", "body", "category"],
        ),
        # A simple modifier that adds a period to the end of each email category
        Modify(
            modifier_fn=AddPeriod(),
            input_fields=["category"],
            output_fields=["category"],
        ),
        # Apply a chat template to the subject, body, and category fields to create a "text" field
        Modify(
            modifier_fn=ApplyChatTemplate(tokenizer),
            input_fields=[["subject", "body", "category"]],
            output_fields=["text"],
        ),
        # Add a column for the number of tokens in each record
        Modify(
            modifier_fn=AddTokenCount(tokenizer),
            input_fields=["text"],
            output_fields=["num_tokens"],
        ),
        # Write the results
        JsonlWriter(curated_dir),
    ]

    pipeline = Pipeline(
        name="Enron Emails curation",
        description="Download and curation pipeline for the Enron Emails dataset.",
        stages=stages,
    )

    print("Starting the curation pipeline")
    start_time = time.time()
    # Run the pipeline
    results = pipeline.run()
    end_time = time.time()
    execution_time = end_time - start_time
    # Count the total number of records
    print(f"\n\nCuration pipeline finished (took {execution_time} seconds)")
    print(f"The results were written to '{[result.data for result in results]}'")

    # Stop the Ray client
    ray_client.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyStories dataset curation example.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)) + "/data",
        help="The path to the data directory, which will store the downloaded data, as well as the final results.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
        help="The tokenizer to use for applying the chat template to each record, and count the total number of tokens.",
    )
    args = parser.parse_args()

    main(args)
