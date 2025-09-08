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

from loguru import logger
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
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modifiers import UnicodeReformatter
from nemo_curator.stages.text.modules import Modify, ScoreFilter


def main(args: argparse.Namespace) -> None:
    ray_client = RayClient()
    ray_client.start()

    raw_dir = os.path.join(args.data_root, "raw")
    curated_dir = os.path.join(args.data_root, "curated")
    # Initialize the directories
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(curated_dir, exist_ok=True)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info("Running the TinyStories curation pipeline")
    logger.info(f"    The dataset will be downloaded to '{raw_dir}'")
    logger.info(f"    The curated dataset will be written to '{curated_dir}'")

    # Define the processing stages
    stages = [
        EnronEmailsDownloadExtractStage(raw_dir),
        ScoreFilter(
            filter_obj=FilterEmailsWithLongBody(),
            text_field="body",
        ),
        ScoreFilter(
            filter_obj=[FilterEmptyEmails(), FilterEmptyEmails(), FilterEmptyEmails()],
            text_field=["subject", "body", "category"],
            invert=True,
        ),
        Modify(
            modifier_fn=[UnicodeReformatter(), UnicodeReformatter(), UnicodeReformatter()],
            input_fields=["subject", "body", "category"],
            output_fields=["subject", "body", "category"],
        ),
        Modify(
            modifier_fn=AddPeriod(),
            input_fields=["category"],
            output_fields=["category"],
        ),
        # Apply a chat template
        Modify(
            modifier_fn=ApplyChatTemplate(tokenizer),
            input_fields=[["subject", "body", "category"]],
            output_fields=["text"],
        ),
        # Add a column for the number of tokens in each record.
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

    logger.info("Starting the curation pipeline")
    start_time = time.time()
    results = pipeline.run()
    end_time = time.time()
    execution_time = end_time - start_time
    # Count the total number of records.
    logger.info(f"\n\nCuration pipeline finished (took {execution_time} seconds)")
    logger.info(f"The results were written to '{[result.data for result in results]}'")

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
