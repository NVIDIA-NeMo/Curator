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
    IncompleteStoryFilter,
    QuotationUnifier,
    TinyStoriesDownloadExtractStage,
)

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.filters import ScoreFilter
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modifiers import Modify


def main(args: argparse.Namespace) -> None:
    # Initialize and start the Ray client
    ray_client = RayClient()
    ray_client.start()

    raw_dir = os.path.join(args.data_root, "raw")
    curated_dir = os.path.join(args.data_root, "curated")
    # Initialize the directories
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(curated_dir, exist_ok=True)

    print("Running the TinyStories curation pipeline")
    print(f"    The dataset will be downloaded to '{raw_dir}'")
    print(f"    The curated dataset will be written to '{curated_dir}'")

    # Define the processing stages
    stages = [
        # Download and conversion to a DocumentBatch
        TinyStoriesDownloadExtractStage(raw_dir, split=args.split),
        # If the document doesn't end with a terminating punctuation mark, then discard
        ScoreFilter(
            filter_obj=IncompleteStoryFilter(),
        ),
        # A simple modifier that unifies the quotation marks in the documents
        Modify(
            modifier_fn=QuotationUnifier(),
        ),
        # Write the results
        JsonlWriter(curated_dir),
    ]

    # Create a pipeline with the stages
    pipeline = Pipeline(
        name="tinystories",
        description="Download and curation pipeline for the TinyStories dataset.",
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
        "--split",
        type=str,
        choices=["train", "valid"],
        default="valid",
        help="The dataset split to process (either 'train' or 'valid')",
    )
    args = parser.parse_args()
    main(args)
