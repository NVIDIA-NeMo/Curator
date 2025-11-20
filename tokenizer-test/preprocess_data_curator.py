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

# ruff: noqa: INP001

import argparse
import time

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import MegatronTokenizerWriter


# uv run python tokenize_dataset.py --output-dir /localhome/local-asolergibert/datasets/output/tokens
def main(args: argparse.Namespace) -> None:
    # Initialize and start the Ray client
    ray_client = RayClient()
    ray_client.start()

    print(f"Running the Megatron-LM Tokenization pipeline for {args.input}")
    print(f"    The tokenized dataset will be written to '{args.output}'")

    # Define the processing stages
    stages = [
        # Read the data from the JSONL files
        JsonlReader(
            file_paths=args.input,
            fields=[args.json_key],
        ),
        # Tokenize the data
        MegatronTokenizerWriter(
            path=args.output,
            model_identifier=args.tokenizer_model,
            append_eod=args.append_eod,
            text_field=args.json_key,
        ),
    ]

    # Create a pipeline with the stages
    pipeline = Pipeline(
        name="parquet-tokenize",
        description="Tokenize the Nemotron-CC-v2 dataset.",
        stages=stages,
    )

    print("Starting the tokenization pipeline")
    start_time = time.time()
    # Run the pipeline
    results = pipeline.run()
    end_time = time.time()
    execution_time = end_time - start_time
    # Count the total number of records
    print(f"\n\nTokenization pipeline finished (took {execution_time} seconds)")
    print(f"The results were written to '{[result.data for result in results]}'")

    # Stop the Ray client
    ray_client.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument("--input", type=str, required=True, help="Path to input JSON")
    group.add_argument("--json-key", type=str, default="text", help="Key to extract from json")
    group.add_argument("--output", type=str, required=True, help="Path to output directory")
    group.add_argument(
        "--tokenizer-model", type=str, required=True, help="Hugging Face model identifier for the tokenizer"
    )
    group.add_argument("--append-eod", action="store_true", help="Append an <eod> token to the end of a document.")

    args = parser.parse_args()
    main(args)
