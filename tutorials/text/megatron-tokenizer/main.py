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
import glob
import os
import time

from datasets import load_dataset

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer.megatron_tokenizer import MegatronTokenizerWriter


def main(args: argparse.Namespace) -> None:
    # Initialize and start the Ray client
    ray_client = RayClient()
    ray_client.start()

    print(f"Running the Megatron-LM Tokenization pipeline for {args.input_path}")
    print(f"    The tokenized dataset will be written to '{args.output_path}'")

    # Download the dataset if it doesn't exist
    if not glob.glob(os.path.join(args.input_path, "*.parquet")):
        print(f"Downloading the TinyStories dataset to {args.input_path}...")
        os.makedirs(args.input_path, exist_ok=True)
        num_rows_per_file = 200_000
        input_df = load_dataset("roneneldan/TinyStories", split="train").to_pandas()
        for i, start_idx in enumerate(range(0, len(input_df), num_rows_per_file)):
            end_idx = min(len(input_df), start_idx + num_rows_per_file)
            subset_df = input_df.iloc[start_idx:end_idx].copy()
            subset_df.to_parquet(os.path.join(args.input_path, f"part_{i}.parquet"), index=False)

        print(f"Downloaded the TinyStories dataset to {args.input_path} with {len(os.listdir(args.input_path))} files")

    # Define the processing stages
    stages = [
        # Read the data from the Parquet files
        ParquetReader(
            file_paths=args.input_path,
        ),
        # Tokenize the data
        MegatronTokenizerWriter(
            path=args.output_path,
            model_identifier=args.tokenizer_model,
            append_eod=args.append_eod,
        ),
    ]

    # Create a pipeline with the stages
    pipeline = Pipeline(
        name="megatron-tokenizer",
        description="Tokenize dataset for Megatron-LM using the MegatronTokenizerWriter stage.",
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
    group.add_argument(
        "--input-path",
        type=str,
        default="tutorials/text/megatron-tokenizer/datasets/tinystories",
        help="Path to folder containing Parquet files",
    )
    group.add_argument(
        "--output-path",
        type=str,
        default="tutorials/text/megatron-tokenizer/datasets/tinystories-tokens",
        help="Path to output directory",
    )
    group.add_argument(
        "--tokenizer-model",
        type=str,
        default="nvidia/NVIDIA-Nemotron-Nano-12B-v2",
        help="Hugging Face model identifier for the tokenizer",
    )
    group.add_argument("--append-eod", action="store_true", help="Append an <eod> token to the end of a document.")

    args = parser.parse_args()
    main(args)
