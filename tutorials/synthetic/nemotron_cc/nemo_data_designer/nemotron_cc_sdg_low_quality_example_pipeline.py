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

"""
NDD-backed NemotronCC SDG pipeline for low-quality Wikipedia paraphrasing.

This is the NeMo Data Designer equivalent of
``tutorials/synthetic/nemotron_cc/nemotron_cc_sdg_low_quality_example_pipeline.py``.
Instead of using AsyncOpenAIClient + GenerationConfig directly, it configures
the LLM through NDD's ModelConfig / ModelProvider / ChatCompletionInferenceParams.
"""

import argparse
import math
import os
import sys
import time

import data_designer.config as dd
import pandas as pd

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.nemotron_cc.nemo_data_designer.nemotron_cc import WikipediaParaphrasingStage
from nemo_curator.stages.synthetic.nemotron_cc.prompts import (
    NEMOTRON_CC_SYSTEM_PROMPT,
    WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE,
)
from nemo_curator.stages.text.filters import Filter
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.tasks.document import DocumentBatch

# Import shared preprocessing/postprocessing helpers from the parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nemotron_cc_pipelines import (
    add_preprocessing_pipeline,
    add_wikipedia_postprocessing_pipeline,
)

BUCKETED_RESULTS_THRESHOLD = 11

TASK_CONFIGS = {
    "wikipedia_paraphrasing": {
        "system_prompt": NEMOTRON_CC_SYSTEM_PROMPT,
        "prompt_template": WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE,
        "min_document_tokens": 5,
        "min_segment_tokens": 5,
        "max_input_tokens": 512,
        "max_output_tokens": 512,
    },
}

GENERATION_CONFIG = {
    "MAX_INPUT_TOKENS": 512,
    "MAX_OUTPUT_TOKENS": 512,
    "TOP_P": 0.9,
    "TEMPERATURE": 0.5,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # API / Provider Configuration
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("NVIDIA_API_KEY", ""),
        help="API key for the LLM provider (or set NVIDIA_API_KEY environment variable)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://integrate.api.nvidia.com/v1",
        help="Base URL for the LLM API endpoint",
    )
    parser.add_argument(
        "--provider-name",
        type=str,
        default="nvidia",
        help="NDD model provider name",
    )

    # Model Configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta/llama-3.3-70b-instruct",
        help="Model identifier (e.g. 'meta/llama-3.3-70b-instruct')",
    )
    parser.add_argument(
        "--model-alias",
        type=str,
        default=None,
        help="NDD model alias (defaults to --model-name)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer for preprocessing and postprocessing",
    )

    # Generation Configuration
    parser.add_argument(
        "--output-path",
        type=str,
        default="./synthetic_output",
        help="Directory path to save the generated synthetic data",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="parquet",
        choices=["jsonl", "parquet"],
        help="Output format for generated data (jsonl or parquet)",
    )
    parser.add_argument(
        "--input-parquet-path",
        type=str,
        default=None,
        help="If set, read inputs from Parquet path/glob via Curator ParquetReader",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use preset in-script input_data instead of reading Parquet input",
    )

    # NDD Inference Parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Range: 0.0-2.0",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling parameter. Range: 0.0-1.0",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate per sample",
    )
    parser.add_argument(
        "--max-parallel-requests",
        type=int,
        default=4,
        help="Maximum number of parallel LLM requests (NDD concurrency)",
    )

    return parser.parse_args()


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Main function to run the synthetic data generation pipeline."""
    args = parse_args()

    # Set tokenizer
    if args.tokenizer is None:
        msg = "Tokenizer is required"
        raise ValueError(msg)
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        msg = (
            "The 'transformers' package is required for tokenizer support. "
            "Install it with: pip install transformers"
        )
        raise ImportError(msg) from e
    args.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    args.hf_token = os.environ.get("HF_TOKEN", "")

    # Validate API key
    if not args.api_key:
        msg = (
            "API key is required. Set NVIDIA_API_KEY environment variable or use --api-key argument. "
            "Get your API key from https://build.nvidia.com/settings/api-keys"
        )
        raise ValueError(msg)

    client = RayClient(include_dashboard=False)
    client.start()

    # Resolve model alias
    model_alias = args.model_alias or args.model_name

    # Build NDD model provider and config
    model_provider = dd.ModelProvider(
        name=args.provider_name,
        endpoint=args.base_url,
        provider_type="openai",
        api_key=args.api_key,
    )

    inference_params = dd.ChatCompletionInferenceParams(
        temperature=args.temperature if args.temperature is not None else GENERATION_CONFIG["TEMPERATURE"],
        top_p=args.top_p if args.top_p is not None else GENERATION_CONFIG["TOP_P"],
        max_tokens=args.max_tokens if args.max_tokens is not None else GENERATION_CONFIG["MAX_OUTPUT_TOKENS"],
        max_parallel_requests=args.max_parallel_requests,
    )

    model_config = dd.ModelConfig(
        alias=model_alias,
        model=args.model_name,
        provider=args.provider_name,
        inference_parameters=inference_params,
    )

    # Set task config
    task_config = TASK_CONFIGS["wikipedia_paraphrasing"]

    # Create pipeline
    pipeline = Pipeline(
        name="nemotron_cc_wikipedia_low_quality_ndd",
        description="Generate Wikipedia-style paraphrases for low quality data using NemotronCC (NDD backend)",
    )

    input_tasks = None
    if args.mock:
        input_data = [
            {
                "id": 0,
                "text": "The Amazon rainforest contains an unparalleled diversity of plant and animal species.",
                "bucketed_results": 12,
            },
            {
                "id": 1,
                "text": "Isaac Newton formulated the laws of motion and universal gravitation.",
                "bucketed_results": 4,
            },
            {
                "id": 2,
                "text": "The Great Wall of China is a historic fortification built to protect ancient Chinese states.",
                "bucketed_results": 17,
            },
            {
                "id": 3,
                "text": "Mercury is the smallest planet in the Solar System and orbits closest to the Sun.",
                "bucketed_results": 1,
            },
            {
                "id": 4,
                "text": "The Parthenon is a classical Greek temple dedicated to the goddess Athena.",
                "bucketed_results": 9,
            },
            {
                "id": 5,
                "text": "Giraffes are the tallest living terrestrial animals, native to African savannas.",
                "bucketed_results": 6,
            },
            {
                "id": 6,
                "text": "Marie Curie made pioneering contributions to the study of radioactivity.",
                "bucketed_results": 14,
            },
            {
                "id": 7,
                "text": "The Pacific Ocean covers more area than all landmasses combined.",
                "bucketed_results": 3,
            },
            {
                "id": 8,
                "text": "The Rosetta Stone provided the key to deciphering ancient Egyptian hieroglyphs.",
                "bucketed_results": 18,
            },
            {
                "id": 9,
                "text": "The cheetah is capable of reaching speeds over 100 kilometers per hour.",
                "bucketed_results": 8,
            },
        ]
        batch_size = 5
        input_batches = [input_data[i : i + batch_size] for i in range(0, len(input_data), batch_size)]
        input_tasks = []
        for i, batch in enumerate(input_batches):
            df = pd.DataFrame(batch)
            input_task = DocumentBatch(
                data=df,
                task_id=f"input_batch_{i}",
                dataset_name="data_for_sdg",
            )
            input_tasks.append(input_task)
    else:
        if not args.input_parquet_path:
            msg = "When not using --mock, you must provide --input-parquet-path to read inputs."
            raise ValueError(msg)
        pipeline.add_stage(
            ParquetReader(
                file_paths=[args.input_parquet_path],
                read_kwargs={"engine": "pyarrow", "dtype_backend": "pyarrow"},
            )
        )

    ### Filter low quality data
    pipeline.add_stage(
        Filter(
            filter_fn=lambda x: x is not None and not (isinstance(x, float) and math.isnan(x)) and int(x) <= BUCKETED_RESULTS_THRESHOLD,
            filter_field="bucketed_results",
        ),
    )

    ### Preprocessing Stages
    print(
        f"Adding preprocessing pipeline (min_document_tokens={task_config['min_document_tokens']}, min_segment_tokens={task_config['min_segment_tokens']})..."
    )
    pipeline = add_preprocessing_pipeline(
        pipeline=pipeline,
        text_field="text",
        system_prompt=task_config["system_prompt"],
        user_prompt_template=task_config["prompt_template"],
        min_document_tokens=task_config["min_document_tokens"],
        min_segment_tokens=task_config["min_segment_tokens"],
        max_input_tokens=task_config["max_input_tokens"],
        args=args,
    )

    ### Wikipedia Paraphrasing Stage (NDD-backed)
    pipeline.add_stage(
        WikipediaParaphrasingStage(
            model_alias=model_alias,
            model_configs=[model_config],
            model_providers=[model_provider],
            input_field="text",
            output_field="rephrased",
        )
    )

    ### Postprocessing Stages
    print("Adding postprocessing pipeline...")
    pipeline = add_wikipedia_postprocessing_pipeline(
        pipeline=pipeline,
        llm_response_field="rephrased",
        args=args,
    )

    ### Write output
    if args.output_format == "jsonl":
        pipeline.add_stage(JsonlWriter(path=args.output_path))
    else:
        pipeline.add_stage(ParquetWriter(path=args.output_path))

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    # Create executor
    executor = XennaExecutor()

    # Execute pipeline with timing
    print("Starting synthetic data generation pipeline...")
    start_time = time.time()
    results = pipeline.run(executor, input_tasks)
    end_time = time.time()

    elapsed_time = end_time - start_time

    # Print results
    print("\nPipeline completed!")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

    output_files = []
    all_data_frames = []
    if results:
        print(f"\nGenerated data saved to: {args.output_path}")
        for result in results:
            if hasattr(result, "data") and result.data:
                for file_path in result.data:
                    print(f"  - {file_path}")
                    output_files.append(file_path)
                    if file_path.endswith(".jsonl"):
                        df = pd.read_json(file_path, lines=True)
                    elif file_path.endswith(".parquet"):
                        df = pd.read_parquet(file_path)
                    else:
                        continue
                    all_data_frames.append(df)

    # Display sample of generated documents
    print("\n" + "=" * 50)
    print("Sample of generated documents:")
    print("=" * 50)
    for i, df in enumerate(all_data_frames):
        out_path = output_files[i]
        print(f"\nFile {i + 1}: {out_path}")
        print(f"Number of documents: {len(df)}")
        print("\nFirst 5 rows:")
        for j, row in enumerate(df.head(5).to_dict(orient="records")):
            print(f"Document {j + 1}: {row}")
            print("-" * 40)

    client.stop()


if __name__ == "__main__":
    main()
