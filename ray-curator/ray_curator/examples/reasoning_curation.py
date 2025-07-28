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

import pandas as pd

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.modules.score_filter import ScoreFilter
from ray_curator.stages.reasoning.correctness_filter import LLMBasedCorrectnessFilter, LLMBasedGrader
from ray_curator.stages.reasoning.difficulty_filter import (
    LLMBasedDifficultyFilter,
    LLMBasedDifficultyFilterFunction,
    ReasoningLengthDifficultyFilter,
)
from ray_curator.stages.reasoning.diversity_filter import DiversitySampler, LLMBasedDomainClassifier
from ray_curator.stages.reasoning.reasoning_traces_synthetic import ReasoningTracesSyntheticStage
from ray_curator.stages.services.openai_client import AsyncOpenAIClient, OpenAIClient
from ray_curator.tasks import DocumentBatch


def setup_client(args: argparse.Namespace) -> AsyncOpenAIClient | OpenAIClient:
    """Set up the LLM client based on async flag."""
    if args.enable_async:
        print("🚀 Using ASYNC generation with concurrent processing")
        return AsyncOpenAIClient(
            max_concurrent_requests=args.max_concurrent_requests,
            max_retries=args.max_retries,
            base_delay=args.retry_delay,
            api_key=os.environ.get("NVIDIA_API_KEY", "<your-nvidia-api-key>"),
            base_url=args.base_url,
            timeout=args.timeout,
        )
    else:
        print("🐌 Using SYNC generation with sequential processing")
        return OpenAIClient(
            api_key=os.environ.get("NVIDIA_API_KEY", "<your-nvidia-api-key>"),
            base_url=args.base_url,
            timeout=args.timeout,
        )


def process_input_data(args: argparse.Namespace) -> tuple[pd.DataFrame, list[DocumentBatch]]:
    """Read and batch input data."""
    input_data = pd.read_csv(args.input_path)
    print(f"📊 Processing {len(input_data)} rows")

    # Divide input data into batches
    batch_size = args.batch_size
    data_batches = []
    for i in range(0, len(input_data), batch_size):
        batch_data = input_data.iloc[i:i + batch_size]
        data_batches.append(batch_data)

    print(f"📦 Created {len(data_batches)} data batches with batch size {batch_size}")
    print(f"📊 Batch sizes: {[len(batch) for batch in data_batches]}")

    # Wrap each batch with DocumentBatch
    input_batches = []
    for i, batch_data in enumerate(data_batches):
        input_batch = DocumentBatch(
            data=batch_data,
            task_id=f"input_questions_batch_{i}",
            dataset_name="reasoning_traces_synthetic",
        )
        input_batches.append(input_batch)

    return input_data, input_batches


def setup_pipeline_stages(pipeline: Pipeline, llm_client: AsyncOpenAIClient | OpenAIClient, args: argparse.Namespace) -> None:
    """Add all stages to the pipeline."""
    # 1. Generate reasoning traces (automatically detects async/sync based on client type)
    pipeline.add_stage(
        ReasoningTracesSyntheticStage(
            prompt=None,
            client=llm_client,
            model_name=args.llm_reasoning_model,
            input_problem_field="question",
            output_field="reasoning_trace_attempt",
        ),
    )

    # 2. Correctness filter
    pipeline.add_stage(
        LLMBasedGrader(
            client=llm_client,
            model_name=args.llm_grader_model,
            prompt=None,
            input_problem_field="question",
            input_solution_field="solution",
            input_attempt_field="reasoning_trace_attempt",
            output_field="reasoning_trace_correctness",
        ),
    )
    pipeline.add_stage(
        ScoreFilter(
            LLMBasedCorrectnessFilter(),
            text_field="reasoning_trace_correctness",
        ),
    )

    # 3. Difficulty filter
    # 3.1. Length difficulty filter
    pipeline.add_stage(
        ScoreFilter(
            ReasoningLengthDifficultyFilter(
                min_length=100,
            ),
            text_field="reasoning_trace_attempt",
        ),
    )

    # 3.2. LLM-based difficulty filter
    pipeline.add_stage(
        ReasoningTracesSyntheticStage(
            prompt=None,
            client=llm_client,
            model_name=args.llm_difficulty_model_1,
            input_problem_field="question",
            output_field="llm_difficulty_1_attempt",
        ),
    )
    pipeline.add_stage(
        ReasoningTracesSyntheticStage(
            prompt=None,
            client=llm_client,
            model_name=args.llm_difficulty_model_2,
            input_problem_field="question",
            output_field="llm_difficulty_2_attempt",
        ),
    )
    pipeline.add_stage(
        LLMBasedGrader(
            client=llm_client,
            model_name=args.llm_grader_model,
            prompt=None,
            input_problem_field="question",
            input_attempt_field="llm_difficulty_1_attempt",
            input_solution_field="solution",
            output_field="llm_difficulty_1_correctness",
        ),
    )
    pipeline.add_stage(
        LLMBasedGrader(
            client=llm_client,
            model_name=args.llm_grader_model,
            prompt=None,
            input_problem_field="question",
            input_attempt_field="llm_difficulty_2_attempt",
            input_solution_field="solution",
            output_field="llm_difficulty_2_correctness",
        ),
    )
    pipeline.add_stage(
        LLMBasedDifficultyFilter(
            filter_obj=LLMBasedDifficultyFilterFunction(
                llm_correctness_fields=["llm_difficulty_1_correctness", "llm_difficulty_2_correctness"],
            ),
            llm_correctness_fields=["llm_difficulty_1_correctness", "llm_difficulty_2_correctness"],
        ),
    )

    # 4. Diversity filter
    # 4.1. Domain classifier
    pipeline.add_stage(
        LLMBasedDomainClassifier(
            client=llm_client,
            model_name=args.llm_domain_classifier_model,
            prompt=None,
            domains_file_path=args.domains_file_path,
            input_problem_field="question",
            output_field="domain",
        ),
    )

    # 4.2. Diversity sampler
    pipeline.add_stage(
        DiversitySampler(
            sampling_size=1000,
            input_problem_field="question",
            input_domain_field="domain",
        ),
    )


def process_results(results: list[DocumentBatch], input_data: pd.DataFrame, execution_time: float, args: argparse.Namespace) -> None:
    """Process and output the pipeline results."""
    # Print results with performance metrics
    print("\nPipeline completed!")
    print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
    print(f"📊 Processed {len(input_data)} rows")
    print(f"⚡ Average time per row: {execution_time / len(input_data):.2f} seconds")
    print(f"🔧 Mode: {'ASYNC' if args.enable_async else 'SYNC'}")
    if args.enable_async:
        print(f"🚀 Max concurrent requests: {args.max_concurrent_requests}")
    print(f"📈 Total output documents: {len(results) if results else 0}")

    if results:
        for i, document_batch in enumerate(results):
            print(f"\nDocument Batch {i}:")
            print(f"Number of documents: {len(document_batch.data)}")
            print("\nGenerated text:")

            # Print the DataFrame data directly
            pd.set_option("display.max_rows", None)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)
            pd.set_option("display.max_colwidth", None)

            print("-" * 40)

    # Write to output file
    print(f"Writing to output file: {args.output_path}")
    combined_data = pd.concat([document_batch.data for document_batch in results])
    combined_data.to_json(args.output_path, orient="records", lines=True)


def main(args: argparse.Namespace) -> None:
    """Main function to run the reasoning curation pipeline."""
    # Create pipeline
    pipeline = Pipeline(name="reasoning_curation", description="Curation pipeline for reasoning traces")

    # Set up client
    llm_client = setup_client(args)

    # Process input data
    input_data, input_batches = process_input_data(args)

    # Add stages to the pipeline
    setup_pipeline_stages(pipeline, llm_client, args)

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    # Create executor and run pipeline
    executor = XennaExecutor()
    print("Starting synthetic data generation pipeline...")
    start_time = time.time()
    results = pipeline.run(executor, input_batches)
    end_time = time.time()
    execution_time = end_time - start_time

    # Process and output results
    process_results(results, input_data, execution_time, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reasoning curation pipeline with async/sync comparison")
    parser.add_argument("--input_path", type=str, default=None, help="Path to input CSV file")
    parser.add_argument("--output_path", type=str, default=None, help="Path to output JSON file")
    parser.add_argument("--domains_file_path", type=str, default=None, help="Path to domains file")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of rows per batch (default: 100)")
    parser.add_argument("--base_url", type=str, default="https://integrate.api.nvidia.com/v1", help="Base URL for the API")
    parser.add_argument(
        "--enable-async",
        action="store_true",
        default=False,
        help="Enable async generation with concurrent processing (default: False for sync processing)"
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=5,
        help="Maximum number of concurrent requests when using async mode (default: 5)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for API requests in seconds (default: 300)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retry attempts for failed requests (default: 5)"
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Base delay between retries in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--llm-reasoning-model",
        type=str,
        default="nvdev/nvidia/llama-3.1-nemotron-70b-instruct",
        help="Model name for reasoning trace generation (default: nvdev/nvidia/llama-3.1-nemotron-70b-instruct)"
    )
    parser.add_argument(
        "--llm-domain-classifier-model",
        type=str,
        default="nvdev/nvidia/llama-3.1-nemotron-70b-instruct",
        help="Model name for domain classification (default: nvdev/nvidia/llama-3.1-nemotron-70b-instruct)"
    )
    parser.add_argument(
        "--llm-grader-model",
        type=str,
        default="nvdev/nvidia/llama-3.1-nemotron-70b-instruct",
        help="Model name for grading (default: nvdev/nvidia/llama-3.1-nemotron-70b-instruct)"
    )
    parser.add_argument(
        "--llm-difficulty-model-1",
        type=str,
        default="nvdev/meta/llama-3.2-1b-instruct",
        help="First model for difficulty assessment (default: nvdev/meta/llama-3.2-1b-instruct)"
    )
    parser.add_argument(
        "--llm-difficulty-model-2",
        type=str,
        default="nvdev/meta/llama-3.2-3b-instruct",
        help="Second model for difficulty assessment (default: nvdev/meta/llama-3.2-3b-instruct)"
    )
    
    args = parser.parse_args()

    main(args)
