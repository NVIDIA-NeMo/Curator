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
Quick synthetic data generation example for NeMo Curator
This example shows how to use the QAMultilingualSyntheticStage to generate synthetic data.
It consists of the following steps:
Step 1: Set up pipeline for synthetic data generation using a multilingual Q&A prompt
Step 2: Run the pipeline executor to generate data batches with the LLM client
Step 3: Optionally Filter output using language and score filters
Step 4: Print pipeline description and show generated documents
"""

import argparse
import os
import time

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.models.client.openai_client import AsyncOpenAIClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.qa_multilingual_synthetic import QAMultilingualSyntheticStage
from nemo_curator.stages.text.filters.doc_filter import DocumentFilter
from nemo_curator.stages.text.modules.score_filter import ScoreFilter


class BeginsWithLanguageFilter(DocumentFilter):
    """Filter documents based on language prefix codes.

    Keeps documents that start with any of the specified language codes.
    Designed to work with prompts that instruct LLMs to prefix responses
    with language codes (e.g., [EN], [FR], [DE]).
    """

    def __init__(self, languages: list[str]):
        self._name = "begins_with_language_filter"
        self.languages = languages

    def score_document(self, text: str) -> float:
        if not self.languages:
            return 1.0  # If no languages are specified, keep all documents
        return 1.0 if text.startswith(tuple(self.languages)) else 0.0

    def keep_document(self, score: float) -> bool:
        return score == 1.0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic multilingual Q&A data using LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # API Configuration
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("NVIDIA_API_KEY", ""),
        help="NVIDIA API key (or set NVIDIA_API_KEY environment variable)",
    )
    parser.add_argument(
        "--base-url", type=str, default="https://integrate.api.nvidia.com/v1", help="Base URL for the API endpoint"
    )
    parser.add_argument(
        "--max-concurrent-requests", type=int, default=3, help="Maximum number of concurrent API requests"
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for failed requests")
    parser.add_argument("--base-delay", type=float, default=1.0, help="Base delay between retries (in seconds)")

    # Model Configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="nvdev/nvidia/llama-3.1-nemotron-70b-instruct",
        help="Name of the model to use for generation",
    )

    # Generation Configuration
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["English", "French", "German", "Spanish", "Italian"],
        help="Languages to generate Q&A pairs for",
    )
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--no-filter-languages", action="store_true", help="Do not filter languages")
    parser.add_argument(
        "--prompt", type=str, default=None, help="Custom prompt template (must include {language} placeholder)"
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default=None,
        help="Optional file path to save benchmark results",
    )

    return parser.parse_args()


def main() -> None:
    """Main function to run the synthetic data generation pipeline."""
    args = parse_args()

    # Validate API key
    if not args.api_key:
        msg = (
            "API key is required. Set NVIDIA_API_KEY environment variable or use --api-key argument. "
            "Get your API key from https://build.nvidia.com/settings/api-keys"
        )
        raise ValueError(msg)

    # Create pipeline
    pipeline = Pipeline(name="synthetic_data_generation", description="Generate synthetic text data using LLM")

    # Create NeMo Curator Async LLM client for faster concurrent generation
    llm_client = AsyncOpenAIClient(
        api_key=args.api_key,
        base_url=args.base_url,
        max_concurrent_requests=args.max_concurrent_requests,
        max_retries=args.max_retries,
        base_delay=args.base_delay,
    )

    # Define a prompt for synthetic data generation
    prompt = (
        args.prompt
        if args.prompt
        else """
    Generate a short question and a short answer in the general science domain in the language {language}.
    Begin with the language name using the 2-letter code, which is in square brackets, e.g. [EN] for English, [FR] for French, [DE] for German, [ES] for Spanish, [IT] for Italian.
    """
    )

    # Add the synthetic data generation stage
    pipeline.add_stage(
        QAMultilingualSyntheticStage(
            prompt=prompt,
            languages=args.languages,
            client=llm_client,
            model_name=args.model_name,
            num_samples=args.num_samples,
        )
    )
    if not args.no_filter_languages:
        pipeline.add_stage(
            ScoreFilter(
                BeginsWithLanguageFilter(
                    languages=["[EN]"],  # Only keep English documents
                ),
                text_field="text",
            ),
        )

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    # Create executor
    executor = XennaExecutor()

    # Execute pipeline with timing
    print("Starting synthetic data generation pipeline...")
    start_time = time.time()
    results = pipeline.run(executor)
    end_time = time.time()

    elapsed_time = end_time - start_time

    # Print results
    print("\nPipeline completed!")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    print(f"Total output documents: {len(results) if results else 0}")

    # Calculate throughput if documents were generated
    if results:
        total_docs = sum(len(batch.data) for batch in results)
        if total_docs > 0:
            print(f"Throughput: {total_docs / elapsed_time:.2f} documents/second")
            print(f"Average time per document: {elapsed_time / total_docs:.2f} seconds")

            # Save benchmark results if output file is specified
            if args.benchmark_output:
                with open(args.benchmark_output, "a") as f:
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Model: {args.model_name}\n")
                    f.write(f"Num Samples: {args.num_samples}\n")
                    f.write(f"Languages: {', '.join(args.languages)}\n")
                    f.write(f"Max Concurrent Requests: {args.max_concurrent_requests}\n")
                    f.write(f"Total Execution Time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)\n")
                    f.write(f"Total Documents Generated: {total_docs}\n")
                    f.write(f"Throughput: {total_docs / elapsed_time:.2f} documents/second\n")
                    f.write(f"Average Time per Document: {elapsed_time / total_docs:.2f} seconds\n")
                    f.write("-" * 80 + "\n\n")
                print(f"\nBenchmark results saved to: {args.benchmark_output}")

    if results:
        for i, document_batch in enumerate(results):
            print(f"\nDocument Batch {i}:")
            print(f"Number of documents: {len(document_batch.data)}")
            print("\nGenerated text:")
            for j, text in enumerate(document_batch.data["text"]):
                print(f"Document {j + 1}:")
                print(f"'{text}'")
                print("-" * 40)


if __name__ == "__main__":
    main()
