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
Quick synthetic data generation example for Ray Curator
This example shows how to use the QAMultilingualSyntheticStage to generate synthetic data.
It consists of the following steps:
Step 1: Set up pipeline for synthetic data generation using a multilingual Q&A prompt
Step 2: Run the pipeline executor to generate data batches with the LLM client
Step 3: Filter output using language and score filters
Step 4: Print pipeline description and show generated documents
"""

import argparse
import os

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.modules.score_filter import ScoreFilter
from nemo_curator.models.client.openai_client import AsyncOpenAIClient
from nemo_curator.stages.synthetic.qa_multilingual_synthetic import LanguageFilter, QAMultilingualSyntheticStage


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic multilingual Q&A data using LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # API Configuration
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("NVIDIA_API_KEY", ""),
        help="NVIDIA API key (or set NVIDIA_API_KEY environment variable)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://integrate.api.nvidia.com/v1",
        help="Base URL for the API endpoint"
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=3,
        help="Maximum number of concurrent API requests"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed requests"
    )
    parser.add_argument(
        "--base-delay",
        type=float,
        default=1.0,
        help="Base delay between retries (in seconds)"
    )
    
    # Model Configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="nvdev/nvidia/llama-3.1-nemotron-70b-instruct",
        help="Name of the model to use for generation"
    )
    
    # Generation Configuration
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["English", "French", "German", "Spanish", "Italian"],
        help="Languages to generate Q&A pairs for"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--filter-languages",
        nargs="+",
        default=["[EN]"],
        help="Language codes to filter (e.g., [EN] [FR] [DE] [ES] [IT])"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt template (must include {language} placeholder)"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function to run the synthetic data generation pipeline."""
    args = parse_args()
    
    # Validate API key
    if not args.api_key:
        raise ValueError(
            "API key is required. Set NVIDIA_API_KEY environment variable or use --api-key argument. "
            "Get your API key from https://build.nvidia.com/settings/api-keys"
        )

    # Create pipeline
    pipeline = Pipeline(name="synthetic_data_generation", description="Generate synthetic text data using LLM")

    # Create NeMo Curator Async LLM client for faster concurrent generation
    llm_client = AsyncOpenAIClient(
        api_key=args.api_key,
        base_url=args.base_url,
        max_concurrent_requests=args.max_concurrent_requests,
        max_retries=args.max_retries,
        base_delay=args.base_delay
    )

    # Define a prompt for synthetic data generation
    prompt = args.prompt if args.prompt else """
    Generate a short question and a short answer in the general science domain in the language {language}.
    Begin with the language name using the 2-letter code, which is in square brackets, e.g. [EN] for English, [FR] for French, [DE] for German, [ES] for Spanish, [IT] for Italian.
    """

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
    pipeline.add_stage(
        ScoreFilter(
            LanguageFilter(
                languages=args.filter_languages,
            ),
            text_field="text",
        ),
    )

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    # Create executor
    executor = XennaExecutor()

    # Execute pipeline
    print("Starting synthetic data generation pipeline...")
    results = pipeline.run(executor)

    # Print results
    print("\nPipeline completed!")
    print(f"Total output documents: {len(results) if results else 0}")

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