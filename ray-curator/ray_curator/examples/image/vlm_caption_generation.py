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
import json
import os
import time
from typing import Any

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.examples.image.helper import save_imagebatch_to_webdataset
from ray_curator.examples.image.vlm_helper import MultiSourceImageReaderStage
from ray_curator.pipeline import Pipeline
from ray_curator.stages.image.captioning import VLMCaptioningStage
from ray_curator.stages.image.io.image_reader import ImageReaderStage
from ray_curator.stages.services.openai_client import (
    AsyncOpenAIClient,
    OpenAIClient,
)

DEFAULT_NGC_BASE_URL = "https://integrate.api.nvidia.com/v1"


def setup_vlm_client(args: argparse.Namespace) -> AsyncOpenAIClient | OpenAIClient:
    """Set up the VLM client based on async flag."""
    # Get API key from argument or environment variable
    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")

    # Handle NGC endpoint
    base_url = args.nim_endpoint
    if args.use_ngc:
        base_url = DEFAULT_NGC_BASE_URL
        if not api_key:
            print(
                "ERROR: --use-ngc is set but NVIDIA_API_KEY is not provided. "
                "Set --api-key or NVIDIA_API_KEY environment variable."
            )
            raise SystemExit(2)

    if args.enable_async:
        print("🚀 Using ASYNC generation with concurrent processing")
        return AsyncOpenAIClient(
            max_concurrent_requests=args.max_concurrent_requests,
            max_retries=args.max_retries,
            base_delay=args.retry_delay,
            api_key=api_key,
            base_url=base_url,
            timeout=args.timeout,
        )
    else:
        print("🐌 Using SYNC generation with sequential processing")
        return OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            timeout=args.timeout,
        )


def create_caption_pipeline(args: argparse.Namespace) -> Pipeline:
    """Create image captioning pipeline."""
    pipeline = Pipeline(
        name="vlm_caption_generation",
        description=("Generate captions for images using VLM NIM"),
    )

    # Stage 1: Read images from specified source
    if args.source_type == "webdataset":
        pipeline.add_stage(
            ImageReaderStage(
                input_dataset_path=args.input_path,
                image_limit=args.image_limit,
                batch_size=args.reader_batch_size,
                verbose=args.verbose,
            )
        )
    else:
        pipeline.add_stage(
            MultiSourceImageReaderStage(
                input_path=args.input_path,
                source_type=args.source_type,
                image_limit=args.image_limit,
                batch_size=args.reader_batch_size,
                verbose=args.verbose,
            )
        )

    # Stage 2: Generate captions using VLM
    vlm_client = setup_vlm_client(args)
    pipeline.add_stage(
        VLMCaptioningStage(
            client=vlm_client,
            model_name=args.model_name,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            verbose=args.verbose,
        )
    )

    return pipeline


def save_results_json(results: list[Any], output_path: str, verbose: bool = False) -> None:
    """Save captioning results to JSON file."""
    output_data = []

    for batch in results:
        for image_obj in batch.data:
            item = {
                "image_id": image_obj.image_id,
                "image_path": image_obj.image_path,
                "caption": image_obj.metadata.get("caption", ""),
                "metadata": image_obj.metadata,
            }
            output_data.append(item)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print(f"💾 Saved {len(output_data)} captioned images to {output_path}")


def main(args: argparse.Namespace) -> None:
    """Main execution function for VLM caption generation pipeline."""
    print("Starting VLM caption generation pipeline...")
    print(f"📁 Input: {args.input_path} (type: {args.source_type})")
    print(f"🤖 Model: {args.model_name}")

    # Show endpoint configuration
    if args.use_ngc:
        print(f"🌐 Endpoint: NGC ({DEFAULT_NGC_BASE_URL})")
        has_api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
        api_key_status = "✅ Set" if has_api_key else "❌ Missing"
        print(f"🔑 API Key: {api_key_status}")
    else:
        print(f"🌐 Endpoint: NIM ({args.nim_endpoint})")
        print("🔑 API Key: Not required for local NIM")

    print(f"💬 Prompt: '{args.prompt}'")
    limit_text = args.image_limit if args.image_limit > 0 else "No limit"
    print(f"🔢 Image limit: {limit_text}")
    print(f"📦 Reader batch size: {args.reader_batch_size}")
    print(f"⚡ Mode: {'ASYNC' if args.enable_async else 'SYNC'}")
    if args.enable_async:
        print(f"🚀 Max concurrent requests: {args.max_concurrent_requests}")
    print("\n" + "=" * 50 + "\n")

    # Create and run pipeline
    start_time = time.time()
    pipeline = create_caption_pipeline(args)

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    # Create executor and run pipeline
    executor = XennaExecutor()
    print("Starting caption generation...")

    results = pipeline.run(executor)
    end_time = time.time()
    execution_time = end_time - start_time

    # Process and save results
    if results:
        total_images = sum(len(batch.data) for batch in results)

        print("\nPipeline completed!")
        print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
        print(f"📊 Processed {total_images} images")
        if total_images > 0:
            avg = execution_time / total_images
            print(f"⚡ Average time per image: {avg:.2f} seconds")
        print(f"🔧 Mode: {'ASYNC' if args.enable_async else 'SYNC'}")

        # Save results
        if args.output_format == "json":
            save_results_json(results, args.output_path, args.verbose)
        elif args.output_format == "webdataset":
            save_imagebatch_to_webdataset(
                image_batches=results,
                output_path=args.output_path,
                samples_per_shard=args.samples_per_shard,
                max_shards=args.max_shards,
            )
            if args.verbose:
                print(f"💾 Saved webdataset to {args.output_path}")

    else:
        print("❌ No results generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("VLM-based image caption generation pipeline"))

    # Input/Output arguments
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help=("Path to input (file, directory, or URL list file)"),
    )
    parser.add_argument(
        "--source-type",
        type=str,
        choices=["local", "urls", "webdataset"],
        default="local",
        help=("Type of input source: local files, URL list, or webdataset"),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help=("Path for output (JSON file or webdataset directory)"),
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "webdataset"],
        default="json",
        help="Output format: JSON file or webdataset",
    )

    # Processing arguments
    parser.add_argument(
        "--image-limit",
        type=int,
        default=-1,
        help=("Limit number of images to process (-1 for no limit)"),
    )
    parser.add_argument(
        "--reader-batch-size",
        type=int,
        default=32,
        help=("Number of images per batch for reading"),
    )

    # VLM/NIM/NGC arguments
    parser.add_argument(
        "--nim-endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help=("NIM endpoint URL (ignored if --use-ngc is set)"),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        help="VLM model name",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Prompt for caption generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum tokens for caption generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for caption generation",
    )

    # Async processing arguments
    parser.add_argument(
        "--enable-async",
        action="store_true",
        default=False,
        help="Enable async processing for faster caption generation",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=5,
        help=("Maximum concurrent requests for async processing"),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for API requests in seconds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Base delay between retries in seconds",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=("API key for NGC endpoint (can also use NVIDIA_API_KEY env var)"),
    )
    parser.add_argument(
        "--use-ngc",
        action="store_true",
        default=False,
        help="Use NVIDIA NGC endpoint instead of local NIM",
    )

    # Output webdataset arguments
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=1000,
        help=("Number of samples per shard in output webdataset"),
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=5,
        help=("Maximum number of shards for output webdataset"),
    )

    # General arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    main(args)
