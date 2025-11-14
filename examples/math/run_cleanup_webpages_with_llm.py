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
from datetime import datetime

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.math import LLMCleanupStage, TokenSplitterStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules.modifier import Modify
from nemo_curator.utils import prompts


def fill_null_text(text):
    """Fill null/NaN text values with empty string."""
    import pandas as pd

    if pd.isna(text) or text is None:
        return ""
    return str(text)


def build_pipeline(
    input_files: list[str],
    reader_type: str,
    output_dir: str,
    model: str,
    prompt: str,
    chunk_length: int | None = None,
    chunk_data: bool = False,
    classification: bool = False,
    max_model_len: int | None = None,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    max_tokens: int | None = None,
    cache_dir: str | None = None,
    filter_by_n_tokens: bool = False,
) -> Pipeline:
    """Build the LLM cleanup pipeline."""
    p = Pipeline(
        name="math_cleanup_webpages_with_llm",
        description="Clean up HTML/text content using LLM (with optional chunking)",
    )

    # Reader stage
    if reader_type == "parquet":
        p.add_stage(
            ParquetReader(file_paths=input_files).with_(
                {
                    "file_partitioning": {"resources": Resources(cpus=0.5)},
                    "parquet_reader": {"resources": Resources(cpus=0.5)},
                }
            )
        )
    else:
        p.add_stage(
            JsonlReader(file_paths=input_files).with_(
                {
                    "file_partitioning": {"resources": Resources(cpus=0.5)},
                    "jsonl_reader": {"resources": Resources(cpus=0.5)},
                }
            )
        )

    # Fill null text values
    p.add_stage(
        Modify(modifier_fn=fill_null_text, input_fields="text", output_fields="text").with_(
            resources=Resources(cpus=0.5)
        )
    )

    # Optional chunking stage
    if chunk_data and chunk_length:
        p.add_stage(
            TokenSplitterStage(
                model_name=model,
                text_field="text",
                max_length_tokens=chunk_length,
            ).with_(resources=Resources(cpus=1))
        )

    # Get prompt from prompts module
    try:
        system_prompt = getattr(prompts, prompt)
    except AttributeError:
        # If not found in prompts module, use the string directly
        system_prompt = prompt

    # LLM cleanup stage
    p.add_stage(
        LLMCleanupStage(
            model=model,
            system_prompt=system_prompt,
            text_field="text",
            output_field="cleaned_text",
            max_model_len=max_model_len,
            classification=classification,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            cache_dir=cache_dir,
            filter_by_n_tokens=filter_by_n_tokens,
        ).with_(resources=Resources(cpus=1, gpus=1))
    )

    # Writer stage
    p.add_stage(JsonlWriter(path=output_dir).with_(resources=Resources(cpus=0.5)))

    return p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean up webpages using LLM with optional chunking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input directory or glob pattern for JSONL/Parquet files")
    parser.add_argument("--output", required=True, help="Output directory for cleaned JSONL files")
    parser.add_argument("--model", required=True, help="Model identifier (e.g., microsoft/phi-4)")
    parser.add_argument("--prompt", required=True, help="Prompt name from prompts module (e.g., HTML_TO_TEXT_PROMPT)")
    parser.add_argument(
        "--input_filetype",
        choices=["jsonl", "parquet"],
        default="jsonl",
        help="Input file type",
    )
    parser.add_argument(
        "--chunk_data",
        action="store_true",
        help="Enable token-based chunking before LLM processing",
    )
    parser.add_argument(
        "--chunk_length",
        type=int,
        default=None,
        help="Maximum tokens per chunk when chunk_data is enabled",
    )
    parser.add_argument(
        "--classification",
        action="store_true",
        help="Use classification mode (outputs 'label' instead of 'cleaned_text')",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum model context length. If not specified, vLLM will auto-detect from model config.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling parameter")
    parser.add_argument("--min_p", type=float, default=0.0, help="Min-p sampling parameter (for Qwen3)")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum tokens to generate")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for model weights")
    parser.add_argument(
        "--filter_by_n_tokens",
        action="store_true",
        help="Filter chunks by n_tokens field (requires chunk_data)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.chunk_data and not args.chunk_length:
        parser.error("--chunk_length is required when --chunk_data is enabled")
    if args.filter_by_n_tokens and not args.chunk_data:
        parser.error("--filter_by_n_tokens requires --chunk_data")

    # Expand input glob pattern
    if os.path.isdir(args.input):
        if args.input_filetype == "parquet":
            input_files = glob.glob(os.path.join(args.input, "**/*.parquet"), recursive=True)
        else:
            input_files = glob.glob(os.path.join(args.input, "**/*.jsonl"), recursive=True)
            input_files.extend(glob.glob(os.path.join(args.input, "**/*.json"), recursive=True))
    else:
        input_files = glob.glob(args.input)

    if not input_files:
        logger.error(f"No input files found matching: {args.input}")
        return

    logger.info(f"Found {len(input_files)} input files")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"cleanup_{timestamp}")

    ray_client = RayClient()
    ray_client.start()

    try:
        pipeline = build_pipeline(
            input_files=input_files,
            reader_type=args.input_filetype,
            output_dir=output_dir,
            model=args.model,
            prompt=args.prompt,
            chunk_length=args.chunk_length,
            chunk_data=args.chunk_data,
            classification=args.classification,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            max_tokens=args.max_tokens,
            cache_dir=args.cache_dir,
            filter_by_n_tokens=args.filter_by_n_tokens,
        )

        logger.info(pipeline.describe())

        executor = XennaExecutor()
        pipeline.run(executor)

        logger.info("Pipeline completed successfully.")
        logger.info(f"Output written to: {output_dir}")

    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
