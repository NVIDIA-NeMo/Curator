# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Modify benchmarking script.

This script runs document modifying benchmarks
with comprehensive metrics collection using various executors and logs results to configured sinks.
"""

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger
from utils import load_dataset_files, setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.stages.text.modifiers import Modify
from nemo_curator.stages.text.modifiers.string import (
    BoilerPlateStringModifier,
    MarkdownRemover,
    NewlineNormalizer,
    QuotationRemover,
    UrlRemover,
)
from nemo_curator.stages.text.modifiers.unicode import UnicodeReformatter


def run_modify_benchmark(  # noqa: PLR0913
    input_path: Path,
    output_path: Path,
    executor_name: str,
    benchmark_results_path: Path,
    dataset_size_gb: float | None = None,
    input_filetype: str = "jsonl",
) -> dict[str, Any]:
    """Run the Modify benchmark and collect comprehensive metrics."""

    executor = setup_executor(executor_name)

    input_path = str(input_path.absolute())

    # Ensure output directory
    output_path = output_path.absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = str(output_path)

    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.debug(f"Executor: {executor}")

    pipeline = Pipeline(name="modifier_pipeline")

    if dataset_size_gb is not None:
        file_paths = load_dataset_files(input_path, dataset_size_gb, keep_extensions=input_filetype)
        logger.info(f"Dataset size limit: {dataset_size_gb} GB ({len(file_paths)} files selected)")
        reader = (
            JsonlReader(file_paths=file_paths) if input_filetype == "jsonl" else ParquetReader(file_paths=file_paths)
        )
    else:
        reader = JsonlReader(input_path) if input_filetype == "jsonl" else ParquetReader(input_path)
    pipeline.add_stage(reader)

    # Add modify stages
    pipeline.add_stage(Modify(BoilerPlateStringModifier()))
    pipeline.add_stage(Modify(MarkdownRemover()))
    pipeline.add_stage(Modify(NewlineNormalizer()))
    pipeline.add_stage(Modify(QuotationRemover()))
    pipeline.add_stage(Modify(UnicodeReformatter()))
    pipeline.add_stage(Modify(UrlRemover()))

    pipeline.add_stage(ParquetWriter(output_path))

    run_start_time = time.perf_counter()

    try:
        logger.info("Running Modify pipeline...")

        output_tasks = pipeline.run(executor)
        run_time_taken = time.perf_counter() - run_start_time

        num_documents_processed = sum(task._stage_perf[1].num_items_processed for task in output_tasks)

        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        logger.success(f"Processed {num_documents_processed} documents")
        success = True

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        output_tasks = []
        run_time_taken = time.perf_counter() - run_start_time
        num_documents_processed = 0
        success = False

    return {
        "params": {
            "executor": executor_name,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "benchmark_results_path": str(benchmark_results_path),
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            "num_documents_processed": num_documents_processed,
            "num_output_tasks": len(output_tasks),
            "throughput_docs_per_sec": num_documents_processed / run_time_taken if run_time_taken > 0 else 0,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Modify benchmark")
    # Paths
    parser.add_argument("--benchmark-results-path", type=Path, required=True, help="Path to benchmark results")
    parser.add_argument("--input-path", required=True, type=Path, help="Path to input data")
    parser.add_argument(
        "--output-path", default=Path("./modified_output"), type=Path, help="Output directory for results"
    )
    # Executor
    parser.add_argument("--executor", default="ray_data", choices=["xenna", "ray_data"], help="Executor to use")
    parser.add_argument(
        "--dataset-size-gb", type=float, default=None, help="Limit input to approximately this many GB of files"
    )
    parser.add_argument("--input-filetype", default="jsonl", choices=["parquet", "jsonl"], help="Input file format")

    args = parser.parse_args()

    logger.info("=== Modify Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    results = {
        "params": vars(args),
        "metrics": {
            "is_success": False,
        },
        "tasks": [],
    }
    try:
        results = run_modify_benchmark(
            input_path=args.input_path,
            output_path=args.output_path,
            executor_name=args.executor,
            benchmark_results_path=args.benchmark_results_path,
            dataset_size_gb=args.dataset_size_gb,
            input_filetype=args.input_filetype,
        )
    finally:
        write_benchmark_results(results, args.benchmark_results_path)

    # Return proper exit code based on success
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
