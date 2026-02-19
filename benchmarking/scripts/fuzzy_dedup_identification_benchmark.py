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

# ruff: noqa: ERA001

"""Duplicate identification logic benchmarking script for nightly benchmarking framework.

This script runs duplicate identification benchmarks with comprehensive metrics collection
using TaskPerfUtils and logs results to configured sinks.
"""

import argparse
import time
from pathlib import Path
from typing import Any

from loguru import logger
from utils import write_benchmark_results

from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow


def run_duplicate_identification_benchmark(  # noqa: PLR0913
    input_path: str,
    cache_path: str,
    output_path: str,
    input_filetype: str = "jsonl",
    bands_per_iteration: int = 20,  # Number of bands to shuffle concurrently during LSH. Higher values have higher memory pressure but can reduce runtime
    text_field: str = "text",
    input_blocksize: str = "1.5GiB",
    **kwargs,  # noqa: ARG001
) -> dict[str, Any]:
    """Run the duplicate identification benchmark and collect comprehensive metrics."""

    # Ensure directories
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(cache_path).mkdir(parents=True, exist_ok=True)

    logger.info("Starting duplicate identification benchmark")
    run_start_time = time.perf_counter()

    # Create and run workflow-backed pipeline
    workflow = FuzzyDeduplicationWorkflow(
        input_path=input_path,
        cache_path=cache_path,
        output_path=output_path,
        input_filetype=input_filetype,
        bands_per_iteration=bands_per_iteration,
        text_field=text_field,
        input_blocksize=input_blocksize,
    )

    # Run the workflow, extract metrics from the WorkflowRunResult object
    workflow_run_result = workflow.run(initial_tasks=None)

    run_time_taken = time.perf_counter() - run_start_time

    # Extract metrics
    workflow_total_time = workflow_run_result.metadata.get("total_time")
    minhash_time = workflow_run_result.metadata.get("minhash_time")
    lsh_time = workflow_run_result.metadata.get("lsh_time")
    connected_components_time = workflow_run_result.metadata.get("connected_components_pipeline_time")
    num_duplicates = workflow_run_result.metadata.get("num_duplicates")

    minhash_percent_time = None
    lsh_percent_time = None
    connected_components_percent_time = None
    if workflow_total_time:
        if minhash_time is not None:
            minhash_percent_time = round((minhash_time / workflow_total_time) * 100, 2)
        if lsh_time is not None:
            lsh_percent_time = round((lsh_time / workflow_total_time) * 100, 2)
        if connected_components_time is not None:
            connected_components_percent_time = round((connected_components_time / workflow_total_time) * 100, 2)

    logger.success(f"Benchmark completed in {run_time_taken:.2f}s")

    return {
        "metrics": {
            "is_success": True,
            "time_taken_s": run_time_taken,
            "workflow_total_time": workflow_total_time,
            "minhash_time": minhash_time,
            "lsh_time": lsh_time,
            "connected_components_time": connected_components_time,
            "num_duplicates": num_duplicates,
            "minhash_percent_time": minhash_percent_time,
            "lsh_percent_time": lsh_percent_time,
            "connected_components_percent_time": connected_components_percent_time,
        },
        "tasks": workflow_run_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Duplicate identification benchmark for nightly benchmarking")
    parser.add_argument("--benchmark-results-path", required=True, help="Path to benchmark results")
    parser.add_argument("--input-path", required=True, help="Path to input data")
    parser.add_argument("--cache-path", required=True, help="Path to cache directory")
    parser.add_argument("--output-path", required=True, help="Output directory for results")
    parser.add_argument("--input-filetype", default="jsonl", choices=["jsonl", "parquet"], help="Input filetype")
    parser.add_argument(
        "--bands-per-iteration", type=int, default=20, help="Bands per iteration (for LSH deduplication)"
    )
    parser.add_argument("--text-field", default="text", help="Text field to use for duplicate identification")
    parser.add_argument(
        "--input-blocksize", type=str, default="1.5GiB", help="Target partition size for input data (e.g. '512MB')"
    )

    args = parser.parse_args()

    logger.info("=== Duplicate Identification Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    success_code = 1  # assume failure until benchmark succeeds

    # This dictionary will contain benchmark metadata and results, written to files for the benchmark framework to read.
    result_dict = {
        "params": vars(args),
        "metrics": {
            "is_success": False,
        },
        "tasks": [],
    }
    try:
        result_dict.update(run_duplicate_identification_benchmark(**vars(args)))
        success_code = 0 if result_dict["metrics"]["is_success"] else 1
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
