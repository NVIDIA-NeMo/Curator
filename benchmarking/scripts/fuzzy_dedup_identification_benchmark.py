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

# ruff: noqa: ERA001

"""Duplicate identification logic benchmarking script for nightly benchmarking framework.

This script runs duplicate identification benchmarks with comprehensive metrics collection
using TaskPerfUtils and logs results to configured sinks.
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow


def run_duplicate_identification_benchmark(  # noqa: PLR0913
    input_path: str,
    cache_path: str,
    output_path: str,
    input_filetype: str = "jsonl",
    bands_per_iteration: int = 20,  # Number of bands to shuffle concurrently during LSH. Higher values have higher memory pressure but can reduce runtime
    text_field: str = "text",
    input_blocksize: str = "1.5GiB",
) -> dict[str, Any]:
    """Run the duplicate identification benchmark and collect comprehensive metrics."""

    # Ensure directories
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(cache_path).mkdir(parents=True, exist_ok=True)

    logger.info("Starting duplicate identification benchmark")
    run_start_time = time.perf_counter()

    try:
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
        # TODO: Uncomment this when the pipeline is fixed
        # output_tasks = pipeline.run(executor=executor, initial_tasks=None)
        output_tasks = []
        workflow.run(initial_tasks=None)
        run_time_taken = time.perf_counter() - run_start_time

        success = True
        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")

    except Exception as e:  # noqa: BLE001
        logger.error(f"Benchmark failed: {e}")
        output_tasks = []
        run_time_taken = time.perf_counter() - run_start_time
        success = False

    return {
        "params": {
            "input_path": input_path,
            "cache_path": cache_path,
            "output_path": output_path,
            "input_filetype": input_filetype,
            "bands_per_iteration": bands_per_iteration,
            "text_field": text_field,
            "input_blocksize": input_blocksize,
        },
        "metrics": {
            "is_success": success,
            "time_taken": run_time_taken,
            "num_output_tasks": len(output_tasks),
        },
        "tasks": output_tasks,
    }


def write_results(results: dict, output_path: str | None = None) -> None:
    """Write results to a file or stdout."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_path, "params.json"), "w") as f:
        json.dump(results["params"], f, indent=2)
    with open(os.path.join(output_path, "metrics.json"), "w") as f:
        json.dump(results["metrics"], f, indent=2)
    with open(os.path.join(output_path, "tasks.pkl"), "wb") as f:
        pickle.dump(results["tasks"], f)


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

    try:
        results = run_duplicate_identification_benchmark(
            input_path=args.input_path,
            cache_path=args.cache_path,
            output_path=args.output_path,
            input_filetype=args.input_filetype,
            bands_per_iteration=args.bands_per_iteration,
            text_field=args.text_field,
            input_blocksize=args.input_blocksize,
        )

    except Exception as e:  # noqa: BLE001
        print(f"Benchmark failed: {e}")
        results = {
            "params": vars(args),
            "metrics": {
                "is_success": False,
            },
            "tasks": [],
        }
    finally:
        write_results(results, args.benchmark_results_path)

    # Return proper exit code based on success
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
