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

"""Exact duplicate identification benchmarking script for nightly benchmarking framework.

This script runs exact duplicate identification benchmarks with comprehensive metrics collection
and logs results to configured sinks.
"""

import argparse
import time
from pathlib import Path
from typing import Any

from loguru import logger
from utils import write_benchmark_results

from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow


def run_exact_duplicate_identification_benchmark(  # noqa: PLR0913
    input_path: str,
    output_path: str,
    input_filetype: str = "jsonl",
    text_field: str = "text",
    input_blocksize: str = "2GiB",
    assign_id: bool = True,
    id_field: str | None = None,
) -> dict[str, Any]:
    """Run the exact duplicate identification benchmark and collect comprehensive metrics."""

    # Ensure directories
    Path(output_path).mkdir(parents=True, exist_ok=True)

    logger.info("Starting exact duplicate identification benchmark")
    run_start_time = time.perf_counter()

    try:
        # Create and run workflow-backed pipeline
        workflow = ExactDeduplicationWorkflow(
            input_path=input_path,
            output_path=output_path,
            input_filetype=input_filetype,
            text_field=text_field,
            input_blocksize=input_blocksize,
            assign_id=assign_id,
            id_field=id_field,
        )
        workflow_result = workflow.run(initial_tasks=None)
        run_time_taken = time.perf_counter() - run_start_time

        # Extract metrics from workflow result metadata
        num_duplicates = workflow_result.metadata.get("num_duplicates", 0)
        identification_time = workflow_result.metadata.get("identification_time", 0.0)
        input_filegroups_time = workflow_result.metadata.get("input_filegroups_time", 0.0)

        success = True
        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        logger.success(f"Found {num_duplicates} exact duplicates")

    except Exception as e:  # noqa: BLE001
        logger.error(f"Benchmark failed: {e}")
        success = False
        run_time_taken = time.perf_counter() - run_start_time
        num_duplicates = 0
        identification_time = 0.0
        input_filegroups_time = 0.0

    return {
        "params": {
            "input_path": input_path,
            "output_path": output_path,
            "input_filetype": input_filetype,
            "text_field": text_field,
            "input_blocksize": input_blocksize,
            "assign_id": assign_id,
            "id_field": id_field,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            "num_duplicates": num_duplicates,
            "identification_time_s": identification_time,
            "input_filegroups_time_s": input_filegroups_time,
        },
        "tasks": workflow_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Exact duplicate identification benchmark for nightly benchmarking")
    parser.add_argument("--benchmark-results-path", required=True, help="Path to benchmark results")
    parser.add_argument("--input-path", required=True, help="Path to input data")
    parser.add_argument("--output-path", required=True, help="Output directory for results")
    parser.add_argument("--input-filetype", default="jsonl", choices=["jsonl", "parquet"], help="Input filetype")
    parser.add_argument("--text-field", default="text", help="Text field to use for duplicate identification")
    parser.add_argument(
        "--input-blocksize", type=str, default="2GiB", help="Target partition size for input data (e.g. '2GiB')"
    )
    parser.add_argument(
        "--assign-id",
        action="store_true",
        default=True,
        help="Whether to automatically assign a unique id to each document",
    )
    parser.add_argument(
        "--no-assign-id",
        action="store_false",
        dest="assign_id",
        help="Use existing id field instead of assigning new ids",
    )
    parser.add_argument(
        "--id-field",
        type=str,
        default=None,
        help="Existing id field name if not automatically assigning a new id",
    )

    args = parser.parse_args()

    logger.info("=== Exact Duplicate Identification Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    try:
        results = run_exact_duplicate_identification_benchmark(
            input_path=args.input_path,
            output_path=args.output_path,
            input_filetype=args.input_filetype,
            text_field=args.text_field,
            input_blocksize=args.input_blocksize,
            assign_id=args.assign_id,
            id_field=args.id_field,
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
        write_benchmark_results(results, args.benchmark_results_path)

    # Return proper exit code based on success
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
