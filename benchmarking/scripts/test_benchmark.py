#!/usr/bin/env python3
"""Demo benchmarking script for nightly benchmarking framework.

This script runs a demo benchmark with comprehensive metrics collection
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

#from nemo_curator.stages.file_partitioning import FilePartitioningStage
#from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow
#from nemo_curator.tasks import EmptyTask


def run_demo_benchmark(  # noqa: PLR0913
    input_path: str,
    output_path: str,
    benchmark_results_path: str,
    executor_name: str,
) -> dict[str, Any]:
    """Run the demo benchmark and collect comprehensive metrics."""

    # Setup executor
    if executor_name == "ray_data":
        from nemo_curator.backends.experimental.ray_data import RayDataExecutor

        executor = RayDataExecutor()
        #from ray.data import DataContext
        #DataContext.get_current().target_max_block_size = 1

    elif executor_name == "xenna":
        from nemo_curator.backends.xenna import XennaExecutor

        executor = XennaExecutor()
    else:
        msg = f"Executor {executor_name} not supported"
        raise ValueError(msg)

    # Ensure output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    logger.info("Starting demo benchmark")
    run_start_time = time.perf_counter()

    try:
        time.sleep(10)
        output_tasks = []
        run_time_taken = time.perf_counter() - run_start_time
        num_removed = 0
        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        success = True
    except Exception as e:  # noqa: BLE001
        logger.error(f"Benchmark failed: {e}")
        output_tasks = []
        success = False

    return {
        "params": {
            "executor": executor_name,
            "input_path": input_path,
            "output_path": output_path,
        },
        "metrics": {
            "is_success": success,
            "time_taken": run_time_taken,
            "num_removed": num_removed,
            "num_output_tasks": len(output_tasks),
        },
        "tasks": output_tasks,
    }


# FIXME: since the benchmarking framework depends on these files being present for all benchmark scripts,
# the framework should provide a utility (essentially, this function) to ensure they are written correctly
# with the correct names, paths, etc.
def write_results(results: dict, output_path: str | None = None) -> None:
    """Write results to a file or stdout."""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_path, "params.json"), "w") as f:
        json.dump(results["params"], f, indent=2)
    with open(os.path.join(output_path, "metrics.json"), "w") as f:
        json.dump(results["metrics"], f, indent=2)
    with open(os.path.join(output_path, "tasks.pkl"), "wb") as f:
        pickle.dump(results["tasks"], f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo benchmark for nightly benchmarking")
    # Paths
    parser.add_argument("--input-path", required=True, help="Path to input data")
    parser.add_argument("--output-path", required=True, help="Output directory for results")
    # FIXME: the framework will always add this! Look into if this policy should be removed.
    parser.add_argument("--benchmark-results-path", required=True, help="Path to benchmark results")

    args = parser.parse_args()

    logger.info("=== Demo Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    try:
        results = run_demo_benchmark(
            input_path=args.input_path,
            output_path=args.output_path,
            benchmark_results_path=args.benchmark_results_path,
            executor_name="ray_data",
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
