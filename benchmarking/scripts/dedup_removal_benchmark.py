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

"""Removal logic benchmarking script for nightly benchmarking framework.

This script runs removal benchmarks with comprehensive metrics collection
using TaskPerfUtils and logs results to configured sinks.
"""

import argparse
import time
from pathlib import Path
from typing import Any

from loguru import logger
from utils import write_benchmark_results

from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow
from nemo_curator.tasks import EmptyTask


def run_removal_benchmark(  # noqa: PLR0913
    input_path: str,
    ids_to_remove_path: str,
    output_path: str,
    executor_name: str,
    input_filetype: str = "jsonl",
    output_filetype: str = "parquet",
    id_field: str = "_curator_dedup_id",
    duplicate_id_field: str = "id",
    files_per_partition: int | None = None,
    blocksize: str | None = None,
    id_generator_path: str | None = None,
    use_initial_tasks: bool = False,
    limit: int | None = None,
    use_ray_data_settings: bool = False,
) -> dict[str, Any]:
    """Run the removal benchmark and collect comprehensive metrics."""

    # Setup executor
    if executor_name == "ray_data":
        from nemo_curator.backends.experimental.ray_data import RayDataExecutor

        executor = RayDataExecutor()
        if use_ray_data_settings:
            from ray.data import DataContext

            DataContext.get_current().target_max_block_size = 1

    elif executor_name == "xenna":
        from nemo_curator.backends.xenna import XennaExecutor

        executor = XennaExecutor()
    else:
        msg = f"Executor {executor_name} not supported"
        raise ValueError(msg)

    # Ensure output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    logger.info("Starting removal benchmark")
    run_start_time = time.perf_counter()

    try:
        # Validate partitioning: exactly one of files_per_partition or blocksize must be provided
        if (files_per_partition is None) == (blocksize is None):
            msg = "Exactly one of --files-per-partition or --blocksize must be provided"
            raise ValueError(msg)  # noqa: TRY301

        # Create and run workflow-backed pipeline
        workflow = TextDuplicatesRemovalWorkflow(
            input_path=input_path,
            ids_to_remove_path=ids_to_remove_path,
            output_path=output_path,
            input_filetype=input_filetype,  # jsonl or parquet
            input_id_field=id_field,
            input_files_per_partition=files_per_partition,
            input_blocksize=blocksize,
            input_task_limit=limit,
            ids_to_remove_duplicate_id_field=duplicate_id_field,
            output_filetype=output_filetype,
            id_generator_path=id_generator_path,
            input_kwargs={},
            output_kwargs={},
        )

        initial_tasks = None
        if use_initial_tasks:
            logger.info("Using initial tasks produced by FilePartitioningStage on driver")
            partitioner = FilePartitioningStage(
                file_paths=input_path,
                files_per_partition=files_per_partition,
                blocksize=blocksize,
                file_extensions=[".jsonl", ".json", ".parquet"],
                storage_options=None,
            )
            initial_tasks = partitioner.process(EmptyTask)
            log_msg = f"Initial tasks: {len(initial_tasks)}"
            if limit:
                initial_tasks = initial_tasks[:limit]
                log_msg += f" (limited to {limit})"
            logger.info(log_msg)

        output_tasks = workflow.run(executor, initial_tasks=initial_tasks)
        run_time_taken = time.perf_counter() - run_start_time

        # Calculate removal statistics
        num_removed = sum(task._metadata.get("num_removed", 0) for task in output_tasks if hasattr(task, "_metadata"))
        logger.success(f"Benchmark completed in {run_time_taken:.2f}s, removed {num_removed} documents")
        success = True
    except Exception as e:  # noqa: BLE001
        logger.error(f"Benchmark failed: {e}")
        output_tasks = []
        run_time_taken = time.perf_counter() - run_start_time
        num_removed = 0
        success = False

    return {
        "params": {
            "executor": executor_name,
            "input_path": input_path,
            "input_filetype": input_filetype,
            "ids_to_remove_path": ids_to_remove_path,
            "output_filetype": output_filetype,
            "id_field": id_field,
            "duplicate_id_field": duplicate_id_field,
            "files_per_partition": files_per_partition,
            "blocksize": blocksize,
            "id_generator_path": id_generator_path,
            "use_initial_tasks": use_initial_tasks,
            "limit": limit,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            "num_removed": num_removed,
            "num_output_tasks": len(output_tasks),
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Removal logic benchmark for nightly benchmarking")
    # Paths
    parser.add_argument(
        "--benchmark-results-path", required=True, help="Path to benchmark results"
    )  # we should write params.json / metrics.json / tasks.pkl here
    parser.add_argument("--input-path", required=True, help="Path to input data")
    parser.add_argument("--ids-to-remove-path", required=True, help="Path to parquet file with IDs to remove")
    parser.add_argument("--output-path", required=True, help="Output directory for results")
    # Executor
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"], help="Executor to use")
    # Pipeline Specific
    parser.add_argument("--input-filetype", default="jsonl", choices=["jsonl", "parquet"], help="Input filetype")
    parser.add_argument("--output-filetype", default="parquet", choices=["parquet", "jsonl"], help="Output filetype")
    parser.add_argument("--id-field", default="_curator_dedup_id", help="ID field in input data")
    parser.add_argument("--duplicate-id-field", default="id", help="ID field in removal file")
    parser.add_argument(
        "--files-per-partition",
        type=int,
        default=None,
        help="Files per partition (mutually exclusive with --blocksize)",
    )
    parser.add_argument(
        "--blocksize",
        type=str,
        default=None,
        help="Target partition size (e.g. '512MB', '1GiB'); mutually exclusive with --files-per-partition",
    )
    parser.add_argument("--id-generator-path", type=str, default=None, help="Path to ID generator JSON (optional)")
    parser.add_argument(
        "--use-initial-tasks",
        action="store_true",
        help="If set, pre-compute initial FileGroupTasks via FilePartitioningStage and pass to workflow",
    )
    parser.add_argument("--use-ray-data-settings", action="store_true", help="If set, use ray data settings")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of tasks to process")

    args = parser.parse_args()

    logger.info("=== Removal Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    try:
        results = run_removal_benchmark(
            input_path=args.input_path,
            ids_to_remove_path=args.ids_to_remove_path,
            output_path=args.output_path,
            executor_name=args.executor,
            input_filetype=args.input_filetype,
            output_filetype=args.output_filetype,
            id_field=args.id_field,
            duplicate_id_field=args.duplicate_id_field,
            files_per_partition=args.files_per_partition,
            blocksize=args.blocksize,
            id_generator_path=args.id_generator_path,
            use_initial_tasks=args.use_initial_tasks,
            limit=args.limit,
            use_ray_data_settings=args.use_ray_data_settings,
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
