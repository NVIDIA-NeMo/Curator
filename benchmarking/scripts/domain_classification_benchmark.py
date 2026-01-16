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

"""Domain classification benchmarking script.

This script runs domain classification benchmarks with comprehensive metrics collection
using various executors and logs results to configured sinks.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.classifiers import DomainClassifier
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.utils.file_utils import get_all_file_paths_and_size_under

# Import benchmarking utils which are currently only available directly from the Curator source tree.
# __file__ is expected to be <curator repo>/benchmarking/scripts/domain_classification_benchmark.py
_repo_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_dir))
from benchmarking.runner.utils import write_benchmark_results  # noqa: E402

_executor_map = {"ray_data": RayDataExecutor, "xenna": XennaExecutor}


def run_domain_classification_benchmark(
    input_path: str,
    output_path: str,
    executor: str,
    dataset_size_gb: float,
    model_inference_batch_size: int,
    **kwargs,  # noqa: ARG001
) -> dict[str, Any]:
    """Run the domain classification benchmark and collect comprehensive metrics."""

    input_path = Path(input_path)
    output_path = Path(output_path).absolute()

    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Dataset size: {dataset_size_gb} GB")
    logger.info(f"Batch size: {model_inference_batch_size}")
    logger.info(f"Executor: {executor}")

    logger.info("Starting domain classification benchmark")
    run_start_time = time.perf_counter()

    # Load input files
    input_files = load_dataset_files(input_path, dataset_size_gb)

    # Setup executor
    executor_obj = _executor_map[executor]()

    # Create and run pipeline
    pipeline = Pipeline(
        name="domain_classification_pipeline",
        stages=[
            ParquetReader(file_paths=input_files, files_per_partition=1, fields=["text"], _generate_ids=False),
            DomainClassifier(
                text_field="text",
                model_inference_batch_size=model_inference_batch_size,
            ),
            ParquetWriter(path=str(output_path), fields=["domain_pred"]),
        ],
    )
    output_tasks = pipeline.run(executor_obj)

    run_time_taken = time.perf_counter() - run_start_time

    # Calculate metrics
    num_documents_processed = sum(task.num_items for task in output_tasks)
    throughput_docs_per_sec = num_documents_processed / run_time_taken if run_time_taken > 0 else 0

    logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
    logger.success(f"Processed {num_documents_processed} documents")

    return {
        "metrics": {
            "is_success": True,
            "time_taken": run_time_taken,
            "num_documents_processed": num_documents_processed,
            "throughput_docs_per_sec": throughput_docs_per_sec,
        },
        "tasks": output_tasks,
    }


def load_dataset_files(dataset_path: Path, dataset_size_gb: float) -> list[str]:
    """Load the dataset files at the given path and return a subset of the files whose combined size is approximately the given size in GB."""
    input_files = get_all_file_paths_and_size_under(
        dataset_path, recurse_subdirectories=True, keep_extensions="parquet"
    )
    desired_size_bytes = (1024**3) * dataset_size_gb
    total_size = 0
    subset_files = []
    for file, size in input_files:
        if size + total_size > desired_size_bytes:
            break
        else:
            subset_files.append(file)
            total_size += size

    return subset_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Domain classification benchmark for nightly benchmarking")
    parser.add_argument("--benchmark-results-path", required=True, help="Path to benchmark results")
    parser.add_argument("--input-path", required=True, help="Path to input data")
    parser.add_argument("--output-path", default="./domain_classification_output", help="Output directory for results")
    parser.add_argument("--executor", default="ray_data", choices=["xenna", "ray_data"], help="Executor to use")
    parser.add_argument("--dataset-size-gb", type=float, required=True, help="Size of dataset to process in GB")
    parser.add_argument("--model-inference-batch-size", type=int, default=1024, help="Batch size for model inference")

    args = parser.parse_args()

    logger.info("=== Domain Classification Benchmark Starting ===")
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
        result_dict.update(run_domain_classification_benchmark(**vars(args)))
        success_code = 0 if result_dict["metrics"]["is_success"] else 1
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
