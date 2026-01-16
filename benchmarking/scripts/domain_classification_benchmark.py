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

"""Domain classification benchmarking script.

This script runs domain classification benchmarks with comprehensive metrics collection
using various executors and logs results to configured sinks.
"""
# ruff: noqa: ERA001

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger
from utils import load_dataset_files, setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.classifiers import DomainClassifier
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter


def run_domain_classification_benchmark(  # noqa: PLR0913
    input_path: Path,
    output_path: Path,
    executor_name: str,
    dataset_size_gb: float,
    model_inference_batch_size: int,
    benchmark_results_path: Path,
) -> dict[str, Any]:
    """Run the domain classification benchmark and collect comprehensive metrics."""

    executor = setup_executor(executor_name)

    # Ensure output directory
    output_path = output_path.absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Dataset size: {dataset_size_gb} GB")
    logger.info(f"Batch size: {model_inference_batch_size}")
    logger.debug(f"Executor: {executor}")

    run_start_time = time.perf_counter()

    try:
        logger.info("Running domain classification pipeline...")

        input_files = load_dataset_files(input_path, dataset_size_gb)

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
        output_tasks = pipeline.run(executor)
        run_time_taken = time.perf_counter() - run_start_time

        # task._metadata is a dictionary of metadata for the task, but will not be used here.
        # Instead simply use the num_items property of the task to get the number of documents processed.
        # TODO: can we get the number of domains classified?
        num_documents_processed = sum(task.num_items for task in output_tasks)
        # num_domains_classified = 0

        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        logger.success(f"Processed {num_documents_processed} documents")
        success = True

    except Exception as e:  # noqa: BLE001
        error_traceback = traceback.format_exc()
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        output_tasks = []
        run_time_taken = time.perf_counter() - run_start_time
        num_documents_processed = 0
        # num_domains_classified = 0
        success = False

    return {
        "params": {
            "executor": executor_name,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "dataset_size_gb": dataset_size_gb,
            "model_inference_batch_size": model_inference_batch_size,
            "benchmark_results_path": str(benchmark_results_path),
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            "num_documents_processed": num_documents_processed,
            # "num_domains_classified": num_domains_classified,
            "num_output_tasks": len(output_tasks),
            "throughput_docs_per_sec": num_documents_processed / run_time_taken if run_time_taken > 0 else 0,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Domain classification benchmark")
    # Paths
    parser.add_argument("--benchmark-results-path", type=Path, required=True, help="Path to benchmark results")
    parser.add_argument("--input-path", required=True, type=Path, help="Path to input data")
    parser.add_argument(
        "--output-path", default=Path("./domain_classification_output"), type=Path, help="Output directory for results"
    )
    # Executor
    parser.add_argument("--executor", default="ray_data", choices=["xenna", "ray_data"], help="Executor to use")
    # Pipeline Specific
    parser.add_argument("--dataset-size-gb", type=float, required=True, help="Size of dataset to process in GB")
    parser.add_argument("--model-inference-batch-size", type=int, default=1024, help="Batch size for model inference")

    args = parser.parse_args()

    logger.info("=== Domain Classification Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    try:
        results = run_domain_classification_benchmark(
            input_path=args.input_path,
            output_path=args.output_path,
            executor_name=args.executor,
            dataset_size_gb=args.dataset_size_gb,
            model_inference_batch_size=args.model_inference_batch_size,
            benchmark_results_path=args.benchmark_results_path,
        )

    except Exception as e:  # noqa: BLE001
        error_traceback = traceback.format_exc()
        print(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
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
