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

"""Semantic duplicate identification benchmarking script for nightly benchmarking framework.

This script runs semantic duplicate identification benchmarks with comprehensive metrics collection
using the SemanticDeduplicationWorkflow and logs results to configured sinks.

Assumes embeddings are already pre-generated in the input path.
"""

import argparse
import time
from pathlib import Path
from typing import Any

from loguru import logger
from utils import load_dataset_files, setup_executor, write_benchmark_results

from nemo_curator.stages.deduplication.semantic.workflow import SemanticDeduplicationWorkflow
from nemo_curator.tasks.utils import TaskPerfUtils


def run_semdedup_identification_benchmark(  # noqa: PLR0913
    input_path: str,
    cache_path: str,
    output_path: str,
    executor: str = "xenna",
    dataset_size_ratio: float = 1,
    n_clusters: int = 1000,
    id_field: str = "id",
    embedding_field: str = "embeddings",
    embedding_dim: int | None = None,
    input_filetype: str = "parquet",
    eps: float = 0.01,
    which_to_keep: str = "hard",
    pairwise_batch_size: int = 1024,
    **kwargs,  # noqa: ARG001
) -> dict[str, Any]:
    """Run the semantic duplicate identification benchmark and collect comprehensive metrics.

    Args:
        input_path: Path to input data containing pre-generated embeddings
        cache_path: Path to cache directory for intermediate results (kmeans, pairwise)
        output_path: Output directory for duplicate identification results
        dataset_size_ratio: Ratio of dataset to process
        executor: Executor to use for pairwise
        n_clusters: Number of clusters for K-means clustering
        id_field: Name of the ID field in the data
        embedding_field: Name of the embedding field in the data
        embedding_dim: Embedding dimension (for memory estimation)
        input_filetype: Input file type ("parquet" or "jsonl")
        eps: Epsilon value for duplicate identification threshold (cosine_sim >= 1-eps)
        which_to_keep: Strategy for ranking within clusters ("hard", "easy", "random")
        pairwise_batch_size: Batch size for pairwise similarity computation
        **kwargs: Additional arguments (ignored)

    Returns:
        Dictionary containing metrics and workflow results
    """
    # Ensure directories
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(cache_path).mkdir(parents=True, exist_ok=True)

    logger.info("Starting semantic duplicate identification benchmark")
    run_start_time = time.perf_counter()

    # Create and run workflow
    workflow = SemanticDeduplicationWorkflow(
        input_path=load_dataset_files(input_path, dataset_ratio=dataset_size_ratio),
        output_path=output_path,
        cache_path=cache_path,
        n_clusters=n_clusters,
        id_field=id_field,
        embedding_field=embedding_field,
        embedding_dim=embedding_dim,
        input_filetype=input_filetype,
        eps=eps,
        which_to_keep=which_to_keep,
        pairwise_batch_size=pairwise_batch_size,
    )

    # Run the workflow, extract metrics from the WorkflowRunResult object
    executor_obj = setup_executor(executor)
    workflow_run_result = workflow.run(pairwise_executor=executor_obj)

    run_time_taken = time.perf_counter() - run_start_time
    task_metrics = TaskPerfUtils.aggregate_task_metrics(workflow_run_result)

    # Extract metrics from workflow result
    workflow_total_time = workflow_run_result.metadata.get("total_time")
    kmeans_time = workflow_run_result.metadata.get("kmeans_time")
    pairwise_time = workflow_run_result.metadata.get("pairwise_time")
    num_duplicates = workflow_run_result.metadata.get("num_duplicates")

    # Calculate percentage times
    pairwise_percent_time = None
    if workflow_total_time:
        # we get read / fit / write time from task_metrics
        kmeans_read_time = task_metrics.get("kmeans_KMeansStage_custom.kmeans_read_time_mean", 0)
        kmeans_write_time = task_metrics.get("kmeans_KMeansStage_custom.kmeans_write_time_mean", 0)
        kmeans_fit_predict_time = task_metrics.get("kmeans_KMeansStage_custom.kmeans_fit_predict_time_mean", 0)
        # this is different than kmeans_time because kmeans_time also includes setting up actors
        # while this is just sum of mean time taken across actors across the three steps
        _kmeans_time_taken = kmeans_read_time + kmeans_write_time + kmeans_fit_predict_time

        kmeans_read_percent_time = round((kmeans_read_time / _kmeans_time_taken) * 100, 2)
        kmeans_write_percent_time = round((kmeans_write_time / _kmeans_time_taken) * 100, 2)
        kmeans_fit_predict_percent_time = round((kmeans_fit_predict_time / _kmeans_time_taken) * 100, 2)

        kmeans_percent_time = round((kmeans_time / workflow_total_time) * 100, 2)
        pairwise_percent_time = round((pairwise_time / workflow_total_time) * 100, 2)

    logger.success(f"Benchmark completed in {run_time_taken:.2f}s")

    return {
        "metrics": {
            "is_success": True,
            "time_taken_s": run_time_taken,
            "workflow_total_time": workflow_total_time,
            "kmeans_time": kmeans_time,
            "pairwise_time": pairwise_time,
            "num_documents_processed": int(task_metrics.get("kmeans_KMeansStage_custom.num_rows_sum", 0)),
            "num_duplicates": num_duplicates,
            # within kmeans time
            "kmeans_read_percent_time": kmeans_read_percent_time,
            "kmeans_write_percent_time": kmeans_write_percent_time,
            "kmeans_fit_predict_percent_time": kmeans_fit_predict_percent_time,
            # between workflows
            "kmeans_percent_time": kmeans_percent_time,
            "pairwise_percent_time": pairwise_percent_time,
        },
        "tasks": workflow_run_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Semantic duplicate identification benchmark for nightly benchmarking"
    )
    parser.add_argument("--benchmark-results-path", required=True, help="Path to benchmark results")
    parser.add_argument("--input-path", required=True, help="Path to input data with pre-generated embeddings")
    parser.add_argument("--dataset-size-ratio", type=float, default=1, help="Ratio of dataset to process")
    parser.add_argument(
        "--executor", default="xenna", choices=["xenna", "ray_data"], help="Executor to use for pairwise"
    )
    parser.add_argument("--cache-path", required=True, help="Path to cache directory")
    parser.add_argument("--output-path", required=True, help="Output directory for results")
    parser.add_argument("--n-clusters", type=int, default=1000, help="Number of clusters for K-means")
    parser.add_argument("--id-field", default="id", help="ID field name in the data")
    parser.add_argument("--embedding-field", default="embeddings", help="Embedding field name in the data")
    parser.add_argument("--embedding-dim", type=int, default=None, help="Embedding dimension (optional)")
    parser.add_argument("--input-filetype", default="parquet", choices=["jsonl", "parquet"], help="Input filetype")
    parser.add_argument(
        "--eps",
        type=float,
        default=0.01,
        help="Epsilon threshold for duplicate identification (duplicates have cosine_sim >= 1-eps)",
    )
    parser.add_argument(
        "--which-to-keep",
        default="hard",
        choices=["hard", "easy", "random"],
        help="Strategy for ranking within clusters",
    )
    parser.add_argument(
        "--pairwise-batch-size", type=int, default=1024, help="Batch size for pairwise similarity computation"
    )

    args = parser.parse_args()

    logger.info("=== Semantic Duplicate Identification Benchmark Starting ===")
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
        result_dict.update(run_semdedup_identification_benchmark(**vars(args)))
        success_code = 0 if result_dict["metrics"]["is_success"] else 1
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
