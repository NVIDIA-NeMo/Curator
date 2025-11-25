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

"""Embedding generation benchmarking script.

This script runs embedding generation benchmarks with comprehensive metrics collection
using various executors and logs results to configured sinks.
"""
# ruff: noqa: ERA001

import argparse
import json
import os
import pickle
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.embedders import EmbeddingCreatorStage
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.stages.text.models.utils import format_name_with_suffix
from nemo_curator.utils.file_utils import get_all_file_paths_and_size_under

_executor_map = {"ray_data": RayDataExecutor, "xenna": XennaExecutor}


def run_embedding_generation_benchmark(  # noqa: PLR0913
    input_path: Path,
    output_path: Path,
    executor_name: str,
    dataset_size_gb: float,
    model_identifier: str,
    model_inference_batch_size: int,
    files_per_partition: int,
    use_id_generator: bool,
    benchmark_results_path: Path,
) -> dict[str, Any]:
    """Run the embedding generation benchmark and collect comprehensive metrics."""

    # Setup executor
    try:
        executor = _executor_map[executor_name]()
    except KeyError:
        msg = f"Executor {executor_name} not supported"
        raise ValueError(msg) from None

    # Ensure output directory
    output_path = output_path.absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting embedding generation benchmark")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Dataset size: {dataset_size_gb} GB")
    logger.info(f"Model: {model_identifier}")
    logger.info(f"Batch size: {model_inference_batch_size}")
    logger.debug(f"Executor: {executor}")

    run_start_time = time.perf_counter()

    try:
        logger.info("Running embedding generation pipeline...")
        if use_id_generator:
            from nemo_curator.stages.deduplication.id_generator import create_id_generator_actor

            create_id_generator_actor()

        input_files = load_dataset_files(input_path, dataset_size_gb)

        executor = RayDataExecutor() if executor_name == "ray_data" else XennaExecutor()
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embedding_creator = EmbeddingCreatorStage(
            text_field="raw_content", model_identifier=model_name, model_inference_batch_size=256
        ).with_({format_name_with_suffix(model_name, "_model"): {"resources": Resources(gpus=0.5)}})

        pipeline = Pipeline(
            name="embedding_generation_pipeline",
            stages=[
                ParquetReader(
                    file_paths=input_files,
                    files_per_partition=files_per_partition,
                    fields=["id", "raw_content"],
                    _generate_ids=use_id_generator,
                ),
                embedding_creator,
                ParquetWriter(path=str(os.path.join(output_path, "embeddings"))),
            ],
        )

        output_tasks = pipeline.run(executor)
        run_time_taken = time.perf_counter() - run_start_time

        if use_id_generator:
            from nemo_curator.stages.deduplication.id_generator import write_id_generator_to_disk

            id_generator_path = str(
                os.path.join(output_path, f"id_generator_{files_per_partition}fpp_{dataset_size_gb}gb.json")
            )
            logger.info(f"Writing ID generator to {id_generator_path}")
            write_id_generator_to_disk(id_generator_path)

        # task._metadata is a dictionary of metadata for the task, but will not be used here.
        # Instead simply use the num_items property of the task to get the number of documents processed.
        # TODO: can we get the number of embeddings generated?
        num_documents_processed = sum(task.num_items for task in output_tasks)

        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        logger.success(f"Processed {num_documents_processed} documents")
        # logger.success(f"Generated {num_embeddings_generated} embeddings")
        success = True

    except Exception as e:  # noqa: BLE001
        error_traceback = traceback.format_exc()
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        output_tasks = []
        run_time_taken = time.perf_counter() - run_start_time
        num_documents_processed = 0
        # num_embeddings_generated = 0
        # embedding_dimension = 0
        success = False

    return {
        "params": {
            "executor": executor_name,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "dataset_size_gb": dataset_size_gb,
            "model_identifier": model_identifier,
            "model_inference_batch_size": model_inference_batch_size,
            "benchmark_results_path": str(benchmark_results_path),
            "files_per_partition": files_per_partition,
            "use_id_generator": use_id_generator,
            **({"id_generator_path": id_generator_path} if use_id_generator else {}),
        },
        "metrics": {
            "is_success": success,
            "time_taken": run_time_taken,
            "num_documents_processed": num_documents_processed,
            # "num_embeddings_generated": num_embeddings_generated,
            # "embedding_dimension": embedding_dimension,
            "num_output_tasks": len(output_tasks),
            "throughput_docs_per_sec": num_documents_processed / run_time_taken if run_time_taken > 0 else 0,
            # "throughput_embeddings_per_sec": num_embeddings_generated / run_time_taken if run_time_taken > 0 else 0,
        },
        "tasks": output_tasks,
    }


def write_results(results: dict[str, Any], output_path: Path) -> None:
    """Write results to files required by the benchmarking framework at the given path."""
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "params.json").write_text(json.dumps(results["params"], indent=2))
    (output_path / "metrics.json").write_text(json.dumps(results["metrics"], indent=2))
    (output_path / "tasks.pkl").write_bytes(pickle.dumps(results["tasks"]))


def load_dataset_files(dataset_path: Path, dataset_size_gb: float | None = None) -> list[str] | str:
    """Load the dataset files at the given path and return a subset of the files whose combined size is approximately the given size in GB."""
    if dataset_size_gb is None:
        return str(dataset_path)
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
    parser = argparse.ArgumentParser(description="Embedding generation benchmark")
    # Paths
    parser.add_argument("--benchmark-results-path", type=Path, required=True, help="Path to benchmark results")
    parser.add_argument("--input-path", required=True, type=Path, help="Path to input data")
    parser.add_argument(
        "--output-path", default=Path("./embedding_generation_output"), type=Path, help="Output directory for results"
    )
    # Executor
    parser.add_argument("--executor", default="ray_data", choices=["xenna", "ray_data"], help="Executor to use")
    # Pipeline Specific
    parser.add_argument("--dataset-size-gb", type=float, required=True, help="Size of dataset to process in GB")
    parser.add_argument(
        "--model-identifier",
        type=str,
        required=True,
        help="Model identifier (e.g., sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument("--model-inference-batch-size", type=int, default=1024, help="Batch size for model inference")
    parser.add_argument("--files-per-partition", type=int, default=1, help="Files per partition")
    parser.add_argument("--use-id-generator", action="store_true", help="Use ID generator")

    args = parser.parse_args()

    logger.info("=== Embedding Generation Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    try:
        results = run_embedding_generation_benchmark(
            input_path=args.input_path,
            output_path=args.output_path,
            executor_name=args.executor,
            dataset_size_gb=args.dataset_size_gb,
            model_identifier=args.model_identifier,
            model_inference_batch_size=args.model_inference_batch_size,
            benchmark_results_path=args.benchmark_results_path,
            files_per_partition=args.files_per_partition,
            use_id_generator=args.use_id_generator,
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
        write_results(results, args.benchmark_results_path)

    # Return proper exit code based on success
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
