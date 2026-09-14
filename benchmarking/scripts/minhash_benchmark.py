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

"""MinHash benchmarking script for the nightly benchmarking framework.

The benchmark supports both input paths accepted by ``MinHashStage``:

* ``FileGroupTask``: MinHash reads JSONL or Parquet files directly on the GPU.
* ``DocumentBatch``: a CPU JSONL or Parquet reader materializes each batch first.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
from loguru import logger
from utils import load_dataset_files, setup_executor, write_benchmark_results

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.fuzzy.minhash import MinHashStage
from nemo_curator.stages.deduplication.id_generator import (
    create_id_generator_actor,
    kill_id_generator_actor,
)
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.tasks.utils import TaskPerfUtils

InputTaskType = Literal["DocumentBatch", "FileGroupTask"]


def _parse_json_object(value: str) -> dict[str, Any]:
    """Parse a command-line JSON object."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        msg = f"Expected a JSON object, got invalid JSON: {error}"
        raise argparse.ArgumentTypeError(msg) from error
    if not isinstance(parsed, dict):
        msg = "Expected a JSON object"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _dataset_ratio(value: str) -> float:
    """Parse a dataset ratio in the interval (0, 1]."""
    ratio = float(value)
    if not 0 < ratio <= 1:
        msg = "dataset size ratio must be greater than 0 and at most 1"
        raise argparse.ArgumentTypeError(msg)
    return ratio


def _build_pipeline(  # noqa: PLR0913
    input_files: list[str],
    output_path: Path,
    input_task_type: InputTaskType,
    input_filetype: Literal["jsonl", "parquet"],
    files_per_partition: int | None,
    blocksize: str | None,
    text_field: str,
    minhash_field: str,
    char_ngrams: int,
    num_hashes: int,
    seed: int,
    use_64bit_hash: bool,
    normalize_text: bool,
    read_kwargs: dict[str, Any],
    write_kwargs: dict[str, Any],
    pool: bool,
    minhash_num_workers: int | None,
    minhash_ray_data_initial_workers: int | None,
) -> Pipeline:
    """Build a MinHash pipeline for the requested input task type."""
    if input_task_type == "DocumentBatch":
        reader_cls = JsonlReader if input_filetype == "jsonl" else ParquetReader
        input_stage = reader_cls(
            file_paths=input_files,
            files_per_partition=files_per_partition,
            blocksize=blocksize,
            fields=[text_field],
            read_kwargs=read_kwargs,
            _generate_ids=True,
        )
        minhash_read_format = None
    else:
        input_stage = FilePartitioningStage(
            file_paths=input_files,
            files_per_partition=files_per_partition,
            blocksize=blocksize,
            file_extensions=None,
            storage_options=read_kwargs.get("storage_options"),
        )
        minhash_read_format = input_filetype

    minhash_stage = MinHashStage(
        output_path=str(output_path),
        text_field=text_field,
        minhash_field=minhash_field,
        char_ngrams=char_ngrams,
        num_hashes=num_hashes,
        seed=seed,
        use_64bit_hash=use_64bit_hash,
        normalize_text=normalize_text,
        read_format=minhash_read_format,
        read_kwargs=read_kwargs,
        write_kwargs=write_kwargs,
        pool=pool,
    )
    if minhash_num_workers is not None:
        minhash_stage = minhash_stage.with_(num_workers=minhash_num_workers)
    if minhash_ray_data_initial_workers is not None:
        minhash_stage = minhash_stage.with_(
            ray_stage_spec={RayStageSpecKeys.INITIAL_WORKERS: minhash_ray_data_initial_workers}
        )
    return Pipeline(name="minhash_benchmark_pipeline", stages=[input_stage, minhash_stage])


def _count_output_documents(output_tasks: list[Any]) -> int:
    """Count output documents from Parquet metadata without reading signatures."""
    return sum(pq.ParquetFile(path).metadata.num_rows for task in output_tasks for path in task.data)


def run_minhash_benchmark(  # noqa: PLR0913
    input_path: Path,
    output_path: Path,
    executor: str = "ray_actors",
    dataset_size_ratio: float = 1.0,
    input_task_type: InputTaskType = "FileGroupTask",
    input_filetype: Literal["jsonl", "parquet"] = "jsonl",
    files_per_partition: int | None = None,
    blocksize: str | None = None,
    text_field: str = "text",
    minhash_field: str = "_minhash_signature",
    char_ngrams: int = 24,
    num_hashes: int = 260,
    seed: int = 42,
    use_64bit_hash: bool = False,
    normalize_text: bool = False,
    read_kwargs: dict[str, Any] | None = None,
    write_kwargs: dict[str, Any] | None = None,
    pool: bool = True,
    minhash_num_workers: int | None = None,
    minhash_ray_data_initial_workers: int | None = None,
    **kwargs: object,  # noqa: ARG001
) -> dict[str, Any]:
    """Run MinHash over a whole-file fraction of a JSONL or Parquet dataset."""
    if minhash_ray_data_initial_workers is not None:
        if executor != "ray_data":
            msg = "minhash_ray_data_initial_workers is only supported by the ray_data executor"
            raise ValueError(msg)
        if minhash_num_workers is not None:
            msg = "minhash_num_workers and minhash_ray_data_initial_workers are mutually exclusive"
            raise ValueError(msg)

    input_path = input_path.absolute()
    output_path = output_path.absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    read_kwargs = read_kwargs or {}
    write_kwargs = write_kwargs or {}

    input_files = load_dataset_files(
        input_path,
        dataset_ratio=dataset_size_ratio,
        keep_extensions=input_filetype,
    )
    if not input_files:
        msg = (
            f"Dataset ratio {dataset_size_ratio} selected no {input_filetype} files under {input_path}. "
            "Increase --dataset-size-ratio."
        )
        raise ValueError(msg)

    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Selected {len(input_files)} files ({dataset_size_ratio:.2%} target ratio)")
    logger.info(f"MinHash input task type: {input_task_type}")
    logger.info(f"Executor: {executor}")

    pipeline = _build_pipeline(
        input_files=input_files,
        output_path=output_path,
        input_task_type=input_task_type,
        input_filetype=input_filetype,
        files_per_partition=files_per_partition,
        blocksize=blocksize,
        text_field=text_field,
        minhash_field=minhash_field,
        char_ngrams=char_ngrams,
        num_hashes=num_hashes,
        seed=seed,
        use_64bit_hash=use_64bit_hash,
        normalize_text=normalize_text,
        read_kwargs=read_kwargs,
        write_kwargs=write_kwargs,
        pool=pool,
        minhash_num_workers=minhash_num_workers,
        minhash_ray_data_initial_workers=minhash_ray_data_initial_workers,
    )
    executor_obj = setup_executor(executor)

    id_generator_created = False
    run_start_time = time.perf_counter()
    try:
        create_id_generator_actor()
        id_generator_created = True
        output_tasks = pipeline.run(executor_obj)
    finally:
        if id_generator_created:
            kill_id_generator_actor()
    run_time_taken = time.perf_counter() - run_start_time

    num_documents_processed = _count_output_documents(output_tasks)
    task_metrics = TaskPerfUtils.aggregate_task_metrics(output_tasks)
    input_prep_metric_prefix = (
        "MinHashStage_custom.minhash_document_batch_to_cudf_time"
        if input_task_type == "DocumentBatch"
        else "MinHashStage_custom.minhash_file_read_time"
    )

    logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
    logger.success(f"Processed {num_documents_processed} documents")

    return {
        "metrics": {
            "is_success": True,
            "time_taken_s": run_time_taken,
            "num_input_files": len(input_files),
            "num_output_tasks": len(output_tasks),
            "num_documents_processed": num_documents_processed,
            "throughput_docs_per_sec": (num_documents_processed / run_time_taken if run_time_taken > 0 else 0),
            "minhash_input_prep_worker_time_s_sum": task_metrics.get(f"{input_prep_metric_prefix}_sum", 0),
            "minhash_compute_worker_time_s_sum": task_metrics.get("MinHashStage_custom.minhash_compute_time_sum", 0),
            "minhash_write_worker_time_s_sum": task_metrics.get("MinHashStage_custom.minhash_write_time_sum", 0),
            "minhash_input_prep_worker_time_s_mean": task_metrics.get(f"{input_prep_metric_prefix}_mean", 0),
            "minhash_compute_worker_time_s_mean": task_metrics.get("MinHashStage_custom.minhash_compute_time_mean", 0),
            "minhash_write_worker_time_s_mean": task_metrics.get("MinHashStage_custom.minhash_write_time_mean", 0),
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MinHash benchmark for nightly benchmarking")
    parser.add_argument("--benchmark-results-path", type=Path, required=True, help="Path to benchmark results")
    parser.add_argument("--input-path", type=Path, required=True, help="Path to input data")
    parser.add_argument("--output-path", type=Path, required=True, help="Output directory for MinHash files")
    parser.add_argument(
        "--executor",
        default="ray_actors",
        choices=["ray_actors", "xenna", "ray_data"],
        help="Pipeline executor",
    )
    parser.add_argument(
        "--dataset-size-ratio",
        type=_dataset_ratio,
        default=1.0,
        help="Target fraction of input bytes to process, selecting whole files in order",
    )
    parser.add_argument(
        "--input-task-type",
        choices=["DocumentBatch", "FileGroupTask"],
        default="FileGroupTask",
        help="Use a CPU reader first (DocumentBatch) or let MinHash read file groups directly",
    )
    parser.add_argument(
        "--input-filetype",
        choices=["jsonl", "parquet"],
        default="jsonl",
        help="Input file format",
    )

    partitioning = parser.add_mutually_exclusive_group()
    partitioning.add_argument("--files-per-partition", type=int, help="Number of files per input task")
    partitioning.add_argument("--blocksize", help="Target input task size, for example 512MiB")

    parser.add_argument("--text-field", default="text", help="Field containing document text")
    parser.add_argument(
        "--minhash-field",
        default="_minhash_signature",
        help="Output field containing MinHash signatures",
    )
    parser.add_argument("--char-ngrams", type=int, default=24, help="Character n-gram width")
    parser.add_argument("--num-hashes", type=int, default=260, help="Number of hashes per signature")
    parser.add_argument("--seed", type=int, default=42, help="MinHash permutation seed")
    parser.add_argument(
        "--use-64bit-hash",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use 64-bit hashes instead of 32-bit hashes",
    )
    parser.add_argument(
        "--normalize-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize text before computing MinHash signatures",
    )
    parser.add_argument(
        "--read-kwargs",
        type=_parse_json_object,
        default={},
        help="JSON object forwarded to the input reader",
    )
    parser.add_argument(
        "--write-kwargs",
        type=_parse_json_object,
        default={},
        help="JSON object forwarded to the MinHash Parquet writer",
    )
    parser.add_argument(
        "--pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the MinHash GPU memory pool",
    )
    parser.add_argument(
        "--minhash-num-workers",
        type=int,
        default=None,
        help="Fixed number of MinHash workers; executor-selected when omitted",
    )
    parser.add_argument(
        "--minhash-ray-data-initial-workers",
        type=int,
        default=None,
        help="Initial size of the autoscaling Ray Data MinHash actor pool",
    )

    args = parser.parse_args()
    logger.info("=== MinHash Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    results = {
        "params": vars(args),
        "metrics": {"is_success": False},
        "tasks": [],
    }
    try:
        results.update(run_minhash_benchmark(**vars(args)))
    finally:
        write_benchmark_results(results, args.benchmark_results_path)
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
