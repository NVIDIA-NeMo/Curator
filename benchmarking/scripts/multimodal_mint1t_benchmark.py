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

"""Benchmark for multimodal MINT1T workflow: WebDataset -> filter -> parquet."""

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger
from utils import collect_parquet_output_metrics, setup_executor, write_benchmark_results

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.multimodal.io import MultimodalParquetWriter, WebdatasetReader
from nemo_curator.stages.multimodal.stages import MultimodalJpegAspectRatioFilterStage
from nemo_curator.tasks.utils import TaskPerfUtils


def create_pipeline(args: argparse.Namespace) -> Pipeline:
    read_kwargs = {}
    write_kwargs = {}
    if args.parquet_row_group_size is not None:
        write_kwargs["row_group_size"] = args.parquet_row_group_size
    if args.parquet_compression is not None:
        write_kwargs["compression"] = args.parquet_compression
    pipeline = Pipeline(
        name="multimodal_mint1t_benchmark",
        description="Benchmark: WebDataset MINT1T to multimodal parquet",
    )
    pipeline.add_stage(
        WebdatasetReader(
            source_id_field="pdf_name",
            file_paths=args.input_path,
            files_per_partition=args.files_per_partition,
            blocksize=args.input_blocksize,
            max_batch_bytes=args.output_max_batch_bytes,
            read_kwargs=read_kwargs,
            materialize_on_read=args.materialize_on_read,
        )
    )
    pipeline.add_stage(MultimodalJpegAspectRatioFilterStage(drop_invalid_rows=True))
    pipeline.add_stage(
        MultimodalParquetWriter(
            path=args.output_path,
            materialize_on_write=args.materialize_on_write,
            write_kwargs=write_kwargs,
            mode=args.mode,
        )
    )
    return pipeline


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    executor = setup_executor(args.executor)
    input_path = str(Path(args.input_path).absolute())
    output_path = Path(args.output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    output_tasks = []
    success = False
    try:
        pipeline = create_pipeline(args)
        logger.info("Pipeline:\n{}", pipeline.describe())
        output_tasks = pipeline.run(executor)
        success = True
    except Exception as e:  # noqa: BLE001
        logger.error("Benchmark failed: {}", e)
        logger.debug(traceback.format_exc())

    elapsed = time.perf_counter() - start
    output_metrics = collect_parquet_output_metrics(output_path)
    task_metrics = TaskPerfUtils.aggregate_task_metrics(output_tasks, prefix="task")
    writer_stats = {k: v for k, v in task_metrics.items() if "multimodal_" in k and "_writer" in k}
    logger.info("Writer stage stats: {}", writer_stats)
    rows = output_metrics["num_rows"]
    return {
        "params": {
            "executor": args.executor,
            "input_path": input_path,
            "output_path": str(output_path),
            "files_per_partition": args.files_per_partition,
            "input_blocksize": args.input_blocksize,
            "output_max_batch_bytes": args.output_max_batch_bytes,
            "materialize_on_read": args.materialize_on_read,
            "materialize_on_write": args.materialize_on_write,
            "parquet_row_group_size": args.parquet_row_group_size,
            "parquet_compression": args.parquet_compression,
            "mode": args.mode,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": elapsed,
            "throughput_rows_per_sec": (rows / elapsed) if elapsed > 0 else 0.0,
            **task_metrics,
            **output_metrics,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Multimodal MINT1T benchmark")
    parser.add_argument("--benchmark-results-path", type=Path, required=True)
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"])
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--files-per-partition", type=int, default=1)
    parser.add_argument("--input-blocksize", type=str, default=None)
    parser.add_argument("--output-max-batch-bytes", type=int, default=None)
    parser.add_argument("--materialize-on-read", action="store_true", dest="materialize_on_read")
    parser.add_argument("--no-materialize-on-read", action="store_false", dest="materialize_on_read")
    parser.add_argument("--parquet-row-group-size", type=int, default=None)
    parser.add_argument("--parquet-compression", type=str, default=None)
    parser.add_argument("--materialize-on-write", action="store_true", dest="materialize_on_write")
    parser.add_argument("--no-materialize-on-write", action="store_false", dest="materialize_on_write")
    parser.add_argument("--mode", type=str, default="overwrite", choices=["ignore", "overwrite", "append", "error"])
    parser.set_defaults(materialize_on_write=False, materialize_on_read=False)
    args = parser.parse_args()

    ray_client = RayClient()
    ray_client.start()
    try:
        results = run_benchmark(args)
    except Exception as e:  # noqa: BLE001
        logger.error("Benchmark crashed: {}", e)
        logger.debug(traceback.format_exc())
        results = {
            "params": vars(args),
            "metrics": {"is_success": False},
            "tasks": [],
        }
    finally:
        write_benchmark_results(results, args.benchmark_results_path)
        ray_client.stop()

    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
