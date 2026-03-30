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

"""Benchmark for interleaved row-wise filters covered by tests/stages/interleaved/filter/."""

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger
from utils import collect_parquet_output_metrics, setup_executor, validate_parquet_ordering, write_benchmark_results

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.filter import (
    InterleavedBlurFilterStage,
    InterleavedCLIPScoreFilterStage,
    InterleavedImageToTextRatioFilterStage,
    InterleavedQRCodeFilterStage,
)
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage, WebdatasetReader
from nemo_curator.tasks.utils import TaskPerfUtils


def create_pipeline(args: argparse.Namespace) -> Pipeline:
    read_kwargs = {}
    write_kwargs = {}
    if args.parquet_row_group_size is not None:
        write_kwargs["row_group_size"] = args.parquet_row_group_size
    if args.parquet_compression is not None:
        write_kwargs["compression"] = args.parquet_compression
    pipeline = Pipeline(
        name="interleaved_filter_benchmark",
        description=(
            "Benchmark: WebDataset through blur, QR code, CLIP score, and image/text ratio filters to parquet"
        ),
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
            per_image_fields=tuple(args.per_image_fields) if args.per_image_fields else (),
            per_text_fields=tuple(args.per_text_fields) if args.per_text_fields else (),
        )
    )
    max_ratio = float("inf") if args.image_text_max_ratio is None else args.image_text_max_ratio
    pipeline.add_stage(
        InterleavedBlurFilterStage(
            drop_invalid_rows=args.drop_invalid_rows,
            score_threshold=args.blur_score_threshold,
        )
    )
    pipeline.add_stage(
        InterleavedQRCodeFilterStage(
            drop_invalid_rows=args.drop_invalid_rows,
            score_threshold=args.qrcode_score_threshold,
        )
    )
    pipeline.add_stage(
        InterleavedCLIPScoreFilterStage(
            drop_invalid_rows=args.drop_invalid_rows,
            model_dir=args.clip_model_dir,
            min_score=args.clip_min_score,
        )
    )
    pipeline.add_stage(
        InterleavedImageToTextRatioFilterStage(
            drop_invalid_rows=args.drop_invalid_rows,
            min_ratio=args.image_text_min_ratio,
            max_ratio=max_ratio,
        )
    )
    pipeline.add_stage(
        InterleavedParquetWriterStage(
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
    except Exception as e:
        logger.error("Benchmark failed: {}", e)
        logger.debug(traceback.format_exc())

    elapsed = time.perf_counter() - start
    metrics_start = time.perf_counter()
    output_metrics = collect_parquet_output_metrics(output_path)
    metrics_elapsed = time.perf_counter() - metrics_start
    logger.info("collect_parquet_output_metrics took {:.3f}s", metrics_elapsed)
    task_metrics = TaskPerfUtils.aggregate_task_metrics(output_tasks, prefix="task")
    writer_stats = {k: v for k, v in task_metrics.items() if "interleaved_parquet_writer" in k}
    logger.info("Writer stage stats: {}", writer_stats)

    ordering_valid = False
    if success:
        parquet_files = sorted(output_path.glob("*.parquet"))
        if parquet_files:
            result = validate_parquet_ordering(parquet_files[0])
            ordering_valid = result["valid"]
            if not ordering_valid:
                logger.error("Ordering validation failed on {}: {}", parquet_files[0].name, result["errors"])
            else:
                logger.info("Ordering validation passed on {}", parquet_files[0].name)

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
            "per_image_fields": list(args.per_image_fields) if args.per_image_fields else [],
            "per_text_fields": list(args.per_text_fields) if args.per_text_fields else [],
            "parquet_row_group_size": args.parquet_row_group_size,
            "parquet_compression": args.parquet_compression,
            "mode": args.mode,
            "drop_invalid_rows": args.drop_invalid_rows,
            "blur_score_threshold": args.blur_score_threshold,
            "qrcode_score_threshold": args.qrcode_score_threshold,
            "clip_model_dir": args.clip_model_dir,
            "clip_min_score": args.clip_min_score,
            "image_text_min_ratio": args.image_text_min_ratio,
            "image_text_max_ratio": args.image_text_max_ratio,
        },
        "metrics": {
            "is_success": success,
            "ordering_valid": ordering_valid,
            "time_taken_s": elapsed,
            "throughput_rows_per_sec": (rows / elapsed) if elapsed > 0 else 0.0,
            **task_metrics,
            **output_metrics,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interleaved filter benchmark (blur, QR, CLIP score, image/text ratio)"
    )
    parser.add_argument("--benchmark-results-path", type=Path, required=True)
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"])
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--clip-model-dir", type=str, required=True)
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
    parser.add_argument("--per-image-fields", nargs="*", default=["image_metadata"])
    parser.add_argument("--per-text-fields", nargs="*", default=[])
    parser.add_argument("--blur-score-threshold", type=float, default=100.0)
    parser.add_argument("--qrcode-score-threshold", type=float, default=0.05)
    parser.add_argument("--clip-min-score", type=float, default=0.15)
    parser.add_argument("--image-text-min-ratio", type=float, default=0.0)
    parser.add_argument(
        "--image-text-max-ratio",
        type=float,
        default=None,
        help="Upper bound on image/text ratio; omit for no upper limit (infinity).",
    )
    parser.add_argument("--drop-invalid-rows", action="store_true", dest="drop_invalid_rows")
    parser.add_argument("--no-drop-invalid-rows", action="store_false", dest="drop_invalid_rows")
    parser.set_defaults(materialize_on_write=False, materialize_on_read=False, drop_invalid_rows=True)
    args = parser.parse_args()

    ray_client = RayClient()
    ray_client.start()
    try:
        results = run_benchmark(args)
    except Exception as e:
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
