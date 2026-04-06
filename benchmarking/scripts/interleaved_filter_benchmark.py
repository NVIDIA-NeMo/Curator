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
from utils import (
    collect_interleaved_parquet_metrics,
    collect_interleaved_wds_metrics,
    setup_executor,
    write_benchmark_results,
)

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.filter import (
    InterleavedBlurFilterStage,
    InterleavedCLIPScoreFilterStage,
    InterleavedImageToTextRatioFilterStage,
    InterleavedQRCodeFilterStage,
)
from nemo_curator.stages.interleaved.io import (
    InterleavedParquetReader,
    InterleavedParquetWriterStage,
    InterleavedWebdatasetReader,
    InterleavedWebdatasetWriterStage,
)
from nemo_curator.tasks.utils import TaskPerfUtils


def create_pipeline(args: argparse.Namespace) -> Pipeline:
    pipeline = Pipeline(
        name="interleaved_filter_benchmark",
        description=("Benchmark: interleaved reader -> blur, QR, CLIP score, image/text ratio filters -> writer"),
    )

    # ── Reader ────────────────────────────────────────────────────────────────
    if args.reader_type == "wds":
        pipeline.add_stage(
            InterleavedWebdatasetReader(
                file_paths=args.input_path,
                files_per_partition=args.files_per_partition,
                blocksize=args.input_blocksize,
                max_batch_bytes=args.output_max_batch_bytes,
                materialize_on_read=args.materialize_on_read,
                per_image_fields=tuple(args.per_image_fields) if args.per_image_fields else (),
                per_text_fields=tuple(args.per_text_fields) if args.per_text_fields else (),
            )
        )
    else:
        pipeline.add_stage(
            InterleavedParquetReader(
                file_paths=args.input_path,
                files_per_partition=args.files_per_partition,
                blocksize=args.input_blocksize,
                max_batch_bytes=args.output_max_batch_bytes,
                fields=tuple(args.reader_fields) if args.reader_fields else None,
            )
        )

    # ── Filters ────────────────────────────────────────────────────────────────
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

    # ── Writer ────────────────────────────────────────────────────────────────
    if args.writer_format == "wds":
        pipeline.add_stage(
            InterleavedWebdatasetWriterStage(
                path=args.output_path,
                materialize_on_write=args.materialize_on_write,
                on_materialize_error=args.on_materialize_error,
                mode=args.mode,
            )
        )
    else:
        write_kwargs: dict[str, Any] = {}
        if args.parquet_row_group_size is not None:
            write_kwargs["row_group_size"] = args.parquet_row_group_size
        if args.parquet_compression is not None:
            write_kwargs["compression"] = args.parquet_compression
        pipeline.add_stage(
            InterleavedParquetWriterStage(
                path=args.output_path,
                materialize_on_write=args.materialize_on_write,
                on_materialize_error=args.on_materialize_error,
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

    input_metrics_start = time.perf_counter()
    collect_fn = (
        collect_interleaved_parquet_metrics if args.reader_type == "parquet" else collect_interleaved_wds_metrics
    )
    input_metrics = {f"input_{k}": v for k, v in collect_fn(args.input_path).items()}
    input_metrics_elapsed = time.perf_counter() - input_metrics_start
    logger.info("collect_input_metrics took {:.3f}s", input_metrics_elapsed)

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
    collect_fn = (
        collect_interleaved_parquet_metrics if args.writer_format == "parquet" else collect_interleaved_wds_metrics
    )
    output_metrics = {f"output_{k}": v for k, v in collect_fn(output_path).items()}
    metrics_elapsed = time.perf_counter() - metrics_start
    logger.info("collect_output_metrics took {:.3f}s", metrics_elapsed)
    task_metrics = TaskPerfUtils.aggregate_task_metrics(output_tasks, prefix="task")
    writer_stats = {k: v for k, v in task_metrics.items() if "interleaved_" in k and "_writer" in k}
    logger.info("Writer stage stats: {}", writer_stats)

    rows = output_metrics["output_num_rows"]
    samples = output_metrics["output_num_samples"]
    return {
        "params": {
            "executor": args.executor,
            "input_path": input_path,
            "output_path": str(output_path),
            "reader_type": args.reader_type,
            "writer_format": args.writer_format,
            "files_per_partition": args.files_per_partition,
            "input_blocksize": args.input_blocksize,
            "output_max_batch_bytes": args.output_max_batch_bytes,
            "materialize_on_read": args.materialize_on_read,
            "materialize_on_write": args.materialize_on_write,
            "on_materialize_error": args.on_materialize_error,
            "reader_fields": list(args.reader_fields),
            "per_image_fields": list(args.per_image_fields) if args.per_image_fields else [],
            "per_text_fields": list(args.per_text_fields) if args.per_text_fields else [],
            "parquet_row_group_size": args.parquet_row_group_size,
            "parquet_compression": args.parquet_compression,
            "mode": args.mode,
            "clip_model_dir": args.clip_model_dir,
            "drop_invalid_rows": args.drop_invalid_rows,
            "blur_score_threshold": args.blur_score_threshold,
            "qrcode_score_threshold": args.qrcode_score_threshold,
            "clip_min_score": args.clip_min_score,
            "image_text_min_ratio": args.image_text_min_ratio,
            "image_text_max_ratio": args.image_text_max_ratio,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": elapsed,
            "throughput_rows_per_sec": (rows / elapsed) if (elapsed > 0 and rows > 0) else 0.0,
            "throughput_samples_per_sec": (samples / elapsed) if (elapsed > 0 and samples > 0) else 0.0,
            **input_metrics,
            **task_metrics,
            **output_metrics,
            "num_rows": rows,
            "num_output_files": output_metrics["output_num_files"],
            "materialize_error_count": output_metrics.get("output_materialize_error_count", 0),
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interleaved filter benchmark (blur, QR, CLIP score, image/text ratio)"
    )
    parser.add_argument("--benchmark-results-path", type=Path, required=True)
    parser.add_argument("--executor", default="ray_data", choices=["xenna", "ray_data"])
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--clip-model-dir", type=str, required=True)
    parser.add_argument("--reader-type", default="wds", choices=["wds", "parquet"])
    parser.add_argument("--writer-format", default="parquet", choices=["parquet", "wds"])
    parser.add_argument("--files-per-partition", type=int, default=1)
    parser.add_argument("--input-blocksize", type=str, default=None)
    parser.add_argument("--output-max-batch-bytes", type=int, default=None)
    parser.add_argument(
        "--reader-fields",
        nargs="*",
        default=[
            "image_metadata",
            "url",
            "language_id_whole_page_fasttext",
            "pdf_name",
            "previous_word_count",
            "bff_contained_ngram_count_before_dedupe",
        ],
    )
    parser.add_argument("--materialize-on-read", action="store_true", dest="materialize_on_read")
    parser.add_argument("--no-materialize-on-read", action="store_false", dest="materialize_on_read")
    parser.add_argument("--materialize-on-write", action="store_true", dest="materialize_on_write")
    parser.add_argument("--no-materialize-on-write", action="store_false", dest="materialize_on_write")
    parser.add_argument(
        "--on-materialize-error",
        default="error",
        choices=["error", "warn", "drop_row", "drop_sample"],
        dest="on_materialize_error",
    )
    parser.add_argument("--parquet-row-group-size", type=int, default=None)
    parser.add_argument("--parquet-compression", type=str, default=None)
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
    parser.set_defaults(materialize_on_write=True, materialize_on_read=True, drop_invalid_rows=True)
    args = parser.parse_args()

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

    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
