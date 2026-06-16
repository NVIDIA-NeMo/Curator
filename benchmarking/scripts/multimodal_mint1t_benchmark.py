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

"""Benchmark for multimodal MINT1T workflow: WebDataset/Parquet -> filter -> Parquet/WebDataset."""

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
    validate_parquet_ordering,
    validate_wds_ordering,
    write_benchmark_results,
)

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.io import (
    InterleavedParquetReader,
    InterleavedParquetWriterStage,
    InterleavedWebdatasetReader,
    InterleavedWebdatasetWriterStage,
)
from nemo_curator.stages.interleaved.stages import InterleavedAspectRatioFilterStage
from nemo_curator.tasks.utils import TaskPerfUtils


def create_pipeline(args: argparse.Namespace) -> Pipeline:
    pipeline = Pipeline(
        name="multimodal_mint1t_benchmark",
        description="Benchmark: multimodal interleaved IO pipeline",
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
    else:  # parquet
        pipeline.add_stage(
            InterleavedParquetReader(
                file_paths=args.input_path,
                files_per_partition=args.files_per_partition,
                blocksize=args.input_blocksize,
                max_batch_bytes=args.output_max_batch_bytes,
                fields=tuple(args.reader_fields) if args.reader_fields else None,
            )
        )

    # ── Optional filter ───────────────────────────────────────────────────────
    if not args.no_filter:
        pipeline.add_stage(
            InterleavedAspectRatioFilterStage(drop_invalid_rows=True, min_aspect_ratio=1.0, max_aspect_ratio=2.0)
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


def _validate_output(writer_format: str, output_path: Path) -> tuple[bool | None, bool | None]:
    """Spot-check one output file for the given format.

    Returns (ordering_valid, wds_valid). None = not applicable or no files found.
    """
    if writer_format == "wds":
        tar = next(output_path.glob("*.tar"), None)
        if tar is None:
            logger.warning("WDS validation skipped: no output tars found")
            return None, None
        result = validate_wds_ordering(tar)
        ordering_valid = result["ordering_valid"]
        wds_valid = result["valid"]
        if not wds_valid:
            logger.error("WDS output validation failed: {}", result["errors"])
        else:
            logger.info(
                "WDS validation passed on {}: {} samples, {} images, ordering_valid={}",
                tar.name,
                result["num_samples"],
                result["num_images"],
                ordering_valid,
            )
        return ordering_valid, wds_valid
    else:
        pq_file = next(output_path.glob("*.parquet"), None)
        if pq_file is None:
            logger.warning("Parquet ordering validation skipped: no output files found")
            return None, None
        result = validate_parquet_ordering(pq_file)
        ordering_valid = result["valid"]
        if not ordering_valid:
            logger.error("Parquet ordering validation failed on {}: {}", pq_file.name, result["errors"])
        else:
            logger.info("Parquet ordering validation passed on {}", pq_file.name)
        return ordering_valid, None  # wds_valid=None: not applicable for parquet


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
    ordering_valid, wds_valid = None, None
    if success:
        ordering_valid, wds_valid = _validate_output(args.writer_format, output_path)
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
            "no_filter": args.no_filter,
            "per_image_fields": list(args.per_image_fields) if args.per_image_fields else [],
            "per_text_fields": list(args.per_text_fields) if args.per_text_fields else [],
            "parquet_row_group_size": args.parquet_row_group_size,
            "parquet_compression": args.parquet_compression,
            "mode": args.mode,
        },
        "metrics": {
            "is_success": success,
            "ordering_valid": ordering_valid,
            "wds_valid": wds_valid,
            "time_taken_s": elapsed,
            "throughput_rows_per_sec": (rows / elapsed) if (elapsed > 0 and rows > 0) else 0.0,
            "throughput_samples_per_sec": (samples / elapsed) if (elapsed > 0 and samples > 0) else 0.0,
            **input_metrics,
            **task_metrics,
            **output_metrics,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Multimodal interleaved IO benchmark")
    parser.add_argument("--benchmark-results-path", type=Path, required=True)
    parser.add_argument("--executor", default="ray_data", choices=["xenna", "ray_data"])
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
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
    parser.add_argument("--no-filter", action="store_true", default=False)
    parser.add_argument("--parquet-row-group-size", type=int, default=None)
    parser.add_argument("--parquet-compression", type=str, default=None)
    parser.add_argument("--mode", type=str, default="overwrite", choices=["ignore", "overwrite", "append", "error"])
    parser.add_argument("--per-image-fields", nargs="*", default=["image_metadata"])
    parser.add_argument("--per-text-fields", nargs="*", default=[])
    parser.set_defaults(materialize_on_write=True, materialize_on_read=True)
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
