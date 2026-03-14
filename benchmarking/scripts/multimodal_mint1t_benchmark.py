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

"""Benchmark for interleaved multimodal IO: reader -> optional filter -> writer."""

import argparse
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
from loguru import logger
from utils import (
    collect_input_metrics_parquet,
    collect_input_metrics_wds,
    collect_lance_output_metrics,
    collect_parquet_output_metrics,
    collect_webdataset_output_metrics,
    setup_executor,
    validate_parquet_ordering,
    validate_wds_ordering,
    write_benchmark_results,
)

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.io import (
    InterleavedLanceFragmentWriterStage,
    InterleavedParquetReader,
    InterleavedParquetWriterStage,
    InterleavedWebdatasetWriterStage,
    WebdatasetReader,
    commit_lance_fragments,
)
from nemo_curator.stages.interleaved.stages import InterleavedAspectRatioFilterStage
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.tasks.utils import TaskPerfUtils


@dataclass
class _SourceRefSchemeFilter(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Drop entire samples where any image row has a source_ref not matching *scheme*."""

    scheme: str = "s3"
    name: str = "source_ref_scheme_filter"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: InterleavedBatch) -> InterleavedBatch | None:
        df = task.to_pandas()
        images = df[df["modality"] == "image"]
        bad_samples: set[str] = set()
        for sid, ref in zip(images["sample_id"], images["source_ref"], strict=True):
            if ref is None or (isinstance(ref, float) and pd.isna(ref)):
                bad_samples.add(str(sid))
                continue
            path = json.loads(ref).get("path") or ""
            if not path.startswith(f"{self.scheme}://"):
                bad_samples.add(str(sid))
        if not bad_samples:
            return task
        filtered = df[~df["sample_id"].isin(bad_samples)].reset_index(drop=True)
        if filtered.empty:
            return None
        return InterleavedBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=pa.Table.from_pandas(filtered, preserve_index=False),
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


def _build_reader(
    args: argparse.Namespace,
) -> InterleavedParquetReader | WebdatasetReader:
    read_kwargs: dict[str, Any] = {}
    if args.reader_type == "wds":
        return WebdatasetReader(
            source_id_field=args.source_id_field,
            file_paths=args.input_path,
            files_per_partition=args.files_per_partition,
            blocksize=args.input_blocksize,
            max_batch_bytes=args.output_max_batch_bytes,
            read_kwargs=read_kwargs,
            materialize_on_read=args.materialize_on_read,
            per_image_fields=tuple(args.per_image_fields) if args.per_image_fields else (),
            per_text_fields=tuple(args.per_text_fields) if args.per_text_fields else (),
        )
    if args.reader_type == "parquet":
        exclude = set(args.reader_exclude_fields) if args.reader_exclude_fields else set()
        all_fields = [
            "sample_id",
            "position",
            "modality",
            "content_type",
            "text_content",
            "binary_content",
            "source_ref",
            "materialize_error",
        ]
        fields = [f for f in all_fields if f not in exclude] if exclude else None
        return InterleavedParquetReader(
            file_paths=args.input_path,
            files_per_partition=args.files_per_partition,
            max_batch_bytes=args.output_max_batch_bytes,
            read_kwargs=read_kwargs,
            fields=fields,
        )
    msg = f"Unknown reader type: {args.reader_type}"
    raise ValueError(msg)


def _build_writer(
    args: argparse.Namespace,
) -> InterleavedParquetWriterStage | InterleavedWebdatasetWriterStage | InterleavedLanceFragmentWriterStage:
    write_kwargs: dict[str, Any] = {}
    if args.parquet_row_group_size is not None:
        write_kwargs["row_group_size"] = args.parquet_row_group_size
    if args.parquet_compression is not None:
        write_kwargs["compression"] = args.parquet_compression

    common = {
        "path": args.output_path,
        "materialize_on_write": args.materialize_on_write,
        "write_kwargs": write_kwargs,
        "mode": args.mode,
        "on_materialize_error": args.on_materialize_error,
    }

    if args.writer_format == "parquet":
        return InterleavedParquetWriterStage(**common)
    if args.writer_format == "webdataset":
        return InterleavedWebdatasetWriterStage(**common)
    if args.writer_format == "lance":
        return InterleavedLanceFragmentWriterStage(**common)
    msg = f"Unknown writer format: {args.writer_format}"
    raise ValueError(msg)


def create_pipeline(args: argparse.Namespace) -> Pipeline:
    pipeline = Pipeline(
        name="multimodal_mint1t_benchmark",
        description=f"Benchmark: {args.reader_type} reader -> {args.writer_format} writer",
    )
    pipeline.add_stage(_build_reader(args))
    if args.source_ref_filter != "all":
        pipeline.add_stage(_SourceRefSchemeFilter(scheme=args.source_ref_filter))
    if args.use_filter:
        pipeline.add_stage(
            InterleavedAspectRatioFilterStage(drop_invalid_rows=True, min_aspect_ratio=1.0, max_aspect_ratio=2.0)
        )
    pipeline.add_stage(_build_writer(args))
    return pipeline


def _collect_output_metrics(output_path: Path, writer_format: str) -> dict[str, Any]:
    if writer_format == "parquet":
        return collect_parquet_output_metrics(output_path)
    if writer_format == "webdataset":
        return collect_webdataset_output_metrics(output_path)
    if writer_format == "lance":
        return collect_lance_output_metrics(output_path)
    return {}


def _collect_input_metrics(args: argparse.Namespace) -> dict[str, Any]:
    """Collect input dataset statistics before the pipeline runs."""
    try:
        if args.reader_type == "wds":
            return collect_input_metrics_wds(args.input_path)
        return collect_input_metrics_parquet(args.input_path)
    except Exception as exc:
        logger.warning("Input metrics collection failed: {}", exc)
        return {}


def _validate_ordering(success: bool, args: argparse.Namespace, output_path: Path) -> bool:
    """Validate row/sample ordering in the output. Returns True if valid."""
    if not success:
        return False
    if args.writer_format == "parquet":
        parquet_files = sorted(output_path.glob("*.parquet"))
        if not parquet_files:
            return False
        result = validate_parquet_ordering(parquet_files[0])
        if result["valid"]:
            logger.info("Ordering validation passed on {}", parquet_files[0].name)
        else:
            logger.error("Ordering validation failed on {}: {}", parquet_files[0].name, result["errors"])
        return result["valid"]
    if args.writer_format == "webdataset":
        tar_files = sorted(output_path.glob("*.tar"))
        if not tar_files:
            return False
        result = validate_wds_ordering(tar_files[0])
        if result["valid"]:
            logger.info("WDS ordering validation passed on {}", tar_files[0].name)
        else:
            logger.error("WDS ordering validation failed on {}: {}", tar_files[0].name, result["errors"])
        return result["valid"]
    return False


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    executor = setup_executor(args.executor)
    input_path = str(Path(args.input_path).absolute())
    output_path = Path(args.output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Collecting input metrics...")
    input_metrics = _collect_input_metrics(args)
    logger.info("Input metrics: {}", input_metrics)

    start = time.perf_counter()
    output_tasks = []
    success = False
    try:
        pipeline = create_pipeline(args)
        logger.info("Pipeline:\n{}", pipeline.describe())
        output_tasks = pipeline.run(executor)

        if args.writer_format == "lance":
            commit_lance_fragments(str(output_path), output_tasks)

        success = True
    except Exception as e:
        logger.error("Benchmark failed: {}", e)
        logger.debug(traceback.format_exc())

    elapsed = time.perf_counter() - start
    metrics_start = time.perf_counter()
    output_metrics = _collect_output_metrics(output_path, args.writer_format)
    metrics_elapsed = time.perf_counter() - metrics_start
    logger.info("Output metrics collection took {:.3f}s", metrics_elapsed)
    task_metrics = TaskPerfUtils.aggregate_task_metrics(output_tasks, prefix="task")

    ordering_valid = _validate_ordering(success, args, output_path)

    # Row-count delta: only meaningful when input and output count in the same units.
    # WDS counts samples; Parquet counts rows — cross-format deltas are not comparable.
    output_rows = output_metrics.get("num_rows", output_metrics.get("num_samples", 0))
    input_rows = input_metrics.get("input_num_rows", input_metrics.get("input_num_samples"))
    same_format = args.reader_type == "parquet" and args.writer_format == "parquet"
    row_delta = (output_rows - input_rows) if (input_rows is not None and same_format) else None
    if row_delta is not None and row_delta != 0 and not args.materialize_on_write and not args.use_filter:
        logger.warning(
            "Row count delta: input={} output={} delta={} — unexpected for no-filter no-mat run",
            input_rows,
            output_rows,
            row_delta,
        )

    return {
        "params": {
            "executor": args.executor,
            "reader_type": args.reader_type,
            "writer_format": args.writer_format,
            "source_ref_filter": args.source_ref_filter,
            "use_filter": args.use_filter,
            "input_path": input_path,
            "output_path": str(output_path),
            "files_per_partition": args.files_per_partition,
            "input_blocksize": args.input_blocksize,
            "output_max_batch_bytes": args.output_max_batch_bytes,
            "materialize_on_read": args.materialize_on_read,
            "materialize_on_write": args.materialize_on_write,
            "on_materialize_error": args.on_materialize_error,
            "per_image_fields": list(args.per_image_fields) if args.per_image_fields else [],
            "per_text_fields": list(args.per_text_fields) if args.per_text_fields else [],
            "reader_exclude_fields": list(args.reader_exclude_fields) if args.reader_exclude_fields else [],
            "parquet_row_group_size": args.parquet_row_group_size,
            "parquet_compression": args.parquet_compression,
            "mode": args.mode,
        },
        "metrics": {
            "is_success": success,
            "ordering_valid": ordering_valid,
            "time_taken_s": elapsed,
            "throughput_rows_per_sec": (output_rows / elapsed) if elapsed > 0 else 0.0,
            "row_delta": row_delta,
            **task_metrics,
            **input_metrics,
            **output_metrics,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Interleaved multimodal IO benchmark")
    parser.add_argument("--benchmark-results-path", type=Path, required=True)
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"])
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--reader-type", default="wds", choices=["wds", "parquet"])
    parser.add_argument("--writer-format", default="parquet", choices=["parquet", "webdataset", "lance"])
    parser.add_argument("--source-id-field", type=str, default="pdf_name")
    parser.add_argument(
        "--source-ref-filter",
        default="all",
        choices=["all", "s3"],
        help="Drop samples with non-matching source_ref schemes before writing",
    )
    parser.add_argument("--use-filter", action="store_true", dest="use_filter")
    parser.add_argument("--no-filter", action="store_false", dest="use_filter")
    parser.add_argument("--files-per-partition", type=int, default=1)
    parser.add_argument("--input-blocksize", type=str, default=None)
    parser.add_argument("--output-max-batch-bytes", type=int, default=None)
    parser.add_argument("--materialize-on-read", action="store_true", dest="materialize_on_read")
    parser.add_argument("--no-materialize-on-read", action="store_false", dest="materialize_on_read")
    parser.add_argument("--parquet-row-group-size", type=int, default=None)
    parser.add_argument("--parquet-compression", type=str, default=None)
    parser.add_argument("--materialize-on-write", action="store_true", dest="materialize_on_write")
    parser.add_argument("--no-materialize-on-write", action="store_false", dest="materialize_on_write")
    parser.add_argument(
        "--on-materialize-error", type=str, default="error", choices=["error", "warn", "drop_row", "drop_sample"]
    )
    parser.add_argument("--mode", type=str, default="overwrite", choices=["ignore", "overwrite", "append", "error"])
    parser.add_argument("--per-image-fields", nargs="*", default=["image_metadata"])
    parser.add_argument("--per-text-fields", nargs="*", default=[])
    parser.add_argument(
        "--reader-exclude-fields",
        nargs="*",
        default=[],
        metavar="FIELD",
        help="Parquet columns to skip when reading (e.g. binary_content). No-op for WDS reader.",
    )
    parser.set_defaults(materialize_on_write=False, materialize_on_read=False, use_filter=True)
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
