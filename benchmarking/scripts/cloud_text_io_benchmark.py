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

"""Cloud text I/O benchmark for JSONL and Parquet datasets.

This benchmark copies a text dataset from an input URI to an output URI using
Curator reader/writer stages. It is intended for cloud object store paths (for
example S3/GCS/Azure/fsspec URLs), but local paths are also useful for smoke
runs and baselines.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter

Compression = Literal["none", "gzip", "snappy", "zstd"]
Format = Literal["jsonl", "parquet"]


def _json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        msg = "JSON arguments must decode to an object"
        raise TypeError(msg)
    return parsed


def _compression_kwargs(file_format: Format, compression: Compression) -> dict[str, Any]:
    if compression == "none":
        return {"compression": None} if file_format == "parquet" else {}
    return {"compression": compression}


def _build_pipeline(  # noqa: PLR0913
    input_path: str,
    output_path: str,
    file_format: Format,
    compression: Compression,
    fields: list[str] | None,
    files_per_partition: int | None,
    blocksize: str | None,
    read_kwargs: dict[str, Any],
    write_kwargs: dict[str, Any],
) -> Pipeline:
    pipeline = Pipeline(name="cloud_text_io", description="Cloud text I/O benchmark")

    if file_format == "jsonl":
        reader = JsonlReader(
            file_paths=input_path,
            files_per_partition=files_per_partition,
            blocksize=blocksize,
            fields=fields,
            read_kwargs=read_kwargs,
        )
        writer = JsonlWriter(
            path=output_path,
            fields=fields,
            write_kwargs={**_compression_kwargs(file_format, compression), **write_kwargs},
        )
    elif file_format == "parquet":
        reader = ParquetReader(
            file_paths=input_path,
            files_per_partition=files_per_partition,
            blocksize=blocksize,
            fields=fields,
            read_kwargs=read_kwargs,
        )
        writer = ParquetWriter(
            path=output_path,
            fields=fields,
            write_kwargs={**_compression_kwargs(file_format, compression), **write_kwargs},
        )
    else:
        msg = f"Unsupported format: {file_format}"
        raise ValueError(msg)

    pipeline.add_stage(reader)
    pipeline.add_stage(writer)
    return pipeline


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    fields = [field.strip() for field in args.fields.split(",") if field.strip()] if args.fields else None
    read_kwargs = _json_arg(args.read_kwargs_json)
    write_kwargs = _json_arg(args.write_kwargs_json)
    executor_config = _json_arg(args.executor_config_json)

    pipeline = _build_pipeline(
        input_path=args.input_path,
        output_path=args.output_path,
        file_format=args.format,
        compression=args.compression,
        fields=fields,
        files_per_partition=args.files_per_partition,
        blocksize=args.blocksize,
        read_kwargs=read_kwargs,
        write_kwargs=write_kwargs,
    )
    executor = setup_executor(args.executor, config=executor_config or None)

    logger.info(f"Starting cloud text I/O benchmark: {args.input_path} -> {args.output_path}")
    start = time.perf_counter()
    try:
        tasks = pipeline.run(executor, initial_tasks=None)
        success = True
    except Exception as exc:  # pragma: no cover - benchmark failure path
        logger.exception(f"Cloud text I/O benchmark failed: {exc}")
        tasks = []
        success = False
    elapsed = time.perf_counter() - start

    total_documents = sum(task.num_items for task in tasks) if tasks else 0
    output_path = Path(args.output_path)
    output_bytes = (
        sum(path.stat().st_size for path in output_path.rglob("*") if path.is_file()) if output_path.exists() else None
    )

    return {
        "params": {
            "input_path": args.input_path,
            "output_path": args.output_path,
            "format": args.format,
            "compression": args.compression,
            "fields": fields,
            "files_per_partition": args.files_per_partition,
            "blocksize": args.blocksize,
            "executor": args.executor,
            "device_label": args.device_label,
            "read_kwargs": read_kwargs,
            "write_kwargs": write_kwargs,
            "executor_config": executor_config,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": elapsed,
            "num_output_tasks": len(tasks) if tasks else 0,
            "total_documents": total_documents,
            "throughput_docs_per_sec": total_documents / elapsed if elapsed > 0 else 0.0,
            "output_bytes": output_bytes,
            "output_mib_per_sec": (
                (output_bytes / (1024**2) / elapsed) if output_bytes is not None and elapsed > 0 else None
            ),
        },
        "tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark text cloud I/O for JSONL or Parquet datasets")
    parser.add_argument("--benchmark-results-path", required=True, help="Directory to write benchmark results")
    parser.add_argument("--input-path", required=True, help="Input dataset path or cloud URI")
    parser.add_argument("--output-path", required=True, help="Output dataset path or cloud URI")
    parser.add_argument("--format", required=True, choices=["jsonl", "parquet"])
    parser.add_argument("--compression", default="none", choices=["none", "gzip", "snappy", "zstd"])
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data", "ray_actors"])
    parser.add_argument(
        "--device-label",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Label used to distinguish CPU/GPU benchmark runs",
    )
    parser.add_argument("--files-per-partition", type=int, default=None)
    parser.add_argument("--blocksize", default=None)
    parser.add_argument("--fields", default=None, help="Comma-separated subset of columns to read/write")
    parser.add_argument(
        "--read-kwargs-json", default=None, help="JSON object passed to the reader, e.g. storage_options"
    )
    parser.add_argument("--write-kwargs-json", default=None, help="JSON object passed to the writer")
    parser.add_argument("--executor-config-json", default=None, help="JSON object passed to the executor constructor")

    args = parser.parse_args()

    results = {
        "params": vars(args),
        "metrics": {"is_success": False},
        "tasks": [],
    }
    try:
        results = run_benchmark(args)
    finally:
        write_benchmark_results(results, args.benchmark_results_path)

    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
