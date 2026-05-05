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

"""Interleaved multimodal pipeline.

Reads WebDataset tar shards (e.g. MINT-1T), applies optional image and text
filtering, and writes to Parquet or WebDataset tar format.

Example — filter one shard and write to Parquet::

    python interleaved_pipeline.py \\
        --input-path /data/CC-MAIN-20240412101354-20240412131354-00000.tar \\
        --output-path /data/output/ \\
        --on-materialize-error drop_row \\
        --mode overwrite

Example — also drop short text rows and very narrow/wide images::

    python interleaved_pipeline.py \\
        --input-path /data/shard-0/ \\
        --output-path /data/output/ \\
        --min-aspect-ratio 0.5 \\
        --max-aspect-ratio 2.0 \\
        --min-text-chars 50 \\
        --mode overwrite

Example — write back to WebDataset tar format::

    python interleaved_pipeline.py \\
        --input-path /data/CC-MAIN-20240412-shard-0/ \\
        --output-path /data/output_tars/ \\
        --writer-format webdataset \\
        --mode overwrite

Example — add typed schema for custom passthrough columns::

    python interleaved_pipeline.py \\
        --input-path /data/shard-0/ \\
        --output-path /data/output/ \\
        --schema-overrides '{"url": "large_string", "pdf_name": "large_string"}' \\
        --mode overwrite
"""

import argparse
import json
from dataclasses import dataclass

import pandas as pd
import pyarrow as pa

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.io import (
    InterleavedParquetWriterStage,
    InterleavedWebdatasetReader,
    InterleavedWebdatasetWriterStage,
)
from nemo_curator.stages.interleaved.stages import (
    BaseInterleavedFilterStage,
    InterleavedAspectRatioFilterStage,
)
from nemo_curator.tasks.interleaved import InterleavedBatch

# Mapping from simple string names to PyArrow types for --schema-overrides
_PA_TYPE_MAP: dict[str, pa.DataType] = {
    "string": pa.string(),
    "large_string": pa.large_string(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "float32": pa.float32(),
    "float64": pa.float64(),
    "bool": pa.bool_(),
    "binary": pa.binary(),
    "large_binary": pa.large_binary(),
}


@dataclass
class ShortTextFilterStage(BaseInterleavedFilterStage):
    """Drop text rows shorter than min_chars characters."""

    min_chars: int = 50
    name: str = "short_text_filter"

    def content_keep_mask(self, _task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
        too_short = (df["modality"] == "text") & (df["text_content"].fillna("").str.len() < self.min_chars)
        return ~too_short


def _parse_schema_overrides(raw: str) -> dict[str, pa.DataType]:
    """argparse type= callback: parse JSON string into a {column: pa.DataType} dict."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        msg = f"must be valid JSON: {e}"
        raise argparse.ArgumentTypeError(msg) from e
    if not isinstance(parsed, dict):
        msg = "must be a JSON object mapping column names to type strings"
        raise argparse.ArgumentTypeError(msg)
    overrides: dict[str, pa.DataType] = {}
    for col, type_str in parsed.items():
        if type_str not in _PA_TYPE_MAP:
            msg = f"unknown type '{type_str}' for column '{col}'. Supported: {sorted(_PA_TYPE_MAP)}"
            raise argparse.ArgumentTypeError(msg)
        overrides[col] = _PA_TYPE_MAP[type_str]
    return overrides


def _parse_json(raw: str) -> dict:
    """argparse type= callback: parse a JSON string into a dict."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        msg = f"must be valid JSON: {e}"
        raise argparse.ArgumentTypeError(msg) from e


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    shared_kwargs = {"storage_options": args.storage_options_json} if args.storage_options_json else {}

    pipe = Pipeline(
        name="interleaved_pipeline",
        description="WebDataset -> interleaved rows -> filtered -> output",
    )

    pipe.add_stage(
        InterleavedWebdatasetReader(
            file_paths=args.input_path,
            files_per_partition=args.files_per_partition,
            blocksize=args.input_blocksize,
            max_batch_bytes=args.output_max_batch_bytes,
            read_kwargs=shared_kwargs,
            materialize_on_read=args.materialize_on_read,
            fields=tuple(args.fields) if args.fields else None,
            per_image_fields=tuple(args.per_image_fields) if args.per_image_fields else (),
            per_text_fields=tuple(args.per_text_fields) if args.per_text_fields else (),
        )
    )

    pipe.add_stage(
        InterleavedAspectRatioFilterStage(
            min_aspect_ratio=args.min_aspect_ratio,
            max_aspect_ratio=args.max_aspect_ratio,
            drop_invalid_rows=True,
        )
    )

    if args.min_text_chars > 0:
        pipe.add_stage(ShortTextFilterStage(min_chars=args.min_text_chars))

    if args.writer_format == "parquet":
        pipe.add_stage(
            InterleavedParquetWriterStage(
                path=args.output_path,
                materialize_on_write=args.materialize_on_write,
                write_kwargs=shared_kwargs,
                mode=args.mode,
                on_materialize_error=args.on_materialize_error,
                schema_overrides=args.schema_overrides,
            )
        )
    else:  # webdataset
        pipe.add_stage(
            InterleavedWebdatasetWriterStage(
                path=args.output_path,
                materialize_on_write=args.materialize_on_write,
                write_kwargs=shared_kwargs,
                mode=args.mode,
                on_materialize_error=args.on_materialize_error,
            )
        )

    return pipe


def main(args: argparse.Namespace) -> None:
    ray_client = RayClient()
    ray_client.start()
    pipeline = build_pipeline(args)
    print(pipeline.describe())
    pipeline.run(executor=RayDataExecutor())
    ray_client.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interleaved multimodal pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument("--input-path", type=str, required=True, help="Input tar shard path or directory of shards")
    parser.add_argument(
        "--output-path", type=str, required=True, help="Output directory (Parquet files or tar shards)"
    )
    parser.add_argument(
        "--writer-format",
        type=str,
        default="parquet",
        choices=["parquet", "webdataset"],
        help="Output format: 'parquet' (default) or 'webdataset' (tar)",
    )

    # Batching / partitioning
    parser.add_argument(
        "--files-per-partition", type=int, default=1, help="Number of tar shards per pipeline partition"
    )
    parser.add_argument(
        "--input-blocksize", type=str, default=None, help="Block size for file partitioning (e.g. '512MB')"
    )
    parser.add_argument(
        "--output-max-batch-bytes", type=int, default=None, help="Maximum bytes per output batch (splits large shards)"
    )

    # Materialization
    parser.add_argument(
        "--materialize-on-read",
        action="store_true",
        dest="materialize_on_read",
        help="Eagerly load image bytes at read time (default)",
    )
    parser.add_argument(
        "--no-materialize-on-read",
        action="store_false",
        dest="materialize_on_read",
        help="Keep image bytes lazy until write time",
    )
    parser.add_argument(
        "--materialize-on-write",
        action="store_true",
        dest="materialize_on_write",
        help="Fetch lazy binary_content at write time (default)",
    )
    parser.add_argument(
        "--no-materialize-on-write",
        action="store_false",
        dest="materialize_on_write",
        help="Skip materialization at write time (writes null binary_content if lazy)",
    )
    parser.set_defaults(materialize_on_write=True, materialize_on_read=True)

    # Error handling
    parser.add_argument(
        "--on-materialize-error",
        type=str,
        default="error",
        choices=["error", "warn", "drop_row", "drop_sample"],
        dest="on_materialize_error",
        help=(
            "What to do when a binary_content fetch fails. "
            "'error' raises immediately; "
            "'warn' keeps the row with null binary_content; "
            "'drop_row' drops only the failed row; "
            "'drop_sample' drops the entire sample."
        ),
    )

    # Output mode
    parser.add_argument(
        "--mode",
        type=str,
        default="ignore",
        choices=["ignore", "overwrite", "append", "error"],
        help="Output directory handling mode",
    )

    # Image filtering
    parser.add_argument(
        "--min-aspect-ratio", type=float, default=0.5, help="Minimum image aspect ratio (width/height) to keep"
    )
    parser.add_argument(
        "--max-aspect-ratio", type=float, default=2.0, help="Maximum image aspect ratio (width/height) to keep"
    )

    # Text filtering
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=0,
        dest="min_text_chars",
        help="Drop text rows shorter than this many characters (0 = disabled)",
    )

    # Field selection
    parser.add_argument(
        "--fields", nargs="*", default=None, help="Passthrough field names to read from JSON (default: all fields)"
    )
    parser.add_argument(
        "--per-image-fields",
        nargs="*",
        default=["image_metadata"],
        help="Fields that are per-image lists (one entry per image position)",
    )
    parser.add_argument(
        "--per-text-fields", nargs="*", default=[], help="Fields that are per-text lists (one entry per text position)"
    )

    # Schema
    parser.add_argument(
        "--schema-overrides",
        type=_parse_schema_overrides,
        default=None,
        help=(
            "JSON object mapping column names to PyArrow type names, "
            'e.g. \'{"url": "large_string", "pdf_name": "large_string"}\'. '
            f"Supported types: {sorted(_PA_TYPE_MAP)}"
        ),
    )

    # Storage
    parser.add_argument(
        "--storage-options-json",
        type=_parse_json,
        default=None,
        help="JSON-encoded fsspec storage options for cloud paths (e.g. S3 credentials)",
    )

    main(parser.parse_args())
