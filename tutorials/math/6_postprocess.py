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

import argparse
import glob
import os
from datetime import datetime

import pandas as pd
from loguru import logger

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.math.modifiers.merge_chunks import ChunkMergeStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter


def build_pipeline(  # noqa: PLR0913
    input_files: list[str],
    output_dir: str,
    text_field: str = "cleaned_text",
    raw_text_field: str | None = "text",
    chunk_id_field: str = "chunk_id",
    groupby_columns: list[str] | None = None,
    max_text_length: int = 900_000,
) -> Pipeline:
    """Build the post-processing (chunk merge) pipeline."""
    p = Pipeline(
        name="math_postprocess",
        description="Merge chunked LLM output back into one row per document",
    )

    p.add_stage(
        JsonlReader(file_paths=input_files).with_(
            {
                "file_partitioning": {"resources": Resources(cpus=0.5)},
                "jsonl_reader": {"resources": Resources(cpus=0.5)},
            }
        )
    )

    p.add_stage(
        ChunkMergeStage(
            text_field=text_field,
            raw_text_field=raw_text_field,
            chunk_id_field=chunk_id_field,
            groupby_columns=groupby_columns,
            max_text_length=max_text_length,
        ).with_(resources=Resources(cpus=1.0))
    )

    p.add_stage(JsonlWriter(path=output_dir).with_(resources=Resources(cpus=1.0)))

    return p


def _safe_read_row_count(fp: str) -> int:
    """Read row count from a single JSONL file, returning 0 on failure."""
    try:
        return len(pd.read_json(fp, lines=True))
    except (ValueError, FileNotFoundError):
        logger.warning(f"Could not read {fp}, skipping")
        return 0


def count_rows(file_paths: list[str]) -> int:
    """Count total rows across JSONL files."""
    return sum(_safe_read_row_count(fp) for fp in file_paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process LLM cleanup output: merge chunks back into one row per document",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input directory or glob pattern for JSONL files")
    parser.add_argument("--output", required=True, help="Output directory for merged JSONL files")
    parser.add_argument("--text_field", default="cleaned_text", help="Column containing LLM-cleaned text")
    parser.add_argument("--raw_text_field", default="text", help="Column containing raw text (set to '' to skip)")
    parser.add_argument("--chunk_id_field", default="chunk_id", help="Column containing chunk IDs")
    parser.add_argument(
        "--groupby",
        nargs="+",
        default=["url"],
        help="Columns to group by for merging (e.g., url, warc_filename)",
    )
    parser.add_argument("--max_text_length", type=int, default=900_000, help="Maximum merged text length in chars")

    args = parser.parse_args()

    if os.path.isdir(args.input):
        input_files = glob.glob(os.path.join(args.input, "**/*.jsonl"), recursive=True)
        input_files.extend(glob.glob(os.path.join(args.input, "**/*.json"), recursive=True))
    else:
        input_files = glob.glob(args.input)

    if not input_files:
        logger.error(f"No input files found matching: {args.input}")
        return

    logger.info(f"Found {len(input_files)} input files")

    rows_before = count_rows(input_files)
    logger.info(f"Total input rows (chunks): {rows_before}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    output_dir = os.path.join(args.output, f"merged_{timestamp}")

    raw_text_field = args.raw_text_field or None

    ray_client = RayClient()
    ray_client.start()

    try:
        pipeline = build_pipeline(
            input_files=input_files,
            output_dir=output_dir,
            text_field=args.text_field,
            raw_text_field=raw_text_field,
            chunk_id_field=args.chunk_id_field,
            groupby_columns=args.groupby,
            max_text_length=args.max_text_length,
        )

        logger.info(pipeline.describe())

        pipeline.run()

        # Count output rows
        output_files = glob.glob(os.path.join(output_dir, "**/*.jsonl"), recursive=True)
        rows_after = count_rows(output_files)

        logger.info("Pipeline completed successfully.")
        logger.info(f"Output written to: {output_dir}")
        logger.info(f"Rows before (chunks): {rows_before} -> Rows after (documents): {rows_after}")

    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
