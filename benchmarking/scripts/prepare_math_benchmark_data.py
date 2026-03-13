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

"""Offline data preparation for math pipeline benchmarks.

Downloads the FINEMATH_4PLUS dataset from HuggingFace, then fetches raw
binary content from Common Crawl using WARC metadata.  The resulting
"enriched" parquet files contain a ``binary_content`` column that the
nightly math benchmark can consume without network I/O.

This script is NOT part of the nightly benchmark YAML -- it is run once
(or whenever the dataset needs refreshing).

Example usage:

    python prepare_math_benchmark_data.py \\
        --output-path /datasets/finemath4plus_enriched \\
        --max-files 5 --workers 8
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.common_crawl.download import CommonCrawlWARCReader
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MATH_TUTORIAL_DIR = REPO_ROOT / "tutorials" / "math"


def _import_math_download() -> types.ModuleType:
    """Import the math tutorial download module (filename starts with a digit)."""
    spec = importlib.util.spec_from_file_location("math_download", _MATH_TUTORIAL_DIR / "0_download.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["math_download"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_enrichment_pipeline(input_path: str, output_path: str) -> Pipeline:
    """Build a pipeline that reads parquet, fetches CC binary content, and writes parquet."""
    pipeline = Pipeline(
        name="math_benchmark_data_prep",
        description="Fetch binary_content from Common Crawl for FINEMATH_4PLUS benchmark data",
    )

    pipeline.add_stage(
        ParquetReader(file_paths=input_path).with_(
            {
                "file_partitioning": {"resources": Resources(cpus=1.0)},
                "parquet_reader": {"resources": Resources(cpus=1.0)},
            }
        )
    )

    pipeline.add_stage(
        CommonCrawlWARCReader(
            warc_filename_col="warc_filename",
            warc_record_offset_col="warc_record_offset",
            warc_record_length_col="warc_record_length",
        ).with_(resources=Resources(cpus=0.5))
    )

    pipeline.add_stage(ParquetWriter(path=output_path).with_(resources=Resources(cpus=1.0)))

    return pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare enriched FINEMATH_4PLUS dataset for math benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Directory to write enriched parquet files (e.g. /datasets/finemath4plus_enriched)",
    )
    parser.add_argument(
        "--datasets-config",
        type=Path,
        default=REPO_ROOT / "tutorials" / "math" / "datasets.json",
        help="Path to datasets.json configuration file",
    )
    parser.add_argument(
        "--dataset-name",
        default="FINEMATH_4PLUS",
        help="Dataset key from datasets.json to download",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="Maximum number of parquet files to download from HuggingFace (controls dataset size)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers for HuggingFace download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip HuggingFace download (use existing raw parquet at --raw-path)",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=None,
        help="Path to existing raw FINEMATH_4PLUS parquet files (used with --skip-download)",
    )
    parser.add_argument(
        "--executor",
        default="xenna",
        choices=["xenna", "ray_data", "ray_actors"],
        help="Executor to use for the enrichment pipeline",
    )

    args = parser.parse_args()
    output_path = args.output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Download raw FINEMATH_4PLUS from HuggingFace
    if args.skip_download:
        if not args.raw_path:
            parser.error("--raw-path is required when using --skip-download")
        raw_data_dir = args.raw_path.resolve()
        logger.info(f"Skipping download, using existing raw data at: {raw_data_dir}")
    else:
        math_dl = _import_math_download()

        config = math_dl.load_datasets_config(args.datasets_config)
        if args.dataset_name not in config:
            available = ", ".join(config.keys())
            parser.error(f"Unknown dataset: {args.dataset_name}\nAvailable: {available}")

        scratch_dir = output_path / "_raw_download"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {args.dataset_name} from HuggingFace (max_files={args.max_files})...")
        raw_data_dir = math_dl.download_dataset(
            dataset_name=args.dataset_name,
            config=config[args.dataset_name],
            download_config=math_dl.DownloadConfig(
                output_dir=scratch_dir,
                max_files=args.max_files,
                force=args.force,
                workers=args.workers,
            ),
        )
        logger.info(f"Raw data downloaded to: {raw_data_dir}")

    # Step 2: Run enrichment pipeline (fetch binary_content from CC)
    from utils import setup_executor

    logger.info("Building enrichment pipeline (ParquetReader -> CommonCrawlWARCReader -> ParquetWriter)...")
    enriched_dir = str(output_path / "enriched")
    pipeline = build_enrichment_pipeline(
        input_path=str(raw_data_dir),
        output_path=enriched_dir,
    )

    executor = setup_executor(args.executor)
    logger.info(f"Pipeline description:\n{pipeline.describe()}")
    logger.info("Starting enrichment pipeline...")

    try:
        results = pipeline.run(executor, initial_tasks=None)
        total_docs = sum(task.num_items for task in results) if results else 0
        logger.success(f"Enrichment complete: {total_docs} documents with binary_content")
        logger.success(f"Enriched parquet written to: {enriched_dir}")
    except Exception as e:
        logger.error(f"Enrichment pipeline failed: {e}")
        return 1
    else:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
