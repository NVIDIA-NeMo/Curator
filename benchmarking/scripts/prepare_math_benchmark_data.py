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
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.common_crawl.download import CommonCrawlWARCReader
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# HuggingFace download helpers
# ---------------------------------------------------------------------------


@dataclass
class DownloadConfig:
    """Configuration for dataset download."""

    output_dir: Path
    max_files: int | None = None
    force: bool = False
    workers: int = 1


def load_datasets_config(config_path: Path) -> dict:
    """Load dataset configurations from JSON file."""
    with open(config_path) as f:
        config = json.load(f)
    return {k: v for k, v in config.items() if not k.startswith("_")}


def _parse_huggingface_path(hf_path: str) -> tuple[str, str | None]:
    """Parse HuggingFace path into repo_id and optional subset/config."""
    repo_parts = 2
    repo_with_subset_parts = 3

    parts = hf_path.split("/")
    if len(parts) == repo_parts:
        return hf_path, None
    elif len(parts) == repo_with_subset_parts:
        repo_id = f"{parts[0]}/{parts[1]}"
        subset = parts[2]
        return repo_id, subset
    else:
        msg = f"Invalid HuggingFace path format: {hf_path}"
        raise ValueError(msg)


def _get_parquet_files(repo_id: str, subset: str | None = None) -> list[str]:
    """Get list of parquet files from a HuggingFace repository."""
    try:
        all_files = list_repo_files(repo_id, repo_type="dataset")
    except (HfHubHTTPError, RepositoryNotFoundError) as e:
        logger.error(f"Failed to list files in {repo_id}: {e}")
        raise

    parquet_files = [f for f in all_files if f.endswith(".parquet")]

    if subset:
        subset_patterns = [f"{subset}/", f"data/{subset}/", f"train/{subset}/"]
        subset_files = []
        for pattern in subset_patterns:
            subset_files.extend([f for f in parquet_files if f.startswith(pattern)])
        parquet_files = subset_files or [f for f in parquet_files if subset in f]

    if not parquet_files:
        logger.warning(f"No parquet files found in {repo_id}" + (f" for subset {subset}" if subset else ""))

    return sorted(parquet_files)


def _download_single_file(
    repo_id: str,
    file_path: str,
    dataset_dir: Path,
    force: bool = False,
) -> tuple[str, Path | None, str | None]:
    """Download a single file from HuggingFace Hub."""
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset",
            local_dir=dataset_dir,
            force_download=force,
        )
        return (file_path, Path(downloaded), None)
    except (HfHubHTTPError, RepositoryNotFoundError, OSError) as e:
        return (file_path, None, str(e))


def download_dataset(
    dataset_name: str,
    config: dict,
    download_config: DownloadConfig,
) -> Path:
    """Download a dataset from HuggingFace Hub."""
    hf_path = config["huggingface"]
    repo_id, subset = _parse_huggingface_path(hf_path)

    dataset_dir_name = dataset_name.lower()
    dataset_dir = download_config.output_dir / dataset_dir_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {dataset_name} from {repo_id}" + (f" (subset: {subset})" if subset else ""))
    logger.info(f"Output directory: {dataset_dir}")

    parquet_files = _get_parquet_files(repo_id, subset)

    if download_config.max_files:
        parquet_files = parquet_files[: download_config.max_files]
        logger.info(f"Limiting download to {download_config.max_files} files")

    total_files = len(parquet_files)
    logger.info(f"Found {total_files} parquet files to download (workers: {download_config.workers})")

    downloaded_files = []
    failed_files = []

    with ThreadPoolExecutor(max_workers=download_config.workers) as executor:
        futures = {
            executor.submit(_download_single_file, repo_id, fp, dataset_dir, download_config.force): fp
            for fp in parquet_files
        }

        for completed, future in enumerate(as_completed(futures), 1):
            file_path = futures[future]
            filename = Path(file_path).name

            try:
                _, local_path, error = future.result()
                if error:
                    logger.error(f"[{completed}/{total_files}] Failed {filename}: {error}")
                    failed_files.append((file_path, error))
                elif local_path:
                    logger.info(f"[{completed}/{total_files}] Downloaded {filename}")
                    downloaded_files.append(local_path)
            except (RuntimeError, OSError) as e:
                logger.error(f"[{completed}/{total_files}] Failed {filename}: {e}")
                failed_files.append((file_path, str(e)))

    logger.info(f"Successfully downloaded {len(downloaded_files)} files to {dataset_dir}")
    if failed_files:
        logger.warning(f"Failed to download {len(failed_files)} files:")
        for fp, err in failed_files:
            logger.warning(f"  - {fp}: {err}")

    return dataset_dir


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
        choices=["xenna", "ray_data"],
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
        config = load_datasets_config(args.datasets_config)
        if args.dataset_name not in config:
            available = ", ".join(config.keys())
            parser.error(f"Unknown dataset: {args.dataset_name}\nAvailable: {available}")

        scratch_dir = output_path / "_raw_download"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {args.dataset_name} from HuggingFace (max_files={args.max_files})...")
        raw_data_dir = download_dataset(
            dataset_name=args.dataset_name,
            config=config[args.dataset_name],
            download_config=DownloadConfig(
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
