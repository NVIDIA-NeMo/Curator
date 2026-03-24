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

Two subcommands prepare data for different benchmark stages:

  enrichment
      Downloads FINEMATH_4PLUS from HuggingFace, then fetches raw binary
      content from Common Crawl using WARC metadata.  The resulting enriched
      parquet files contain a ``binary_content`` column that the nightly math
      benchmark can consume without network I/O.

  cc-index
      Downloads a URL-only math dataset (e.g., MEGAMATH_WEB) from HuggingFace
      and a subset of CC Index partition files from S3.

This script is NOT part of the nightly benchmark YAML -- it is run once
(or whenever the dataset needs refreshing).

Example usage:

    # Enrichment: download FINEMATH_4PLUS and fetch binary_content
    python prepare_math_benchmark_data.py enrichment \\
        --output-path /datasets/finemath4plus_enriched \\
        --max-files 5 --workers 8

    # CC Index: download OPENWEBMATH + CC Index partitions
    python prepare_math_benchmark_data.py cc-index \\
        --output-path /datasets/cc_index_benchmark \\
        --dataset-name MEGAMATH_WEB --max-files 3 \\
        --crawl CC-MAIN-2024-10 --num-partitions 50
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.common_crawl.download import CommonCrawlWARCReader
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter

if TYPE_CHECKING:
    import botocore.client

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_S3_BUCKET = "cc-index"
DEFAULT_CC_INDEX_PREFIX = "table/cc-main/warc"
_MIN_HF_PATH_PARTS = 2


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
    """Parse HuggingFace path into (repo_id, optional subset).

    Accepts ``org/repo`` (no subset) or ``org/repo/sub/path/...``
    where everything after the second ``/`` is joined as the subset.
    """
    parts = hf_path.split("/")
    if len(parts) < _MIN_HF_PATH_PARTS:
        msg = f"Invalid HuggingFace path (need at least org/repo): {hf_path}"
        raise ValueError(msg)

    repo_id = f"{parts[0]}/{parts[1]}"
    subset = "/".join(parts[2:]) or None
    return repo_id, subset


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


def _create_s3_client() -> botocore.client.BaseClient:
    """Create an S3 client using boto3's standard credential chain."""
    return boto3.client("s3", config=BotoConfig(signature_version="s3v4"))


def list_cc_index_partitions(
    s3_client: botocore.client.BaseClient,
    bucket: str,
    crawl: str,
    prefix: str = DEFAULT_CC_INDEX_PREFIX,
    limit: int | None = None,
) -> list[dict]:
    """List CC Index partition files for a crawl.

    Returns a sorted list of dicts with 'key' and 'size' (bytes).
    When *limit* is set, stops paginating once enough files are collected.

    Raises ``RuntimeError`` on S3 access failures.
    """
    full_prefix = f"{prefix}/crawl={crawl}/subset=warc/"
    paginator = s3_client.get_paginator("list_objects_v2")

    try:
        files: list[dict] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    files.append({"key": obj["Key"], "size": obj["Size"]})
                    if limit and len(files) >= limit:
                        break
            if limit and len(files) >= limit:
                break
    except (BotoCoreError, ClientError) as e:
        msg = f"Failed to list CC Index partitions at s3://{bucket}/{full_prefix}: {e}"
        raise RuntimeError(msg) from e

    return sorted(files, key=lambda f: f["key"])


def download_cc_index_partitions(
    s3_client: botocore.client.BaseClient,
    bucket: str,
    partitions: list[dict],
    warc_dir: Path,
    *,
    force: bool = False,
) -> list[Path]:
    """Download CC Index partition files from S3.

    *warc_dir* is the final directory to write files into (caller builds
    the hive-partitioned path).  Returns list of local paths.

    Raises ``RuntimeError`` on download failure or post-download size mismatch.
    """
    warc_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    total = len(partitions)

    for i, part in enumerate(partitions, 1):
        filename = Path(part["key"]).name
        local_path = warc_dir / filename
        expected_size = part["size"]

        if not force and local_path.exists() and local_path.stat().st_size == expected_size:
            logger.info(f"[{i}/{total}] Already exists: {filename} ({expected_size / (1024**3):.2f} GB)")
            downloaded.append(local_path)
            continue

        logger.info(f"[{i}/{total}] Downloading {filename} ({expected_size / (1024**3):.2f} GB)...")
        try:
            s3_client.download_file(bucket, part["key"], str(local_path))
        except (BotoCoreError, ClientError) as e:
            msg = f"Failed to download s3://{bucket}/{part['key']}: {e}"
            raise RuntimeError(msg) from e

        actual_size = local_path.stat().st_size
        if actual_size != expected_size:
            local_path.unlink(missing_ok=True)
            msg = f"Size mismatch for {filename}: expected {expected_size} bytes, got {actual_size} bytes"
            raise RuntimeError(msg)

        logger.info(f"[{i}/{total}] Done: {actual_size / (1024**3):.2f} GB")
        downloaded.append(local_path)

    return downloaded


def _scan_parquet_dir(path: Path) -> tuple[list[Path], float]:
    """Return (parquet files, total size in bytes) under *path*."""
    files = sorted(path.rglob("*.parquet"))
    total_bytes = sum(f.stat().st_size for f in files)
    return files, total_bytes


def _download_hf_dataset(args: argparse.Namespace, output_subdir: str = "_raw_download") -> Path:
    """Shared HF download logic for both subcommands.  Returns dataset directory.

    Raises ``ValueError`` on configuration errors (missing args, unknown dataset).
    """
    if args.skip_download:
        if not args.raw_path:
            msg = "--raw-path is required when using --skip-download"
            raise ValueError(msg)
        dataset_dir = args.raw_path.resolve()
        logger.info(f"Skipping download, using existing data at: {dataset_dir}")
        return dataset_dir

    config = load_datasets_config(args.datasets_config)
    if args.dataset_name not in config:
        available = ", ".join(config.keys())
        msg = f"Unknown dataset: {args.dataset_name}\nAvailable: {available}"
        raise ValueError(msg)

    dataset_config = config[args.dataset_name]
    output_path = args.output_path.resolve()

    logger.info(f"Downloading {args.dataset_name} from HuggingFace (max_files={args.max_files})...")
    dataset_dir = download_dataset(
        dataset_name=args.dataset_name,
        config=dataset_config,
        download_config=DownloadConfig(
            output_dir=output_path / output_subdir,
            max_files=args.max_files,
            force=args.force,
            workers=args.workers,
        ),
    )
    logger.info(f"Raw data downloaded to: {dataset_dir}")
    return dataset_dir


def run_enrichment(args: argparse.Namespace) -> int:
    """Download HF dataset and run WARC enrichment pipeline."""
    output_path = args.output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        raw_data_dir = _download_hf_dataset(args)
    except ValueError as e:
        logger.error(str(e))
        return 1

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
    except Exception:
        logger.exception("Enrichment pipeline failed")
        return 1
    else:
        return 0


def _prepare_cc_index(args: argparse.Namespace, output_path: Path) -> None:
    """Core logic for cc-index subcommand.

    Raises ``ValueError`` or ``RuntimeError`` on any failure so that
    ``run_cc_index`` can map them to a non-zero exit code.
    """
    dataset_dir = _download_hf_dataset(args, output_subdir="dataset")

    dataset_files, dataset_size = _scan_parquet_dir(dataset_dir)
    if not dataset_files:
        msg = f"No parquet files found under {dataset_dir}"
        raise RuntimeError(msg)
    dataset_size_mb = dataset_size / (1024**2)
    logger.info(f"Dataset: {len(dataset_files)} parquet files, {dataset_size_mb:.1f} MB")

    if args.skip_cc_download:
        if not args.cc_index_path:
            msg = "--cc-index-path is required when using --skip-cc-download"
            raise ValueError(msg)
        cc_index_dir = args.cc_index_path.resolve()
        logger.info(f"Skipping CC Index download, using existing data at: {cc_index_dir}")
    else:
        s3_client = _create_s3_client()
        logger.info(f"Listing CC Index partitions for crawl={args.crawl} in s3://{args.s3_bucket}/...")
        all_partitions = list_cc_index_partitions(s3_client, args.s3_bucket, args.crawl, limit=args.num_partitions)

        if not all_partitions:
            msg = f"No CC Index partitions found for crawl={args.crawl}"
            raise RuntimeError(msg)

        total_size_gb = sum(p["size"] for p in all_partitions) / (1024**3)
        logger.info(f"Selected {len(all_partitions)} partitions ({total_size_gb:.2f} GB) for {args.crawl}")

        cc_index_dir = output_path / "cc_index"
        warc_dir = cc_index_dir / f"crawl={args.crawl}" / "subset=warc"
        download_cc_index_partitions(s3_client, args.s3_bucket, all_partitions, warc_dir, force=args.force)

    cc_files, cc_size = _scan_parquet_dir(cc_index_dir)
    if not cc_files:
        msg = f"No CC Index parquet files found under {cc_index_dir}"
        raise RuntimeError(msg)
    total_cc_size_gb = cc_size / (1024**3)

    logger.success("=" * 60)
    logger.success("CC Index Lookup benchmark data prepared:")
    logger.success(f"  Dataset:         {len(dataset_files)} files, {dataset_size_mb:.1f} MB ({dataset_dir})")
    logger.success(f"  CC Index:        {len(cc_files)} files, {total_cc_size_gb:.2f} GB ({cc_index_dir})")
    logger.success(f"  Output root:     {output_path}")
    logger.success("=" * 60)


def run_cc_index(args: argparse.Namespace) -> int:
    """Download HF dataset and CC Index partitions from S3."""
    output_path = args.output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        _prepare_cc_index(args, output_path)
    except (ValueError, RuntimeError) as e:
        logger.error(str(e))
        return 1

    return 0


def _positive_int(value: str) -> int:
    """argparse type that rejects zero and negative integers."""
    try:
        n = int(value)
    except ValueError:
        msg = f"invalid int value: {value!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if n < 1:
        msg = f"must be >= 1, got {n}"
        raise argparse.ArgumentTypeError(msg)
    return n


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by both subcommands."""
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Root output directory",
    )
    parser.add_argument(
        "--datasets-config",
        type=Path,
        default=REPO_ROOT / "tutorials" / "math" / "datasets.json",
        help="Path to datasets.json configuration file",
    )
    parser.add_argument(
        "--dataset-name",
        help="Dataset key from datasets.json",
    )
    parser.add_argument(
        "--max-files",
        type=_positive_int,
        help="Maximum number of parquet files to download from HuggingFace",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=4,
        help="Number of parallel HuggingFace download workers",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip HuggingFace download (use existing parquet at --raw-path)",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=None,
        help="Path to existing raw parquet files (used with --skip-download)",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare data for math pipeline benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    enrich_parser = subparsers.add_parser(
        "enrichment",
        help="Download HF dataset and fetch binary_content via WARC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(enrich_parser)
    enrich_parser.set_defaults(dataset_name="FINEMATH_4PLUS", max_files=5)
    enrich_parser.add_argument(
        "--executor",
        default="xenna",
        choices=["xenna", "ray_data"],
        help="Executor to use for the enrichment pipeline",
    )

    cc_parser = subparsers.add_parser(
        "cc-index",
        help="Download HF dataset + CC Index partitions from S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(cc_parser)
    cc_parser.set_defaults(dataset_name="MEGAMATH_WEB", max_files=3)
    cc_group = cc_parser.add_argument_group("CC Index Partitions")
    cc_group.add_argument("--crawl", default="CC-MAIN-2024-10", help="Crawl ID (e.g. CC-MAIN-2024-10)")
    cc_group.add_argument(
        "--num-partitions", type=_positive_int, default=50, help="Number of CC Index partition files"
    )
    cc_group.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET, help="S3 bucket for CC Index")
    cc_group.add_argument("--skip-cc-download", action="store_true", help="Skip CC Index download")
    cc_group.add_argument("--cc-index-path", type=Path, default=None, help="Existing local CC Index path")

    args = parser.parse_args()

    if args.command == "enrichment":
        return run_enrichment(args)
    if args.command == "cc-index":
        return run_cc_index(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
