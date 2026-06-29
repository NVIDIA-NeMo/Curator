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

"""Create a global unique URL list from configured Common Crawl Index snapshots."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import OmegaConf

DEFAULT_CONFIG = Path(__file__).with_name("selected_crawls.yaml")
DEFAULT_REGION = "us-east-1"
STAGING_FIELDS = [
    "url",
    "warc_filename",
]


@dataclass(frozen=True)
class CCIndexUrlListConfig:
    """Configuration loaded from ``selected_crawls.yaml``."""

    cc_index_base_uri: str
    endpoint_url: str
    included_crawls: list[str]
    output_name: str = "global_unique_urls"


def load_yaml_mapping(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a mapping."""
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        msg = f"Config must be a YAML mapping: {config_path}"
        raise TypeError(msg)
    return raw


def require_non_empty_string(raw: dict[str, Any], key: str) -> str:
    """Return a stripped non-empty string config value."""
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        msg = f"{key} must be a non-empty string."
        raise ValueError(msg)
    return value.strip()


def validate_included_crawls(raw: dict[str, Any]) -> list[str]:
    """Return validated CC-MAIN crawl IDs from the config."""
    included_crawls = raw["included_crawls"]
    if not isinstance(included_crawls, list) or not included_crawls:
        msg = "included_crawls must be a non-empty YAML list."
        raise ValueError(msg)
    if not all(isinstance(crawl, str) and crawl.startswith("CC-MAIN-") for crawl in included_crawls):
        msg = "Every included_crawls entry must be a CC-MAIN crawl ID string."
        raise ValueError(msg)
    if len(set(included_crawls)) != len(included_crawls):
        msg = "included_crawls contains duplicate crawl IDs."
        raise ValueError(msg)
    return list(included_crawls)


def normalize_output_name(raw: dict[str, Any]) -> str:
    """Return the final output dataset directory name."""
    output_name = raw.get("output_name", "global_unique_urls")
    if not isinstance(output_name, str):
        msg = "output_name must be a non-empty string."
        raise TypeError(msg)
    output_name = output_name.strip().strip("/")
    if not output_name:
        msg = "output_name must be a non-empty string."
        raise ValueError(msg)
    return output_name


def load_config(config_path: str | Path) -> CCIndexUrlListConfig:
    """Load and validate the YAML tutorial config."""
    raw = load_yaml_mapping(config_path)
    if "excluded_crawls" in raw:
        msg = "Config does not support excluded_crawls; omit crawls from included_crawls instead."
        raise ValueError(msg)
    required = {"cc_index_base_uri", "endpoint_url", "included_crawls"}
    missing = required - raw.keys()
    if missing:
        msg = f"Config is missing required key(s): {sorted(missing)}"
        raise ValueError(msg)

    return CCIndexUrlListConfig(
        cc_index_base_uri=require_non_empty_string(raw, "cc_index_base_uri").rstrip("/"),
        endpoint_url=require_non_empty_string(raw, "endpoint_url"),
        included_crawls=validate_included_crawls(raw),
        output_name=normalize_output_name(raw),
    )


def build_crawl_uri(cc_index_base_uri: str, crawl: str) -> str:
    """Return the CC Index parquet directory for one crawl."""
    return f"{cc_index_base_uri.rstrip('/')}/crawl={crawl}/subset=warc/"


def build_crawl_uris(config: CCIndexUrlListConfig) -> list[str]:
    """Return CC Index parquet directories for the configured crawls."""
    return [build_crawl_uri(config.cc_index_base_uri, crawl) for crawl in config.included_crawls]


def build_storage_options(config: CCIndexUrlListConfig, region_name: str = DEFAULT_REGION) -> dict[str, Any]:
    """Build fsspec/s3fs storage options for the PBSS Common Crawl namespace."""
    if not config.cc_index_base_uri.startswith("s3://"):
        return {}

    access_key = os.environ.get("PBSS_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("PBSS_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        msg = "Set PBSS_ACCESS_KEY_ID and PBSS_SECRET_ACCESS_KEY before reading PBSS CC Index parquet."
        raise RuntimeError(msg)

    storage_options: dict[str, Any] = {
        "key": access_key,
        "secret": secret_key,
        "client_kwargs": {
            "endpoint_url": config.endpoint_url,
            "region_name": region_name,
        },
    }
    if session_token := os.environ.get("AWS_SESSION_TOKEN"):
        storage_options["token"] = session_token
    return storage_options


def limit_files_per_crawl(
    crawl_uris: list[str],
    storage_options: dict[str, Any],
    max_files_per_crawl: int,
) -> list[str]:
    """Expand crawl directories to a capped file list for smoke tests."""
    from nemo_curator.utils.file_utils import get_all_file_paths_under

    file_paths: list[str] = []
    for crawl_uri in crawl_uris:
        files = sorted(
            get_all_file_paths_under(
                crawl_uri,
                keep_extensions=[".parquet"],
                storage_options=storage_options,
            )
        )
        if not files:
            msg = f"No parquet files found at {crawl_uri}"
            raise FileNotFoundError(msg)
        file_paths.extend(files[:max_files_per_crawl])
    return file_paths


def build_input_paths(
    config: CCIndexUrlListConfig,
    storage_options: dict[str, Any],
    max_files_per_crawl: int | None,
) -> list[str]:
    """Return crawl directories for full runs, or capped files for smoke tests."""
    crawl_uris = build_crawl_uris(config)
    if max_files_per_crawl is None:
        return crawl_uris
    return limit_files_per_crawl(crawl_uris, storage_options, max_files_per_crawl)


def parquet_summary(directory: Path) -> tuple[int, float]:
    """Return ``(file_count, total_megabytes)`` for local parquet files."""
    files = sorted(directory.rglob("*.parquet"))
    total_mb = sum(file.stat().st_size for file in files) / (1024 * 1024)
    return len(files), total_mb


def stage_url_dataset(
    input_paths: list[str],
    storage_options: dict[str, Any],
    staging_dir: Path,
    blocksize: str,
) -> None:
    """Use Curator's ParquetReader to stage URL and WARC filename fields."""
    from nemo_curator.backends.ray_data import RayDataExecutor
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.text.io.reader import ParquetReader
    from nemo_curator.stages.text.io.writer import ParquetWriter

    read_kwargs = {"storage_options": storage_options} if storage_options else {}
    logger.info(
        f"Phase 1: staging URL/WARC filename columns from {len(input_paths):,} input path(s) "
        f"with target blocksize {blocksize}",
    )

    Pipeline(
        name="cc_index_url_column",
        stages=[
            ParquetReader(
                file_paths=input_paths,
                blocksize=blocksize,
                fields=STAGING_FIELDS,
                read_kwargs=read_kwargs,
            ),
            ParquetWriter(
                path=str(staging_dir),
                write_kwargs={"compression": "snappy"},
                fields=STAGING_FIELDS,
                mode="overwrite",
            ),
        ],
    ).run(executor=RayDataExecutor())

    n_files, total_mb = parquet_summary(staging_dir)
    logger.info(f"Phase 1 complete: {n_files:,} file(s), {total_mb:,.1f} MiB at {staging_dir}")


def run_exact_url_dedup(staging_dir: Path, dedup_ids_dir: Path, final_dir: Path, blocksize: str) -> None:
    """Run Curator exact deduplication over the staged ``url`` dataset."""
    from nemo_curator.stages.deduplication.exact.identification import ExactDuplicateIdentification
    from nemo_curator.stages.deduplication.exact.workflow import (
        ID_GENERATOR_OUTPUT_FILENAME,
        ExactDeduplicationWorkflow,
    )
    from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
    from nemo_curator.stages.text.deduplication import TextDuplicatesRemovalWorkflow

    logger.info(f"Phase 2: identifying duplicate URLs in {staging_dir}")
    dedup_result = ExactDeduplicationWorkflow(
        input_path=str(staging_dir),
        output_path=str(dedup_ids_dir),
        text_field="url",
        input_filetype="parquet",
        input_blocksize=blocksize,
        assign_id=True,
    ).run()
    logger.info(f"Phase 2 complete: {dedup_result.metadata.get('num_duplicates', 0):,} duplicate IDs")

    logger.info(f"Phase 3: writing global unique URLs to {final_dir}")
    removal_result = TextDuplicatesRemovalWorkflow(
        input_path=str(staging_dir),
        ids_to_remove_path=str(dedup_ids_dir / ExactDuplicateIdentification.name),
        output_path=str(final_dir),
        input_filetype="parquet",
        input_blocksize=blocksize,
        output_fields=STAGING_FIELDS,
        output_filetype="parquet",
        output_mode="overwrite",
        duplicate_id_field=CURATOR_DEDUP_ID_STR,
        id_generator_path=str(dedup_ids_dir / ID_GENERATOR_OUTPUT_FILENAME),
    ).run()
    logger.info(f"Phase 3 complete: {removal_result.metadata.get('num_duplicates_removed', 0):,} rows removed")


def run(config: CCIndexUrlListConfig, args: argparse.Namespace) -> None:
    """Run the URL-column staging and Curator exact dedup workflow."""
    t0 = time.time()
    output_root = Path(args.output)
    staging_dir = output_root / "_staging_url_column"
    dedup_ids_dir = output_root / "_dedup_ids"
    final_dir = output_root / config.output_name
    needs_storage = not args.dry_run or args.max_files_per_crawl is not None
    storage_options = build_storage_options(config, region_name=args.region_name) if needs_storage else {}
    input_paths = build_input_paths(config, storage_options, args.max_files_per_crawl)

    logger.info(f"Configured crawls: {', '.join(config.included_crawls)}")
    logger.info(f"Input paths: {len(input_paths):,}")
    logger.info(f"Output dataset: {final_dir}")
    if args.dry_run:
        return

    stage_url_dataset(input_paths, storage_options, staging_dir, args.staging_blocksize)
    run_exact_url_dedup(staging_dir, dedup_ids_dir, final_dir, args.dedup_blocksize)

    n_files, total_mb = parquet_summary(final_dir)
    logger.info("=" * 72)
    logger.info(f"Output files: {n_files:,}")
    logger.info(f"Output size:  {total_mb:,.1f} MiB")
    logger.info(f"Output dir:   {final_dir}")
    logger.info(f"Elapsed:      {time.time() - t0:.0f}s")
    logger.info("=" * 72)


def positive_int(value: str) -> int:
    """Argparse type for positive integer options."""
    parsed = int(value)
    if parsed <= 0:
        msg = "value must be positive"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a global unique URL parquet dataset from configured CC Index snapshots.",
    )
    parser.add_argument("--output", required=True, help="Output root directory")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"YAML config path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--max-files-per-crawl",
        type=positive_int,
        default=None,
        help="Use only the first N parquet files per crawl for smoke tests",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print configured inputs without running Curator")
    parser.add_argument("--region-name", default=DEFAULT_REGION, help=f"S3 region name (default: {DEFAULT_REGION})")
    parser.add_argument(
        "--staging-blocksize",
        default="512MB",
        help="Target source parquet bytes per URL-column staging task (default: 512MB)",
    )
    parser.add_argument(
        "--dedup-blocksize",
        default="512MB",
        help="Input blocksize for exact dedup and duplicate removal (default: 512MB)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(load_config(args.config), args)


if __name__ == "__main__":
    main()
