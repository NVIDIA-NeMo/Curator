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

"""
Download math datasets from HuggingFace Hub.

See --help for usage and README.md for full documentation.
For authentication, set HF_TOKEN or run: huggingface-cli login
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from loguru import logger


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

    # Filter out schema/comment entries
    return {k: v for k, v in config.items() if not k.startswith("_")}


def parse_huggingface_path(hf_path: str) -> tuple[str, str | None]:
    """
    Parse HuggingFace path into repo_id and optional subset/config.

    Examples:
        "HuggingFaceTB/finemath/finemath-4plus" -> ("HuggingFaceTB/finemath", "finemath-4plus")
        "open-web-math/open-web-math" -> ("open-web-math/open-web-math", None)
        "OpenCoder-LLM/opc-fineweb-math-corpus/infiwebmath-4plus"
            -> ("OpenCoder-LLM/opc-fineweb-math-corpus", "infiwebmath-4plus")
    """
    # Constants for path validation
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


def get_parquet_files(repo_id: str, subset: str | None = None) -> list[str]:
    """Get list of parquet files from a HuggingFace repository."""
    try:
        all_files = list_repo_files(repo_id, repo_type="dataset")
    except (HfHubHTTPError, RepositoryNotFoundError) as e:
        logger.error(f"Failed to list files in {repo_id}: {e}")
        raise

    # Filter for parquet files
    parquet_files = [f for f in all_files if f.endswith(".parquet")]

    # If subset specified, filter to that subdirectory
    if subset:
        # Try common patterns: subset/, data/subset/, train/subset/
        subset_patterns = [
            f"{subset}/",
            f"data/{subset}/",
            f"train/{subset}/",
        ]
        subset_files = []
        for pattern in subset_patterns:
            subset_files.extend([f for f in parquet_files if f.startswith(pattern)])

        # Use subset files if found, otherwise fallback to files containing subset name
        parquet_files = subset_files if subset_files else [f for f in parquet_files if subset in f]

    if not parquet_files:
        logger.warning(f"No parquet files found in {repo_id}" + (f" for subset {subset}" if subset else ""))

    return sorted(parquet_files)


def download_single_file(
    repo_id: str,
    file_path: str,
    dataset_dir: Path,
    force: bool = False,
) -> tuple[str, Path | None, str | None]:
    """
    Download a single file from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID
        file_path: Path to file within the repository
        dataset_dir: Local directory to save the file
        force: Force re-download even if file exists

    Returns:
        Tuple of (file_path, local_path or None, error_message or None)
    """
    try:
        # Download directly to target directory (preserves repo structure)
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
    """
    Download a dataset from HuggingFace Hub.

    Args:
        dataset_name: Name of the dataset (key in datasets.json)
        config: Dataset configuration dict
        download_config: Configuration object with output_dir, max_files, force, workers

    Returns:
        Path to the downloaded dataset directory
    """
    hf_path = config["huggingface"]
    repo_id, subset = parse_huggingface_path(hf_path)

    # Create output directory using dataset name (lowercase, underscores)
    dataset_dir_name = dataset_name.lower()
    dataset_dir = download_config.output_dir / dataset_dir_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {dataset_name} from {repo_id}" + (f" (subset: {subset})" if subset else ""))
    logger.info(f"Output directory: {dataset_dir}")

    # Get list of parquet files
    parquet_files = get_parquet_files(repo_id, subset)

    if download_config.max_files:
        parquet_files = parquet_files[: download_config.max_files]
        logger.info(f"Limiting download to {download_config.max_files} files")

    total_files = len(parquet_files)
    logger.info(f"Found {total_files} parquet files to download (workers: {download_config.workers})")

    downloaded_files = []
    failed_files = []

    with ThreadPoolExecutor(max_workers=download_config.workers) as executor:
        futures = {
            executor.submit(download_single_file, repo_id, fp, dataset_dir, download_config.force): fp
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

    # Summary
    logger.info(f"Successfully downloaded {len(downloaded_files)} files to {dataset_dir}")
    if failed_files:
        logger.warning(f"Failed to download {len(failed_files)} files:")
        for fp, err in failed_files:
            logger.warning(f"  - {fp}: {err}")

    return dataset_dir


def list_datasets(config: dict) -> None:
    """Print available datasets and their info."""
    print("\nAvailable datasets:\n")
    print(f"{'Name':<20} {'HuggingFace Path':<50} {'Needs CC Lookup'}")
    print("-" * 85)

    for name, cfg in config.items():
        hf_path = cfg.get("huggingface", "N/A")
        needs_lookup = "Yes" if cfg.get("needs_cc_lookup", False) else "No"
        print(f"{name:<20} {hf_path:<50} {needs_lookup}")

    print("\nUsage: python 0_download.py --dataset <DATASET_NAME> --output-dir ./data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download math datasets from HuggingFace Hub")

    parser.add_argument(
        "--dataset",
        nargs="+",
        help="Dataset name(s) to download (from datasets.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("MATH_DATA_DIR", ".")) / "raw",
        help="Base output directory for downloaded data (default: $MATH_DATA_DIR/raw or ./raw)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "datasets.json",
        help="Path to datasets.json configuration file",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to download per dataset (for testing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel download workers (default: 1, recommended: 4-8 for large datasets)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_datasets_config(args.config)

    if args.list:
        list_datasets(config)
        return

    if not args.dataset:
        parser.error("--dataset is required (or use --list to see available datasets)")

    # Validate dataset names
    for dataset_name in args.dataset:
        if dataset_name not in config:
            available = ", ".join(config.keys())
            parser.error(f"Unknown dataset: {dataset_name}\nAvailable: {available}")

    # Download each dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.dataset:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {dataset_name}")
        logger.info(f"{'=' * 60}")

        try:
            download_config = DownloadConfig(
                output_dir=args.output_dir,
                max_files=args.max_files,
                force=args.force,
                workers=args.workers,
            )
            dataset_dir = download_dataset(
                dataset_name=dataset_name,
                config=config[dataset_name],
                download_config=download_config,
            )
            logger.info(f"Dataset ready at: {dataset_dir}")
        except (HfHubHTTPError, RepositoryNotFoundError, OSError) as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            raise

    logger.info("\nAll downloads complete!")


if __name__ == "__main__":
    main()
