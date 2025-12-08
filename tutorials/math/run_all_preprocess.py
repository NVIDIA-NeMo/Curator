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
Orchestration script for running math preprocessing on multiple datasets.

Dataset configurations are loaded from datasets.json in the same directory.

This script automatically handles sequencing for two types of datasets:

1. DATASETS WITH FULL WARC METADATA (Direct Fetch):
   - FINEMATH_3PLUS, FINEMATH_4PLUS
   - Config: fetch_cc=True, needs_cc_lookup=False
   - These have: url, warc_filename, warc_record_offset, warc_record_length
   - Fetches directly from Common Crawl via HTTPS

2. DATASETS REQUIRING CC INDEX LOOKUP (Two-Phase):
   - OPENWEBMATH, INFIWEBMATH, MEGAMATH
   - Config: fetch_cc=True, needs_cc_lookup=True
   - These only have: url (no WARC metadata)
   - Script automatically:
     1. Queries CC Index on S3 to find WARC locations (no download required)
     2. Keeps only URLs that exist in the specified crawl(s)
     3. Fetches content from CC using enriched WARC metadata

Examples:
    # FineMath - has WARC metadata, fetches directly from CC
    python run_all_preprocess.py --output-base /output --datasets FINEMATH_4PLUS

    # OpenWebMath - needs CC lookup, specify crawls to search
    python run_all_preprocess.py --output-base /output --datasets OPENWEBMATH \\
        --crawls CC-MAIN-2024-10

    # Multiple crawls for better URL coverage
    python run_all_preprocess.py --output-base /output --datasets OPENWEBMATH \\
        --crawls CC-MAIN-2024-10 CC-MAIN-2024-18 CC-MAIN-2023-50

    # Process all datasets that are ready (have WARC metadata, no lookup needed)
    python run_all_preprocess.py --output-base /output --ready-only

    # Process all datasets with CC lookup
    python run_all_preprocess.py --output-base /output \\
        --crawls CC-MAIN-2024-10 CC-MAIN-2024-18

    # Continue processing even if one dataset fails
    python run_all_preprocess.py --output-base /output \\
        --crawls CC-MAIN-2024-10 \\
        --continue-on-error

    # Use a custom datasets config file
    python run_all_preprocess.py --output-base /output \\
        --datasets-config /path/to/my_datasets.json \\
        --crawls CC-MAIN-2024-10
"""

import json
import subprocess
import sys
from pathlib import Path

from loguru import logger

# Default path to datasets configuration
DEFAULT_DATASETS_CONFIG = Path(__file__).parent / "datasets.json"


def load_datasets_config(config_path: Path) -> dict:
    """
    Load dataset configurations from JSON file.

    Args:
        config_path: Path to the datasets.json file.

    Returns:
        Dictionary of dataset configurations.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Datasets config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Remove metadata keys (those starting with _)
    datasets = {k: v for k, v in config.items() if not k.startswith("_")}

    logger.info(f"Loaded {len(datasets)} dataset configurations from {config_path}")
    return datasets


def run_cc_index_lookup(
    name: str,
    config: dict,
    base_output_dir: str,
    crawls: list[str],
    continue_on_error: bool = False,
) -> str | None:
    """
    Run CC Index lookup to enrich dataset with WARC metadata.

    Queries the public CC Index on S3 directly.

    Returns the path to the enriched output directory.
    """
    script_path = Path(__file__).parent / "run_cc_index_lookup_local.py"
    enriched_output = Path(base_output_dir) / f"{config['output']}_enriched"

    logger.info(f"Running CC Index lookup for: {name}")
    logger.info(f"Input: {config['input']}")
    logger.info(f"Crawls: {crawls}")
    logger.info(f"Enriched output: {enriched_output}")

    cmd = [
        sys.executable,
        str(script_path),
        "--input", config["input"],
        "--output", str(enriched_output),
        "--url-col", config.get("url_col", "url"),
        "--crawls", *crawls,
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)  # noqa: S603
        logger.success(f"Successfully enriched {name} with WARC metadata")
        return str(enriched_output)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to enrich {name}: {e}")
        if not continue_on_error:
            raise
        return None


def run_preprocess(
    name: str,
    config: dict,
    input_path: str,
    base_output_dir: str,
    fetch_cc: bool = False,
    continue_on_error: bool = False,
) -> None:
    """Run the preprocessing pipeline on a dataset."""
    script_path = Path(__file__).parent / "run_text_preprocess.py"
    output_path = Path(base_output_dir) / config["output"]

    logger.info(f"Processing dataset: {name}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--input", input_path,
        "--output", str(output_path),
        "--report-stats",
    ]

    if fetch_cc:
        cmd.append("--fetch-cc")
        cols = config.get("columns", {})
        if "warc_filename" in cols:
            cmd.extend(["--warc-filename-col", cols["warc_filename"]])
        if "offset" in cols:
            cmd.extend(["--offset-col", cols["offset"]])
        if "length" in cols:
            cmd.extend(["--length-col", cols["length"]])

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)  # noqa: S603
        logger.success(f"Successfully processed {name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to process {name}: {e}")
        if not continue_on_error:
            raise


def run_dataset(
    name: str,
    config: dict,
    base_output_dir: str,
    crawls: list[str] | None = None,
    continue_on_error: bool = False,
) -> None:
    """
    Run the full pipeline for a dataset.

    Automatic sequencing logic:
    - If needs_cc_lookup=True AND fetch_cc=True AND crawls provided:
        1. Run CC Index lookup (queries S3 directly)
        2. Run preprocessing with --fetch-cc on enriched data
    - If needs_cc_lookup=False AND fetch_cc=True:
        Run preprocessing with --fetch-cc directly (has WARC metadata)
    - If fetch_cc=False:
        Run preprocessing without CC fetch (use existing text in dataset)
    """
    needs_lookup = config.get("needs_cc_lookup", False)
    fetch_cc = config.get("fetch_cc", False)

    # Both flags are True: needs lookup AND wants to fetch from CC
    if needs_lookup and fetch_cc:
        if not crawls:
            logger.error(
                f"Dataset {name} requires CC Index lookup but no --crawls provided. "
                "Specify crawl IDs to search (e.g., --crawls CC-MAIN-2024-10)."
            )
            if not continue_on_error:
                raise ValueError(f"Missing --crawls for {name}")
            return

        # Step 1: Run CC Index lookup on S3
        logger.info(f"Dataset {name} needs CC Index lookup. Querying CC Index on S3...")
        enriched_path = run_cc_index_lookup(
            name, config, base_output_dir, crawls, continue_on_error
        )
        if not enriched_path:
            return  # Lookup failed

        # Step 2: Run preprocessing on enriched data with CC fetch
        logger.info(f"Dataset {name} lookup complete. Now fetching from CC...")
        run_preprocess(
            name,
            config,
            input_path=f"{enriched_path}/*.parquet",
            base_output_dir=base_output_dir,
            fetch_cc=True,
            continue_on_error=continue_on_error,
        )
    elif fetch_cc and not needs_lookup:
        # Has WARC metadata - fetch directly from CC
        logger.info(f"Dataset {name} has full WARC metadata. Fetching from CC directly.")
        run_preprocess(
            name,
            config,
            input_path=config["input"],
            base_output_dir=base_output_dir,
            fetch_cc=True,
            continue_on_error=continue_on_error,
        )
    else:
        # No CC fetch - process existing text data in dataset
        logger.info(f"Dataset {name}: Processing existing text data (no CC fetch).")
        run_preprocess(
            name,
            config,
            input_path=config["input"],
            base_output_dir=base_output_dir,
            fetch_cc=False,
            continue_on_error=continue_on_error,
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run preprocessing for all configured math datasets. "
        "Dataset configurations are loaded from datasets.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FineMath - has WARC metadata, fetches directly from CC
  python run_all_preprocess.py --output-base /output --datasets FINEMATH_4PLUS

  # OpenWebMath - needs CC lookup, specify crawls to search
  python run_all_preprocess.py --output-base /output --datasets OPENWEBMATH \\
      --crawls CC-MAIN-2024-10

  # Multiple crawls for better URL coverage
  python run_all_preprocess.py --output-base /output --datasets OPENWEBMATH \\
      --crawls CC-MAIN-2024-10 CC-MAIN-2024-18 CC-MAIN-2023-50

  # Process all datasets that are ready (have WARC metadata, no lookup needed)
  python run_all_preprocess.py --output-base /output --ready-only

  # Use a custom datasets config file
  python run_all_preprocess.py --output-base /output \\
      --datasets-config /path/to/my_datasets.json \\
      --crawls CC-MAIN-2024-10

  # Continue processing even if one dataset fails
  python run_all_preprocess.py --output-base /output \\
      --crawls CC-MAIN-2024-10 \\
      --continue-on-error

Dataset Types (see datasets.json):
  - FINEMATH_3PLUS, FINEMATH_4PLUS: Have full WARC metadata (can fetch directly)
  - OPENWEBMATH, INFIWEBMATH_3PLUS, INFIWEBMATH_4PLUS, MEGAMATH_WEB, MEGAMATH_PRO:
    Need CC Index lookup (provide --crawls to enable)
        """,
    )
    parser.add_argument(
        "--output-base",
        required=True,
        help="Base directory for all outputs",
    )
    parser.add_argument(
        "--datasets-config",
        type=Path,
        default=DEFAULT_DATASETS_CONFIG,
        help=f"Path to datasets JSON config file (default: {DEFAULT_DATASETS_CONFIG})",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to run (default: all). See datasets.json for available datasets.",
    )
    parser.add_argument(
        "--crawls",
        nargs="+",
        help="Crawl IDs to search in CC Index (e.g., CC-MAIN-2024-10). "
        "Required for datasets that need CC lookup. Multiple crawls improve URL coverage.",
    )
    parser.add_argument(
        "--ready-only",
        action="store_true",
        help="Only process datasets that have full WARC metadata (skip those needing CC lookup)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing datasets even if one fails",
    )

    args = parser.parse_args()

    # Load datasets configuration
    datasets = load_datasets_config(args.datasets_config)

    # Validate dataset names if specified
    if args.datasets:
        invalid = [d for d in args.datasets if d not in datasets]
        if invalid:
            logger.error(f"Unknown datasets: {invalid}. Available: {list(datasets.keys())}")
            return
        targets = args.datasets
    elif args.ready_only:
        targets = [name for name, cfg in datasets.items() if not cfg.get("needs_cc_lookup")]
        logger.info(f"Processing only datasets with full WARC metadata: {targets}")
    else:
        targets = list(datasets.keys())

    # Process each dataset
    for name in targets:
        config = datasets[name]

        run_dataset(
            name,
            config,
            args.output_base,
            crawls=args.crawls,
            continue_on_error=args.continue_on_error,
        )


if __name__ == "__main__":
    main()
