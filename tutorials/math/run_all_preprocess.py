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

This script automatically handles sequencing for two types of datasets:

1. DATASETS WITH FULL WARC METADATA (Direct Range Fetch):
   - FINEMATH_3PLUS, FINEMATH_4PLUS
   - Config: fetch_cc=True, needs_cc_lookup=False
   - These have: url, warc_filename, warc_record_offset, warc_record_length
   - Fetches directly from Common Crawl

2. DATASETS REQUIRING CC INDEX LOOKUP (Automatic Two-Phase):
   - OPENWEBMATH, INFIWEBMATH, MEGAMATH
   - Config: fetch_cc=True, needs_cc_lookup=True
   - These only have: url (no WARC metadata)
   - Script automatically:
     1. Runs CC Index lookup for the specified --crawl-id
     2. Keeps only URLs that exist in that crawl (others are dropped)
     3. Fetches content from CC using enriched WARC metadata

   IMPORTANT: Not all URLs will be found in a given crawl!
   - The output will be a SUBSET of URLs that were captured in that crawl
   - Choose a crawl ID from: https://index.commoncrawl.org/
   - Recent crawls (e.g., CC-MAIN-2024-10) have better coverage of current pages

Examples:
    # FineMath - has WARC metadata, fetches directly from CC
    python run_all_preprocess.py --output-base /output --datasets FINEMATH_4PLUS

    # OpenWebMath - needs CC lookup, provide crawl-id
    # Only URLs found in CC-MAIN-2024-10 will be processed
    python run_all_preprocess.py --output-base /output --datasets OPENWEBMATH \\
        --crawl-id CC-MAIN-2024-10

    # Process all datasets that are ready (have WARC metadata, no lookup needed)
    python run_all_preprocess.py --output-base /output --ready-only

    # Process all datasets with CC lookup (requires --crawl-id)
    python run_all_preprocess.py --output-base /output --crawl-id CC-MAIN-2024-10

    # Process multiple specific datasets
    python run_all_preprocess.py --output-base /output \\
        --datasets FINEMATH_4PLUS OPENWEBMATH INFIWEBMATH_4PLUS \\
        --crawl-id CC-MAIN-2024-10

    # Continue processing even if one dataset fails
    python run_all_preprocess.py --output-base /output \\
        --crawl-id CC-MAIN-2024-10 \\
        --continue-on-error

Note: For datasets requiring CC lookup, if --crawl-id is not provided, the script
will process existing text data only (no CC fetch) and warn the user.
"""

import subprocess
import sys
from pathlib import Path

from loguru import logger

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ FULL WARC METADATA (Direct Range Fetch Possible)                            │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ ✅ FINEMATH_3PLUS │ HuggingFaceTB/finemath │ finemath-3plus/                │
# │ ✅ FINEMATH_4PLUS │ HuggingFaceTB/finemath │ finemath-4plus/                │
# │                                                                             │
# │ Columns: url, warc_filename, warc_record_offset, warc_record_length         │
# │ → Can fetch specific bytes directly from WARC (no CC Index needed)          │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ REQUIRES CC INDEX LOOKUP                                                    │
# │ (Has URL, needs offset/length from CC Index via run_cc_index_lookup.py)     │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ ❌ OPENWEBMATH    │ open-web-math/open-web-math       │ data/               │
# │ ❌ INFIWEBMATH_3+ │ OpenCoder-LLM/opc-fineweb-...     │ infiwebmath-3plus/  │
# │ ❌ INFIWEBMATH_4+ │ OpenCoder-LLM/opc-fineweb-...     │ infiwebmath-4plus/  │
# │ ❌ MEGAMATH_WEB   │ LLM360/MegaMath                   │ megamath-web/       │
# │ ❌ MEGAMATH_PRO   │ LLM360/MegaMath                   │ megamath-web-pro/   │
# │                                                                             │
# │ → Has URL (and cc-path for MegaMath) but NO offset/length                   │
# │ → Run run_cc_index_lookup.py first to get WARC locations from CC Index      │
# │ → Then use --fetch-cc to fetch specific records                             │
# └─────────────────────────────────────────────────────────────────────────────┘

DATASETS = {
    # =========================================================================
    # READY FOR DIRECT FETCH (have full WARC metadata)
    # =========================================================================
    "FINEMATH_4PLUS": {
        # HuggingFace: hf://HuggingFaceTB/finemath/finemath-4plus
        "input": "/lustre/fsw/portfolios/llmservice/users/rkarimimahab/data/finemath/original-data/finemath-4plus/*.parquet",
        "output": "finemath_4plus_processed",
        "fetch_cc": True,
        "needs_cc_lookup": False,  # Already has WARC metadata
        "columns": {
            "warc_filename": "warc_filename",
            "offset": "warc_record_offset",
            "length": "warc_record_length",
        },
    },
    "FINEMATH_3PLUS": {
        # HuggingFace: hf://HuggingFaceTB/finemath/finemath-3plus
        "input": "/lustre/fsw/portfolios/llmservice/users/rkarimimahab/data/finemath/original-data/finemath-3plus/*.parquet",
        "output": "finemath_3plus_processed",
        "fetch_cc": True,
        "needs_cc_lookup": False,  # Already has WARC metadata
        "columns": {
            "warc_filename": "warc_filename",
            "offset": "warc_record_offset",
            "length": "warc_record_length",
        },
    },
    # =========================================================================
    # REQUIRE CC INDEX LOOKUP FIRST (only have URL)
    # Both fetch_cc=True and needs_cc_lookup=True → script automatically:
    #   1. Runs CC Index lookup (requires --crawl-id)
    #   2. Then fetches from CC using enriched WARC metadata
    # =========================================================================
    "OPENWEBMATH": {
        # HuggingFace: hf://open-web-math/open-web-math
        "input": "/home/sasatheesh/data/20t/jsonls/nv-math/open-web-math/*.parquet",
        "output": "openwebmath_processed",
        "fetch_cc": True,  # Will fetch from CC after lookup
        "needs_cc_lookup": True,  # Needs CC Index lookup first (automatic)
        "url_col": "url",  # Column containing the URL for CC lookup
        "columns": {
            # After CC lookup, these columns will be added:
            "warc_filename": "warc_filename",
            "offset": "warc_record_offset",
            "length": "warc_record_length",
        },
    },
    "INFIWEBMATH_4PLUS": {
        # HuggingFace: hf://OpenCoder-LLM/opc-fineweb-math-corpus/infiwebmath-4plus
        "input": "/lustre/fsw/portfolios/llmservice/users/rkarimimahab/data/finemath/original-data/infiwebmath-4plus/*.parquet",
        "output": "infiwebmath_4plus_processed",
        "fetch_cc": True,  # Will fetch from CC after lookup
        "needs_cc_lookup": True,  # Needs CC Index lookup first (automatic)
        "url_col": "url",
        "columns": {
            "warc_filename": "warc_filename",
            "offset": "warc_record_offset",
            "length": "warc_record_length",
        },
    },
    "INFIWEBMATH_3PLUS": {
        # HuggingFace: hf://OpenCoder-LLM/opc-fineweb-math-corpus/infiwebmath-3plus
        "input": "/lustre/fsw/portfolios/llmservice/users/rkarimimahab/data/finemath/original-data/infiwebmath-3plus/*.parquet",
        "output": "infiwebmath_3plus_processed",
        "fetch_cc": True,  # Will fetch from CC after lookup
        "needs_cc_lookup": True,  # Needs CC Index lookup first (automatic)
        "url_col": "url",
        "columns": {
            "warc_filename": "warc_filename",
            "offset": "warc_record_offset",
            "length": "warc_record_length",
        },
    },
    "MEGAMATH_PRO": {
        # HuggingFace: hf://LLM360/MegaMath/megamath-web-pro
        "input": "/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/adlr-stem/megamath_dataset/megamath-web-pro/*.parquet",
        "output": "megamath_pro_processed",
        "fetch_cc": True,  # Will fetch from CC after lookup
        "needs_cc_lookup": True,  # Needs CC Index lookup first (automatic)
        "url_col": "url",
        "columns": {
            "warc_filename": "warc_filename",
            "offset": "warc_record_offset",
            "length": "warc_record_length",
        },
    },
    "MEGAMATH_WEB": {
        # HuggingFace: hf://LLM360/MegaMath/megamath-web
        "input": "/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/adlr-stem/megamath_dataset/megamath-web/**/*.parquet",
        "output": "megamath_web_processed",
        "fetch_cc": True,  # Will fetch from CC after lookup
        "needs_cc_lookup": True,  # Needs CC Index lookup first (automatic)
        "url_col": "url",
        "columns": {
            "warc_filename": "warc_filename",
            "offset": "warc_record_offset",
            "length": "warc_record_length",
        },
    },
}


def run_cc_index_lookup(
    name: str,
    config: dict,
    base_output_dir: str,
    crawl_id: str,
    continue_on_error: bool = False,
) -> str | None:
    """
    Run CC Index lookup to enrich dataset with WARC metadata.

    Returns the path to the enriched output directory.
    """
    script_path = Path(__file__).parent / "run_cc_index_lookup.py"
    enriched_output = Path(base_output_dir) / f"{config['output']}_enriched"

    logger.info(f"Running CC Index lookup for: {name}")
    logger.info(f"Input: {config['input']}")
    logger.info(f"Enriched output: {enriched_output}")

    cmd = [
        sys.executable,
        str(script_path),
        "--input", config["input"],
        "--output", str(enriched_output),
        "--url-col", config.get("url_col", "url"),
        "--crawl-id", crawl_id,
        "--drop-missing",  # Drop rows where lookup fails
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
    crawl_id: str | None = None,
    continue_on_error: bool = False,
) -> None:
    """
    Run the full pipeline for a dataset.

    Automatic sequencing logic:
    - If needs_cc_lookup=True AND fetch_cc=True AND crawl_id provided:
        1. Run CC Index lookup to get WARC metadata
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
        if not crawl_id:
            logger.error(
                f"Dataset {name} requires CC Index lookup but no --crawl-id provided. "
                "Either provide --crawl-id or set fetch_cc=False to process existing data only."
            )
            if not continue_on_error:
                raise ValueError(f"Missing --crawl-id for {name}")
            return

        # Step 1: Run CC Index lookup automatically
        logger.info(f"Dataset {name} needs CC Index lookup. Running lookup first...")
        enriched_path = run_cc_index_lookup(
            name, config, base_output_dir, crawl_id, continue_on_error
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
        description="Run preprocessing for all configured math datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FineMath - has WARC metadata, fetches directly from CC
  python run_all_preprocess.py --output-base /output --datasets FINEMATH_4PLUS

  # OpenWebMath - needs CC lookup, provide crawl-id (automatically runs lookup + fetch)
  python run_all_preprocess.py --output-base /output --datasets OPENWEBMATH \\
      --crawl-id CC-MAIN-2024-10

  # Process all datasets that are ready (have WARC metadata, no lookup needed)
  python run_all_preprocess.py --output-base /output --ready-only

  # Process all datasets with CC lookup (requires --crawl-id)
  python run_all_preprocess.py --output-base /output --crawl-id CC-MAIN-2024-10

  # Process multiple specific datasets
  python run_all_preprocess.py --output-base /output \\
      --datasets FINEMATH_4PLUS OPENWEBMATH INFIWEBMATH_4PLUS \\
      --crawl-id CC-MAIN-2024-10

  # Continue processing even if one dataset fails
  python run_all_preprocess.py --output-base /output \\
      --crawl-id CC-MAIN-2024-10 \\
      --continue-on-error

Dataset Types:
  - FINEMATH_3PLUS, FINEMATH_4PLUS: Have full WARC metadata (can fetch directly)
  - OPENWEBMATH, INFIWEBMATH_3PLUS, INFIWEBMATH_4PLUS, MEGAMATH_WEB, MEGAMATH_PRO:
    Need CC Index lookup (provide --crawl-id to enable)
        """,
    )
    parser.add_argument(
        "--output-base",
        required=True,
        help="Base directory for all outputs",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to run (default: all)",
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "--crawl-id",
        help="Common Crawl crawl ID for CC Index lookup (e.g., 'CC-MAIN-2024-10'). "
        "Required for datasets that need CC lookup.",
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

    # Determine which datasets to process
    if args.datasets:
        targets = args.datasets
    elif args.ready_only:
        targets = [name for name, cfg in DATASETS.items() if not cfg.get("needs_cc_lookup")]
        logger.info(f"Processing only datasets with full WARC metadata: {targets}")
    else:
        targets = list(DATASETS.keys())

    # Process each dataset
    for name in targets:
        if name not in DATASETS:
            logger.warning(f"Dataset {name} not found in configuration. Skipping.")
            continue

        config = DATASETS[name]

        run_dataset(
            name,
            config,
            args.output_base,
            crawl_id=args.crawl_id,
            continue_on_error=args.continue_on_error,
        )


if __name__ == "__main__":
    main()
