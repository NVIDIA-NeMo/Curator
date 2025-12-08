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
CC Index Lookup Script

This script enriches a dataset that has URLs but lacks WARC metadata
(warc_filename, offset, length) by joining against the Common Crawl Index.

The CC Index is publicly available on S3 and is queried directly.

How it works:
    1. Load all URLs from the input dataset
    2. Stream CC Index parquet files from S3 in batches
    3. Perform an inner join on URL to find matching WARC metadata
    4. Output enriched parquet files ready for run_text_preprocess.py --fetch-cc

Usage:
    # Single crawl
    python run_cc_index_lookup_local.py \\
        --input /path/to/openwebmath/*.parquet \\
        --output /path/to/enriched \\
        --crawls CC-MAIN-2024-10

    # Multiple crawls for better URL coverage
    python run_cc_index_lookup_local.py \\
        --input /path/to/openwebmath/*.parquet \\
        --output /path/to/enriched \\
        --crawls CC-MAIN-2024-10 CC-MAIN-2024-18 CC-MAIN-2023-50

Output columns added:
    - warc_filename: Path to the WARC file in Common Crawl
    - warc_record_offset: Byte offset of the record
    - warc_record_length: Length of the record in bytes
    - content_mime_type: MIME type from CC Index

This enriched dataset can then be used with run_text_preprocess.py --fetch-cc
"""

import argparse
from pathlib import Path

import polars as pl
from loguru import logger

# Public Common Crawl Index location on S3
CC_INDEX_S3_BASE = "s3://commoncrawl/cc-index/table/cc-main/warc"

# Columns we need from CC Index
CC_INDEX_COLS = [
    "url",
    "filename",  # CC Index uses 'filename', we'll rename to warc_filename
    "offset",
    "length",
    "mime",
    "status",
]


def get_s3_storage_options() -> dict:
    """Get storage options for anonymous S3 access."""
    return {
        "aws_region": "us-east-1",
        "aws_access_key_id": "",
        "aws_secret_access_key": "",
    }


def load_dataset_urls(input_glob: str, url_col: str) -> pl.DataFrame:
    """
    Load unique URLs from the input dataset.

    Args:
        input_glob: Glob pattern for input parquet files.
        url_col: Name of the URL column.

    Returns:
        DataFrame with unique URLs.
    """
    logger.info(f"Loading URLs from: {input_glob}")

    df = pl.scan_parquet(input_glob).select(url_col).unique().collect()

    logger.info(f"Loaded {len(df):,} unique URLs from input dataset")
    return df


def join_with_cc_index(
    urls_df: pl.DataFrame,
    crawls: list[str],
    url_col: str,
) -> pl.DataFrame:
    """
    Join dataset URLs with CC Index on S3 to get WARC metadata.

    Streams the CC Index directly from S3 without downloading.

    Args:
        urls_df: DataFrame with URLs to look up.
        crawls: List of crawl IDs to search.
        url_col: Name of the URL column.

    Returns:
        DataFrame with URLs and their WARC metadata.
    """
    logger.info(f"Joining {len(urls_df):,} URLs against CC Index on S3")
    logger.info(f"Crawls to search: {crawls}")

    results = []
    storage_options = get_s3_storage_options()

    for crawl in crawls:
        s3_path = f"{CC_INDEX_S3_BASE}/crawl={crawl}/subset=warc/*.parquet"
        logger.info(f"Querying: {s3_path}")

        try:
            # Scan CC Index from S3 lazily
            cc_index_lazy = pl.scan_parquet(
                s3_path,
                storage_options=storage_options,
            ).select(CC_INDEX_COLS)

            # Perform the join
            matched = (
                cc_index_lazy
                .join(
                    urls_df.lazy(),
                    left_on="url",
                    right_on=url_col,
                    how="inner",
                )
                .collect()
            )

            if len(matched) > 0:
                results.append(matched)
                logger.info(f"  Found {len(matched):,} matches in {crawl}")

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error querying {crawl}: {e}")
            continue

    if not results:
        logger.warning("No matches found in CC Index!")
        return pl.DataFrame()

    # Combine all results and deduplicate
    combined = pl.concat(results)
    deduplicated = combined.unique(subset=["url"], keep="first")

    logger.info(f"Total unique URLs matched: {len(deduplicated):,}")
    return deduplicated


def enrich_dataset(
    input_glob: str,
    enriched_urls: pl.DataFrame,
    url_col: str,
    output_dir: str,
) -> None:
    """
    Enrich the original dataset with WARC metadata and write output.

    Args:
        input_glob: Glob pattern for input parquet files.
        enriched_urls: DataFrame with URLs and WARC metadata.
        url_col: Name of the URL column.
        output_dir: Output directory for enriched parquet files.
    """
    import glob as glob_module

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_files = glob_module.glob(input_glob)
    logger.info(f"Enriching {len(input_files)} input files")

    total_input_rows = 0
    total_output_rows = 0

    # Prepare the enriched URLs for joining
    join_cols = ["url", "filename", "offset", "length", "mime", "status"]
    available_cols = [c for c in join_cols if c in enriched_urls.columns]
    enriched_for_join = enriched_urls.select(available_cols)

    for input_file in input_files:
        original_df = pl.read_parquet(input_file)
        total_input_rows += len(original_df)

        enriched_df = original_df.join(
            enriched_for_join,
            left_on=url_col,
            right_on="url",
            how="inner",
        )

        # Rename columns to match expected format
        rename_map = {
            "filename": "warc_filename",
            "offset": "warc_record_offset",
            "length": "warc_record_length",
            "mime": "content_mime_type",
            "status": "http_status",
        }
        for old_name, new_name in rename_map.items():
            if old_name in enriched_df.columns:
                enriched_df = enriched_df.rename({old_name: new_name})

        total_output_rows += len(enriched_df)

        output_file = output_path / Path(input_file).name
        enriched_df.write_parquet(output_file)
        logger.info(f"Wrote {len(enriched_df):,} rows to {output_file}")

    # Summary
    logger.info("=" * 60)
    logger.info(f"Input rows:  {total_input_rows:,}")
    logger.info(f"Output rows: {total_output_rows:,}")
    if total_input_rows > 0:
        logger.info(f"Match rate:  {total_output_rows / total_input_rows * 100:.1f}%")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich dataset with WARC metadata from Common Crawl Index. "
        "Queries the public CC Index on S3 directly.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single crawl
  python run_cc_index_lookup_local.py \\
      --input /path/to/openwebmath/*.parquet \\
      --output /path/to/enriched \\
      --crawls CC-MAIN-2024-10

  # Multiple crawls for better URL coverage
  python run_cc_index_lookup_local.py \\
      --input /path/to/openwebmath/*.parquet \\
      --output /path/to/enriched \\
      --crawls CC-MAIN-2024-10 CC-MAIN-2024-18 CC-MAIN-2023-50

Available crawl IDs: https://index.commoncrawl.org/

Next step:
  python run_text_preprocess.py --input /path/to/enriched/*.parquet --fetch-cc ...
        """,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input parquet file(s) glob pattern (e.g., '/path/to/data/*.parquet')",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for enriched parquet files",
    )
    parser.add_argument(
        "--crawls",
        nargs="+",
        required=True,
        help="Crawl IDs to search (e.g., CC-MAIN-2024-10). "
        "Multiple crawls can be specified for better URL coverage.",
    )
    parser.add_argument(
        "--url-col",
        default="url",
        help="Name of the URL column in the input data (default: 'url')",
    )

    args = parser.parse_args()

    # Load dataset URLs
    urls_df = load_dataset_urls(args.input, args.url_col)
    if len(urls_df) == 0:
        logger.error("No URLs found in input dataset")
        return

    # Join with CC Index on S3
    enriched_urls = join_with_cc_index(
        urls_df,
        args.crawls,
        args.url_col,
    )

    if len(enriched_urls) == 0:
        logger.error("No URLs matched in CC Index. Try adding more crawls.")
        return

    # Enrich original dataset and write output
    enrich_dataset(args.input, enriched_urls, args.url_col, args.output)

    logger.info(
        f"Next step: Run 'python run_text_preprocess.py --input {args.output}/*.parquet --fetch-cc ...'"
    )


if __name__ == "__main__":
    main()
