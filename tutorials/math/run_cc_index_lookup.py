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
(warc_filename, offset, length) by querying the Common Crawl CDX API.

How it works:
    1. User provides a Common Crawl crawl ID (e.g., CC-MAIN-2024-10)
    2. For each URL in the dataset, we query that crawl's index
    3. If the URL exists in that crawl → we get WARC metadata (filename, offset, length)
    4. If the URL doesn't exist in that crawl → row is dropped (with --drop-missing)

    This means: only URLs that were captured in the specified crawl will be kept.
    The output is a subset of URLs that can be fetched from that specific crawl.

Usage:
    python run_cc_index_lookup.py \\
        --input /path/to/dataset/*.parquet \\
        --output /path/to/enriched_output \\
        --url-col url \\
        --crawl-id CC-MAIN-2024-10 \\
        --drop-missing

Choosing a crawl ID:
    - Available crawl IDs: https://index.commoncrawl.org/
    - Recent crawls have more coverage of current web pages
    - Older crawls may have pages that no longer exist
    - You may need to try multiple crawls to maximize URL coverage

The output will contain the original data plus:
    - warc_filename: Path to the WARC file in Common Crawl
    - warc_record_offset: Byte offset of the record
    - warc_record_length: Length of the record in bytes

This enriched dataset can then be used with run_text_preprocess.py --fetch-cc

Note: This uses Common Crawl's CDX API (index.commoncrawl.org), not pywb's CDX server API.
      Not all URLs will be found in a given crawl - coverage depends on the crawl.
"""

import argparse
import concurrent.futures
import time
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

# Common Crawl CDX API base URL
# Common Crawl organizes indexes by crawl ID. Each crawl has its own endpoint:
# Format: https://index.commoncrawl.org/{crawl-id}-index?url={url}&output=json
# Example: https://index.commoncrawl.org/CC-MAIN-2024-10-index?url=example.com&output=json
# Available crawl IDs: https://index.commoncrawl.org/
CC_CDX_API_BASE = "https://index.commoncrawl.org"

# Rate limiting settings
DEFAULT_RATE_LIMIT_DELAY = 0.1  # seconds between requests
DEFAULT_MAX_WORKERS = 8  # parallel requests (be respectful to CC servers)
DEFAULT_TIMEOUT = 30  # seconds


def query_cdx_api(
    url: str,
    crawl_id: str,
    session: requests.Session,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict | None:
    """
    Query the Common Crawl CDX API for a single URL.

    Args:
        url: The URL to look up.
        crawl_id: The Common Crawl crawl ID (e.g., "CC-MAIN-2024-10").
                  Required because Common Crawl organizes indexes by crawl ID.
                  Each crawl has its own endpoint: /{crawl-id}-index
        session: Requests session for connection pooling.
        timeout: Request timeout in seconds.

    Returns:
        Dictionary with warc_filename, offset, length, or None if not found.
    """
    try:
        api_url = f"{CC_CDX_API_BASE}/{crawl_id}-index"
        params = {
            "url": url,
            "output": "json",
            "limit": "1",  # We only need the first/best match
        }

        response = session.get(api_url, params=params, timeout=timeout)

        if response.status_code == 200:
            # CDX API returns newline-delimited JSON
            lines = response.text.strip().split("\n")
            if lines and lines[0]:
                import json

                record = json.loads(lines[0])
                return {
                    "warc_filename": record.get("filename"),
                    "warc_record_offset": int(record.get("offset", 0)),
                    "warc_record_length": int(record.get("length", 0)),
                    "cdx_timestamp": record.get("timestamp"),
                    "cdx_status": record.get("status"),
                    "cdx_mime": record.get("mime"),
                }
        elif response.status_code == 404:
            # URL not found in this crawl
            return None
        else:
            logger.warning(f"CDX API returned {response.status_code} for {url}")
            return None

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout querying CDX API for {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error querying CDX API for {url}: {e}")
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Unexpected error querying CDX API for {url}: {e}")
        return None


def process_batch(
    df: pd.DataFrame,
    url_col: str,
    crawl_id: str,
    max_workers: int,
    rate_limit_delay: float,
) -> pd.DataFrame:
    """
    Process a batch of URLs, querying the CDX API for each.

    Args:
        df: Input DataFrame with URLs.
        url_col: Name of the URL column.
        crawl_id: Common Crawl crawl ID.
        max_workers: Number of parallel workers.
        rate_limit_delay: Delay between requests (per worker).

    Returns:
        DataFrame with added WARC metadata columns.
    """
    results = [None] * len(df)
    urls = df[url_col].tolist()

    # Create a session for connection pooling
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers * 2,
    )
    session.mount("https://", adapter)

    def fetch_with_delay(idx_url):
        idx, url = idx_url
        result = query_cdx_api(url, crawl_id, session)
        time.sleep(rate_limit_delay)  # Rate limiting
        return idx, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_with_delay, (i, url)) for i, url in enumerate(urls)]

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
                completed += 1
                if completed % 100 == 0:
                    logger.info(f"Processed {completed}/{len(urls)} URLs")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error in thread: {e}")

    session.close()

    # Add results to DataFrame
    df = df.copy()
    df["warc_filename"] = [r.get("warc_filename") if r else None for r in results]
    df["warc_record_offset"] = [r.get("warc_record_offset") if r else None for r in results]
    df["warc_record_length"] = [r.get("warc_record_length") if r else None for r in results]
    df["cdx_timestamp"] = [r.get("cdx_timestamp") if r else None for r in results]
    df["cdx_status"] = [r.get("cdx_status") if r else None for r in results]
    df["cdx_mime"] = [r.get("cdx_mime") if r else None for r in results]

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich a dataset with WARC metadata from Common Crawl CDX API. "
        "Only URLs that exist in the specified crawl will have WARC metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Look up URLs in CC-MAIN-2024-10, drop URLs not found in that crawl
  python run_cc_index_lookup.py \\
      --input /path/to/openwebmath/*.parquet \\
      --output /path/to/enriched \\
      --crawl-id CC-MAIN-2024-10 \\
      --drop-missing

  # Keep all rows (URLs not found will have null WARC metadata)
  python run_cc_index_lookup.py \\
      --input /path/to/data/*.parquet \\
      --output /path/to/enriched \\
      --crawl-id CC-MAIN-2023-50

Note: Not all URLs will be found in a given crawl. The output is a subset
of URLs that were captured during that specific Common Crawl snapshot.
Choose a crawl ID from: https://index.commoncrawl.org/
        """,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input Parquet file(s) glob pattern (e.g., '/path/to/data/*.parquet')",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for enriched Parquet files",
    )
    parser.add_argument(
        "--url-col",
        default="url",
        help="Name of the URL column in the input data (default: 'url')",
    )
    parser.add_argument(
        "--crawl-id",
        required=True,
        help="Common Crawl crawl ID (e.g., 'CC-MAIN-2024-10'). "
        "Only URLs captured in this crawl will have WARC metadata. "
        "See https://index.commoncrawl.org/ for available crawls.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=DEFAULT_RATE_LIMIT_DELAY,
        help=f"Delay between requests in seconds (default: {DEFAULT_RATE_LIMIT_DELAY})",
    )
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="Drop rows where URL was not found in the specified crawl (recommended)",
    )

    args = parser.parse_args()

    # Read input files
    import glob

    input_files = glob.glob(args.input)
    if not input_files:
        logger.error(f"No files found matching: {args.input}")
        return

    logger.info(f"Found {len(input_files)} input files")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    enriched_rows = 0

    for input_file in input_files:
        logger.info(f"Processing: {input_file}")

        # Read the input file
        df = pd.read_parquet(input_file)
        total_rows += len(df)

        if args.url_col not in df.columns:
            logger.error(f"Column '{args.url_col}' not found in {input_file}")
            continue

        # Process the batch
        df_enriched = process_batch(
            df,
            url_col=args.url_col,
            crawl_id=args.crawl_id,
            max_workers=args.max_workers,
            rate_limit_delay=args.rate_limit,
        )

        # Optionally drop rows without WARC metadata
        if args.drop_missing:
            before_count = len(df_enriched)
            df_enriched = df_enriched.dropna(subset=["warc_filename"])
            dropped = before_count - len(df_enriched)
            if dropped > 0:
                logger.info(f"Dropped {dropped} rows without WARC metadata")

        enriched_rows += len(df_enriched[df_enriched["warc_filename"].notna()])

        # Write output
        output_file = output_dir / Path(input_file).name
        df_enriched.to_parquet(output_file, index=False)
        logger.info(f"Wrote enriched data to: {output_file}")

    # Summary
    logger.info("=" * 60)
    logger.info(f"Total rows processed: {total_rows}")
    logger.info(f"Rows with WARC metadata: {enriched_rows}")
    logger.info(f"Success rate: {enriched_rows / total_rows * 100:.1f}%")
    logger.info("=" * 60)
    logger.info(
        f"Next step: Run 'python run_text_preprocess.py --input {args.output}/*.parquet --fetch-cc ...'"
    )


if __name__ == "__main__":
    main()

