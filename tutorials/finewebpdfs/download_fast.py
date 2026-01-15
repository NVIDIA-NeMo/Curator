"""
Tutorial: Fast PDF download from Common Crawl using the FinePDFs index.

Pipeline steps:
- URL generation from the HuggingFaceFW/finepdfs dataset. Each entry contains a WARC `file_path` and an `offset`.
- Direct download and PDF extraction: Use fsspec to open the WARC file (S3 or HTTPS), seek to offset,
  and extract PDF bytes without downloading the entire WARC file.

This is faster than the three-stage pipeline in download.py because it:
1. Avoids downloading the full WARC file to disk
2. Only reads the necessary bytes starting from the offset
3. Reduces I/O and storage requirements
"""

import argparse
import os
import pickle
from dataclasses import dataclass
from urllib.parse import urlparse

import fsspec
import pandas as pd
from fastwarc.warc import ArchiveIterator
from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage, URLGenerator
from nemo_curator.tasks import FileGroupTask


def generate_output_filename(file_path: str, offset: int) -> str:
    parsed_path = urlparse(file_path).path
    parsed_path = parsed_path.removeprefix("/")
    path_part = parsed_path.replace("/", "-")
    return f"{path_part}-startpos-{offset}_{offset}.pdf"


class FinePDFsURLGenerator(URLGenerator):
    """Generate S3 or HTTPS URLs with embedded offset from the FinePDFs dataset.

    Each generated URL includes a URL fragment `#offset=...` to propagate the byte offset
    through subsequent stages without changing stage interfaces.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def generate_urls(self) -> list[str]:
        dataset = pd.read_json("/raid/praateekm/NeMo-Curator/finepdfs.jsonl", lines=True)[:100_000]
        num_rows = len(dataset)
        # Remove rows with .parquet or /cc-index/table/
        dataset = dataset[
            ~dataset["file_path"].str.endswith(".parquet") & ~dataset["file_path"].str.contains("/cc-index/table/")
        ]
        num_rows_after_filtering = len(dataset)
        logger.debug(
            f"Filtered from {num_rows:,} rows to {num_rows_after_filtering:,} rows because they are parquet index files or cc-index table files"
        )

        urls: list[str] = []
        skipped_existing_files_count = 0

        for _, row in dataset.iterrows():
            file_path: str = row["file_path"]
            offset_val: int = int(row["offset"])  # ensure int for naming

            # Compute expected PDF output path to skip existing files
            expected_pdf_name = generate_output_filename(file_path, offset_val)
            expected_pdf_path = os.path.join(self.output_dir, expected_pdf_name)

            if os.path.exists(expected_pdf_path):
                skipped_existing_files_count += 1
                continue

            # Encode offset in URL fragment so it flows in task metadata as `source_url`
            url_with_offset = f"{file_path}#offset={offset_val}"
            urls.append(url_with_offset)

        urls_after_deduplication = list(dict.fromkeys(urls))
        logger.info(
            f"Generated {len(urls_after_deduplication):,} URLs to process (skipped {skipped_existing_files_count:,} existing files)"
        )
        return urls_after_deduplication


@dataclass
class DirectPDFDownloadStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Stage that directly downloads PDFs from WARC files using fsspec.

    This stage:
    1. Opens the WARC file using fsspec (supports S3, HTTPS, etc.)
    2. Seeks to the offset specified in the URL fragment
    3. Extracts the PDF bytes from the WARC record
    4. Saves the PDF to disk

    Input:  FileGroupTask with WARC URL(s) in `data`, and `source_url` metadata
            containing a `#offset=...` URL fragment.
    Output: FileGroupTask with saved PDF file path(s) in `data`.
    """

    output_dir: str
    use_aws: bool = False
    verbose: bool = False
    _resources = Resources(cpus=1)

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self._name = "direct_pdf_download"

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [])

    def _extract_offset(self, url: str) -> int:
        """Extract offset from URL fragment."""
        parsed = urlparse(url)
        if parsed.fragment:
            try:
                key, value = parsed.fragment.split("=", 1)
                if key == "offset":
                    return int(value)
            except Exception as parsing_error:  # noqa: BLE001
                logger.debug(f"Failed to parse offset from fragment: {parsing_error!s}")

        msg = f"Offset not found in URL fragment: {url}"
        raise ValueError(msg)

    def process(self, task: FileGroupTask) -> FileGroupTask:
        """Process a task by downloading PDFs directly from WARC files."""
        saved_files: list[str] = []
        total_urls = len(task.data)
        success_count = 0
        fail_count = 0
        skipped_count = 0

        for idx, url in enumerate(task.data, 1):
            try:
                clean_url, offset = url.split("#offset=")
                offset = int(offset)
                output_filename = generate_output_filename(clean_url, offset)
                output_path = os.path.join(self.output_dir, output_filename)

                # Skip if file already exists (for retries)
                if os.path.exists(output_path):
                    skipped_count += 1
                    logger.info(f"Skipped file because it exists: {output_path} ({skipped_count:,}/{total_urls:,})")
                    continue

                if self.use_aws:
                    fs = fsspec.filesystem("s3", anon=False, client_kwargs={"region_name": "us-east-1"})
                else:
                    clean_url = clean_url.replace("s3://commoncrawl", "https://data.commoncrawl.org")
                    fs = fsspec.filesystem("https", headers={"User-Agent": "Mozilla/5.0 (compatible;)"})

                with fs.open(clean_url, "rb") as f:
                    f.seek(offset)

                    found_record = False
                    for record in ArchiveIterator(f):
                        pdf_bytes = record.reader.read()
                        if len(pdf_bytes) == 0:
                            logger.warning(f"[{idx}/{total_urls}] Empty PDF bytes at offset {offset} in {url}")
                            continue

                        with open(output_path, "wb") as pdf_f:
                            pdf_f.write(pdf_bytes)

                        saved_files.append(output_path)
                        success_count += 1

                        logger.info(f"[{idx}/{total_urls}] Saved {len(pdf_bytes):,} bytes to {output_path}")

                        found_record = True
                        # Only process the first response record at this offset
                        break

                    if not found_record:
                        fail_count += 1
                        logger.warning(f"[{idx}/{total_urls}] No response record found at offset {offset} in {url}")

            except Exception as e:  # noqa: BLE001
                fail_count += 1
                logger.error(
                    f"[{idx}/{total_urls}] Failed to download PDF from {url} at offset {offset} due to {type(e).__name__}"
                )
                continue

        # Log summary for this task
        if fail_count > 0:
            logger.error(f"Task {task.task_id}: {fail_count}/{total_urls} PDFs downloaded failed")
        else:
            logger.info(f"Task {task.task_id}: {success_count}/{total_urls} PDFs downloaded successfully")

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=saved_files or task.data,
            _metadata={
                **task._metadata,
                "saved_pdf_files": saved_files,
                "success_count": success_count,
                "fail_count": fail_count,
                "skipped_count": skipped_count,
            },
            _stage_perf=task._stage_perf,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast FinePDFs Common Crawl PDF downloader tutorial")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store extracted PDF files")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of URLs to process")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for downloads")
    parser.add_argument("--num-cpus", type=int, default=128, help="Number of CPUs for Ray")
    parser.add_argument("--use-aws", action="store_true", help="Use AWS for downloads")
    args = parser.parse_args()

    # Expand paths to handle ~/ and relative paths
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    # Initialize Ray and pipeline
    logger.info(f"Initializing Ray with {args.num_cpus} CPUs")
    ray_client = RayClient(num_cpus=args.num_cpus, num_gpus=0, ray_dashboard_host="0.0.0.0")  # noqa: S104
    ray_client.start()

    pipeline = Pipeline(
        name="finepdfs_fast_download",
        description="Fast PDF download from FinePDFs Common Crawl index using direct fsspec access",
    )

    # 1) URL generation
    url_generator = FinePDFsURLGenerator(output_dir=output_dir)
    pipeline.add_stage(URLGenerationStage(url_generator=url_generator, limit=args.limit))

    # 2) Direct PDF download and save (combined stage)
    pipeline.add_stage(DirectPDFDownloadStage(output_dir=output_dir, use_aws=args.use_aws, verbose=args.verbose))

    # Execute with Xenna executor
    logger.info("Starting pipeline execution")
    executor = XennaExecutor()
    results = pipeline.run(executor)

    # Save results
    with open("results_fast.pkl", "wb") as f:
        pickle.dump(results, f)

    total_saved = sum(len(task._metadata["saved_pdf_files"]) for task in (results or []))
    total_skipped = sum(task._metadata["skipped_count"] for task in (results or []))
    total_failed = sum(task._metadata["fail_count"] for task in (results or []))
    logger.info(
        f"Successfully saved {total_saved} PDF files to {output_dir} (skipped {total_skipped:,} files, failed {total_failed:,} files)"
    )

    ray_client.stop()


if __name__ == "__main__":
    main()
