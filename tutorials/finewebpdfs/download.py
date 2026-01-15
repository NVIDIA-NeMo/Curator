"""
Tutorial: Download PDFs from Common Crawl using the FinePDFs index.

Pipeline steps:
- URL generation from the HuggingFaceFW/finepdfs dataset. Each entry contains a WARC `file_path` and an `offset`.
  We convert `s3://commoncrawl` to `https://data.commoncrawl.org` and embed the offset as a URL fragment
  (e.g., https://data.commoncrawl.org/crawl-data/CC-MAIN-.../segments/.../warc/....warc.gz#offset=123456789).
- Download the WARC files with CommonCrawlWARCStartPosDownloader (wget with --start-pos).
- Save PDF bytes at the given offset as `original_file_name_offset.pdf` into a target directory.
"""

import argparse
import os
import pickle
import subprocess
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import pandas as pd
from loguru import logger
from warcio.archiveiterator import ArchiveIterator

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.base.download import DocumentDownloader, DocumentDownloadStage
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage, URLGenerator
from nemo_curator.tasks import FileGroupTask


class FinePDFsURLGenerator(URLGenerator):
    """Generate Common Crawl HTTPS WARC URLs with embedded offset from the FinePDFs dataset.

    Each generated URL includes a URL fragment `#offset=...` to propagate the byte offset
    through subsequent stages without changing stage interfaces.
    """

    def __init__(self, output_dir: str, streaming: bool = True) -> None:
        self.output_dir = output_dir
        self.streaming = streaming

    def generate_urls(self) -> list[str]:
        dataset = pd.read_json("/raid/praateekm/NeMo-Curator/finepdfs.jsonl", lines=True)
        num_rows = len(dataset)
        # remove rows with .parquet or /cc-index/table/
        dataset = dataset[
            ~dataset["file_path"].str.endswith(".parquet") & ~dataset["file_path"].str.contains("/cc-index/table/")
        ]
        num_rows_after_filtering = len(dataset)
        logger.debug(
            f"Filtered from {num_rows:,} rows to {num_rows_after_filtering:,} rows because they are parquet index files or cc-index table files"
        )

        urls: list[str] = []
        skipped_existing_files_count = 0
        replace_prefix = ("s3://commoncrawl", "https://data.commoncrawl.org")
        for _, row in dataset.iterrows():
            file_path: str = row["file_path"]
            offset_val: int = int(row["offset"])  # ensure int for naming

            # Replace S3 with HTTPS
            if file_path.startswith(replace_prefix[0]):
                file_path = file_path.replace(replace_prefix[0], replace_prefix[1], 1)

            # Compute expected PDF output path to skip existing files
            path_part = urlparse(file_path).path[1:].replace("/", "-")
            expected_pdf_name = f"{path_part}-startpos-{offset_val}_{offset_val}.pdf"
            expected_pdf_path = os.path.join(self.output_dir, expected_pdf_name)
            if os.path.exists(expected_pdf_path):
                skipped_existing_files_count += 1
                logger.debug(
                    f"Skipping {expected_pdf_path} ({skipped_existing_files_count:,} total) because it already exists"
                )
                continue

            # Encode offset in URL fragment so it flows in task metadata as `source_url`
            url_with_offset = f"{file_path}#offset={offset_val}"
            urls.append(url_with_offset)

        return urls


class CommonCrawlWARCStartPosDownloader(DocumentDownloader):
    """Downloader that always uses wget with --start-pos=<offset> and HTTPS URLs.

    Expects the input URL to carry the offset in the URL fragment, e.g.
    https://data.commoncrawl.org/.../warc/....warc.gz#offset=123456789
    """

    def __init__(self, download_dir: str, verbose: bool = False):
        super().__init__(download_dir=download_dir, verbose=verbose)

    def _parse_offset(self, url: str) -> int:
        parsed = urlparse(url)
        if parsed.fragment:
            try:
                key, value = parsed.fragment.split("=", 1)
                if key == "offset":
                    return int(value)
            except Exception as parsing_error:  # noqa: BLE001
                debug_msg = f"Failed to parse offset from fragment: {parsing_error!s}"
                logger.debug(debug_msg)
        if parsed.query:
            q = parse_qs(parsed.query)
            if q.get("offset"):
                return int(q["offset"][0])
        return 0

    def _https_url_without_fragment(self, url: str) -> str:
        parsed = urlparse(url)
        # Normalize to data.commoncrawl.org
        https_base = "https://data.commoncrawl.org"
        return f"{https_base}{parsed.path}"

    def _get_output_filename(self, url: str) -> str:
        # Include start position in filename to signal partial file semantics
        offset = self._parse_offset(url)
        path_part = urlparse(url).path[1:].replace("/", "-")
        return f"{path_part}-startpos-{offset}"

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        offset = self._parse_offset(url)
        https_url = self._https_url_without_fragment(url)

        if self._verbose:
            logger.info(f"Downloading from {https_url} with --start-pos={offset} to {path}")

        cmd = [
            "wget",
            "--quiet",  # Suppress wget progress output
            "--no-verbose",  # No verbose output
            f"--start-pos={offset}",
            https_url,
            "-O",
            path,
        ]

        # Always capture stderr for error messages
        result = subprocess.run(  # noqa: S603, PLW1510
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        if result.returncode == 0:
            return True, None
        else:
            error_msg = result.stderr.decode("utf-8") if result.stderr else "Unknown error"
            return False, error_msg


@dataclass
class SavePDFStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Stage that seeks to a given offset in a WARC file and saves the PDF payload.

    Input:  FileGroupTask with local WARC file path(s) in `data`, and `source_url` metadata
            containing a `#offset=...` URL fragment.
    Output: FileGroupTask with saved PDF file path(s) in `data`.
    """

    output_dir: str
    _resources = Resources(cpus=0.5)

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self._name = "save_pdf_from_warc"

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [])

    def _extract_offset(self, source_url: str) -> int:
        # Parse offset from fragment (supports either #offset=123 or ?offset=123#...)
        parsed = urlparse(source_url)
        # First check fragment form: "offset=123"
        if parsed.fragment:
            try:
                key, value = parsed.fragment.split("=", 1)
                if key == "offset":
                    return int(value)
            except Exception as parsing_error:  # noqa: BLE001
                debug_msg = f"Failed to parse offset from fragment: {parsing_error!s}"
                logger.debug(debug_msg)

        # Fallback: check query
        if parsed.query:
            q = parse_qs(parsed.query)
            if q.get("offset"):
                return int(q["offset"][0])

        msg = "Offset not found in source_url fragment or query"
        raise ValueError(msg)

    def process(self, task: FileGroupTask) -> FileGroupTask:
        source_url = task._metadata.get("source_url")
        if not source_url:
            msg = "Task metadata missing 'source_url'"
            raise ValueError(msg)
        offset = self._extract_offset(source_url)

        saved_files: list[str] = []

        for warc_path in task.data:
            base_name = os.path.basename(warc_path)
            output_pdf = os.path.join(self.output_dir, f"{base_name}_{offset}.pdf")
            try:
                with open(warc_path, "rb") as fp:
                    # If the file name indicates a start-pos partial download, do not seek again
                    if "-startpos-" not in base_name:
                        fp.seek(offset)
                    for record in ArchiveIterator(fp):
                        if record.rec_type == "response":
                            pdf_bytes = record.content_stream().read()
                            with open(output_pdf, "wb") as pdf_f:
                                pdf_f.write(pdf_bytes)
                            saved_files.append(output_pdf)
                            break
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to save PDF from {warc_path} at offset {offset}: {e!s}")
                continue

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=saved_files,
            _metadata={
                **task._metadata,
                "saved_pdf_files": saved_files,
                "offset": offset,
            },
            _stage_perf=task._stage_perf,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="FinePDFs Common Crawl PDF downloader tutorial")
    parser.add_argument("--download-dir", type=str, required=True, help="Directory to store downloaded WARC files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store extracted PDF files")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of URLs to process")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for downloads")
    args = parser.parse_args()

    # Expand paths to handle ~/ and relative paths
    download_dir = os.path.abspath(os.path.expanduser(args.download_dir))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    # Initialize Ray and pipeline
    ray_client = RayClient(num_cpus=128, num_gpus=0, ray_dashboard_host="0.0.0.0")  # noqa: S104
    ray_client.start()
    pipeline = Pipeline(name="finepdfs_download", description="Download PDFs from FinePDFs Common Crawl index")

    # 1) URL generation
    url_generator = FinePDFsURLGenerator(output_dir=output_dir, streaming=True)
    pipeline.add_stage(URLGenerationStage(url_generator=url_generator, limit=args.limit))

    # 2) Download WARC using wget with --start-pos
    downloader = CommonCrawlWARCStartPosDownloader(download_dir=download_dir, verbose=args.verbose)
    pipeline.add_stage(DocumentDownloadStage(downloader=downloader))

    # 3) Save PDF by offset
    pipeline.add_stage(SavePDFStage(output_dir=output_dir))

    # Execute with Xenna executor
    executor = XennaExecutor()
    results = pipeline.run(executor)
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    total_saved = sum(len(task.data) for task in (results or []))
    logger.info(f"Saved {total_saved} PDF files to {output_dir}")
    ray_client.stop()


if __name__ == "__main__":
    main()
