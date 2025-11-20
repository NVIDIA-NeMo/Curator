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

import os
import subprocess
from urllib.parse import urlparse

import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch
from nemo_curator.stages.text.download import DocumentDownloader


def _check_s5cmd_installed() -> bool:
    """Check if s5cmd is installed."""
    try:
        subprocess.run(["s5cmd", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)  # noqa: S603, S607
        return True
    except FileNotFoundError:
        return False


class CommonCrawlWARCDownloader(DocumentDownloader):
    """
    Downloads WARC files from the Common Crawl to a local directory
    """

    def __init__(self, download_dir: str, use_aws_to_download: bool = False, verbose: bool = False):
        """
        Creates a downloader

        Args:
          download_dir: Path to store raw compressed WARC files
          use_aws_to_download: If True, uses the s5cmd command to download from the Common Crawl's S3 bucket.
            If False, uses wget.
          verbose: If True, logs stdout and stderr of the download command (s5cmd/wget)
        """
        super().__init__(download_dir, verbose)
        self.use_aws_to_download = use_aws_to_download
        if self.use_aws_to_download and not self._check_s5cmd_installed():
            msg = "s5cmd is not installed. Please install it from https://github.com/peak/s5cmd"
            raise RuntimeError(msg)

    def _get_output_filename(self, url: str) -> str:
        """Generate output filename from URL."""
        return urlparse(url).path[1:].replace("/", "-")

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        """Download a file to a temporary file.

        Args:
            url: URL to download
            path: Local path to save file

        Returns:
            Tuple of (success, error_message). If success is True, error_message is None.
            If success is False, error_message contains the error details.
        """
        urlpath = urlparse(url).path[1:]

        url_to_download = os.path.join("s3://commoncrawl/", urlpath) if self.use_aws_to_download else url

        if self._verbose:
            logger.info(f"Downloading {url_to_download} to {path}")

        # Download with either wget or s5cmd (aws) to temporary file
        if self.use_aws_to_download:
            cmd = ["s5cmd", "cp", url_to_download, path]
        else:
            # We don't use -c (for continue resume) because we want to download file to temp path using -O
            # but -c and -O don't work well together
            cmd = ["wget", url_to_download, "-O", path]

        # Always capture stderr so we can provide meaningful error messages
        if self._verbose:
            stdout, stderr = None, None
        else:
            stdout, stderr = subprocess.DEVNULL, subprocess.PIPE

        result = subprocess.run(  # noqa: S603, PLW1510
            cmd,
            stdout=stdout,
            stderr=stderr,
        )

        if result.returncode == 0:
            return True, None
        else:
            error_msg = result.stderr.decode("utf-8") if result.stderr else "Unknown error"
            return False, error_msg


class CommonCrawlWarcReader(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Reads WARC records directly from Common Crawl S3 bucket using offset and length metadata.
    Uses s5cmd via subprocess to fetch specific byte ranges.
    """

    def __init__(
        self,
        warc_filename_col: str = "warc_filename",
        warc_record_offset_col: str = "warc_record_offset",
        warc_record_length_col: str = "warc_record_length",
        binary_content_col: str = "binary_content",
        s3_bucket: str = "commoncrawl",
    ):
        self.warc_filename_col = warc_filename_col
        self.warc_record_offset_col = warc_record_offset_col
        self.warc_record_length_col = warc_record_length_col
        self.binary_content_col = binary_content_col
        self.s3_bucket = s3_bucket
        self._name = "CommonCrawlWarcReader"
        if not _check_s5cmd_installed():
            msg = "s5cmd is not installed. Please install it from https://github.com/peak/s5cmd"
            raise RuntimeError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return (
            ["data"],
            [self.warc_filename_col, self.warc_record_offset_col, self.warc_record_length_col],
        )

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.binary_content_col]

    def _read_warc_record(self, row) -> bytes | None:
        try:
            filename = row[self.warc_filename_col]
            offset = int(row[self.warc_record_offset_col])
            length = int(row[self.warc_record_length_col])
            
            # s5cmd cat s3://bucket/key --offset 123 --length 456
            # Note: --length is distinct from --offset in s5cmd (byte count, not end byte)
            
            s3_uri = f"s3://{self.s3_bucket}/{filename}"
            
            # s5cmd cat supports --offset and --count (length)
            # Use --no-sign-request for public bucket if needed, but s5cmd usually handles this via config or args
            # Common Crawl is public, s5cmd usually requires --no-sign-request or valid empty creds for public buckets if not configured
            # We will assume standard s5cmd setup or add --no-sign-request if targeting public commoncrawl
            
            cmd = ["s5cmd", "--no-sign-request", "cat", "--offset", str(offset), "--count", str(length), s3_uri]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                # Only log warning on failure
                logger.warning(f"Failed to fetch WARC record {filename} at {offset}: {result.stderr.decode('utf-8')}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to fetch WARC record {filename} at {offset}: {e}")
            return None

    def _read_warc_records_batch(self, df_partition: pd.DataFrame) -> list[bytes | None]:
        """Fetch multiple records efficiently using batch s5cmd calls."""
        # Generate a temporary input file for s5cmd
        # Format: cat --offset OFFSET --count LENGTH s3://BUCKET/KEY
        
        # Since s5cmd batch input file doesn't easily support mapping output to specific rows 
        # without complex parsing (it writes to stdout), for robustness and simplicity within Ray,
        # we will use a ThreadPoolExecutor to run the subprocess calls in parallel within this partition.
        # This avoids the sequential bottleneck of df.apply while keeping logic simple.
        
        import concurrent.futures
        
        results = [None] * len(df_partition)
        rows = list(df_partition.iterrows())
        
        def fetch_row(row_data):
            idx, row = row_data
            return idx, self._read_warc_record(row)

        # Use a thread pool to parallelize the subprocess calls
        # s5cmd is IO bound/subprocess bound, so threads work well here
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            # Submit all tasks
            futures = [executor.submit(fetch_row, (i, row)) for i, (_, row) in enumerate(rows)]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                i, result = future.result()
                results[i] = result
                
        return results

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        
        if self.warc_filename_col in df.columns:
            # Use batched/parallel processing for the partition
            df[self.binary_content_col] = self._read_warc_records_batch(df)
            
        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
