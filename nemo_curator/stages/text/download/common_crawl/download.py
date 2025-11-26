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

import concurrent.futures
import os
import subprocess
from urllib.parse import urlparse

import pandas as pd
import requests
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.download import DocumentDownloader
from nemo_curator.tasks import DocumentBatch

# Common Crawl base URL for HTTPS access
CC_BASE_URL = "https://data.commoncrawl.org/"


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


class CommonCrawlWARCReader(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Reads WARC records directly from Common Crawl using HTTPS range requests.

    This stage fetches raw HTML content from Common Crawl's public servers
    using byte-range requests. No AWS credentials or s5cmd required.
    """

    def __init__(
        self,
        warc_filename_col: str = "warc_filename",
        warc_record_offset_col: str = "warc_record_offset",
        warc_record_length_col: str = "warc_record_length",
        binary_content_col: str = "binary_content",
        drop_failed: bool = True,
        max_workers: int = 16,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize the WARC reader.

        Args:
            warc_filename_col: Column name for WARC filename.
            warc_record_offset_col: Column name for byte offset.
            warc_record_length_col: Column name for record length.
            binary_content_col: Output column name for fetched content.
            drop_failed: If True, drop rows where fetch failed.
            max_workers: Number of parallel threads for fetching.
            timeout: HTTP request timeout in seconds.
            max_retries: Number of retries for failed requests.
        """
        self.warc_filename_col = warc_filename_col
        self.warc_record_offset_col = warc_record_offset_col
        self.warc_record_length_col = warc_record_length_col
        self.binary_content_col = binary_content_col
        self.drop_failed = drop_failed
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        self._name = "CommonCrawlWARCReader"
        self._session = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return (
            ["data"],
            [self.warc_filename_col, self.warc_record_offset_col, self.warc_record_length_col],
        )

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.binary_content_col]

    def _get_session(self) -> requests.Session:
        """Get or create a requests session for connection pooling."""
        if self._session is None:
            self._session = requests.Session()
            # Configure connection pooling for better performance
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=self.max_workers,
                pool_maxsize=self.max_workers * 2,
                max_retries=self.max_retries,
            )
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)
        return self._session

    def _read_warc_record(self, row) -> bytes | None:
        """Fetch a single WARC record using HTTPS range request."""
        try:
            filename = row[self.warc_filename_col]
            offset = int(row[self.warc_record_offset_col])
            length = int(row[self.warc_record_length_col])

            # Build the URL
            url = f"{CC_BASE_URL}{filename}"

            # HTTP Range header (inclusive end byte)
            end_byte = offset + length - 1
            headers = {"Range": f"bytes={offset}-{end_byte}"}

            response = self._get_session().get(
                url,
                headers=headers,
                timeout=self.timeout,
            )

            # 206 Partial Content is the expected response for range requests
            if response.status_code == 206:
                return response.content
            elif response.status_code == 200:
                # Server ignored range request, returned full file (unusual but handle it)
                logger.warning(f"Server returned full file instead of range for {filename}")
                return response.content[offset : offset + length]
            else:
                logger.warning(f"Failed to fetch WARC record {filename}: HTTP {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching WARC record {filename} at offset {offset}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch WARC record {filename} at offset {offset}: {e}")
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Unexpected error fetching WARC record: {e}")
            return None

    def _read_warc_records_batch(self, df_partition: pd.DataFrame) -> list[bytes | None]:
        """Fetch multiple records in parallel using ThreadPoolExecutor."""
        results = [None] * len(df_partition)
        rows = list(df_partition.iterrows())

        def fetch_row(row_data):
            idx, row = row_data
            return idx, self._read_warc_record(row)

        # Use a thread pool to parallelize the HTTP requests
        # Requests are IO bound, so threads work well here
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(fetch_row, (i, row)) for i, (_, row) in enumerate(rows)]

            for future in concurrent.futures.as_completed(futures):
                try:
                    i, result = future.result()
                    results[i] = result
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Error in thread pool: {e}")

        return results

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        if self.warc_filename_col in df.columns:
            # Use batched/parallel processing for the partition
            df[self.binary_content_col] = self._read_warc_records_batch(df)

            if self.drop_failed:
                # Drop rows where binary_content is None
                initial_count = len(df)
                df = df.dropna(subset=[self.binary_content_col])
                dropped_count = initial_count - len(df)
                if dropped_count > 0:
                    logger.info(f"Dropped {dropped_count}/{initial_count} rows due to failed WARC fetch.")

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
