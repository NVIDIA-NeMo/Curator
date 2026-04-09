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
import gzip
import io
import os
import subprocess
import threading
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from loguru import logger
from warcio.archiveiterator import ArchiveIterator

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.download import DocumentDownloader
from nemo_curator.stages.text.download.utils import check_s5cmd_installed
from nemo_curator.tasks import DocumentBatch

# Common Crawl base URL for HTTPS access
CC_BASE_URL = "https://data.commoncrawl.org/"

# HTTP status codes
HTTP_OK = 200
HTTP_PARTIAL_CONTENT = 206


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
        if self.use_aws_to_download and not check_s5cmd_installed():
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
            cmd = ["wget", url_to_download, "-O", path, "--retry-on-http-error=503", "--waitretry=5", "--tries=5"]

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
    Reads WARC records directly from Common Crawl using HTTPS or S3 range requests.

    This stage fetches raw HTML content from Common Crawl's public servers
    using byte-range requests.

    Transport modes:
      - **HTTPS** (default): Uses ``data.commoncrawl.org`` via ``requests``.
      - **S3**: Uses ``boto3`` range requests against the ``commoncrawl`` bucket.
        Activated by ``use_s3=True`` or by setting ``CC_USE_S3=1``.
        Credentials, region, and endpoint are resolved
        by boto3's standard credential chain (env vars, ``~/.aws/`` config,
        instance profiles, etc.).
    """

    def __init__(  # noqa: PLR0913
        self,
        warc_filename_col: str = "warc_filename",
        warc_record_offset_col: str = "warc_record_offset",
        warc_record_length_col: str = "warc_record_length",
        binary_content_col: str = "binary_content",
        drop_failed: bool = True,
        max_workers: int = 16,
        timeout: int = 30,
        max_retries: int = 3,
        use_s3: bool | None = None,
        s3_bucket: str | None = None,
        s3_key_prefix: str | None = None,
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
            use_s3: If True, fetch via S3 (boto3) instead of HTTPS.
                If None (default), reads ``CC_USE_S3`` env var.
                Accepted truthy values: ``1``, ``true``, ``yes``.
            s3_bucket: S3 bucket name. Falls back to ``CC_S3_BUCKET`` env var,
                then ``"commoncrawl"``.
            s3_key_prefix: Prefix to strip from ``warc_filename`` when
                building the S3 object key.  Falls back to
                ``CC_S3_KEY_PREFIX`` env var.  Default empty (key =
                warc_filename as-is, correct for the AWS ``commoncrawl``
                bucket).  Set when the bucket name overlaps with the
                leading path segment in the dataset's warc filenames.
        """
        self.warc_filename_col = warc_filename_col
        self.warc_record_offset_col = warc_record_offset_col
        self.warc_record_length_col = warc_record_length_col
        self.binary_content_col = binary_content_col
        self.drop_failed = drop_failed
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.name = "CommonCrawlWARCReader"
        self._session = None
        self._s3_client = None
        self._lock = threading.Lock()

        if use_s3 is None:
            self.use_s3 = os.environ.get("CC_USE_S3", "").lower() in ("1", "true", "yes")
        else:
            self.use_s3 = use_s3
        self.s3_bucket = s3_bucket or os.environ.get("CC_S3_BUCKET", "commoncrawl")
        self.s3_key_prefix = s3_key_prefix if s3_key_prefix is not None else os.environ.get("CC_S3_KEY_PREFIX", "")

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_session"] = None
        state["_s3_client"] = None
        state["_lock"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._session = None
        self._s3_client = None
        self._lock = threading.Lock()

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
            with self._lock:
                if self._session is None:
                    session = requests.Session()
                    adapter = requests.adapters.HTTPAdapter(
                        pool_connections=self.max_workers,
                        pool_maxsize=self.max_workers * 2,
                        max_retries=self.max_retries,
                    )
                    session.mount("https://", adapter)
                    session.mount("http://", adapter)
                    self._session = session
        return self._session

    def _get_s3_client(self) -> object:
        """Get or create a boto3 S3 client with double-checked locking.

        Credentials, region, and endpoint are resolved entirely by boto3's
        standard chain (``AWS_*`` env vars, ``~/.aws/config``, instance
        profiles).  Only connection-pool and retry settings are overridden.
        """
        if self._s3_client is None:
            with self._lock:
                if self._s3_client is None:
                    try:
                        import boto3
                    except ModuleNotFoundError as exc:
                        msg = (
                            "CommonCrawlWARCReader configured with use_s3=True but boto3 is not installed. "
                            "Install boto3 or set use_s3=False (or unset CC_USE_S3)."
                        )
                        raise RuntimeError(msg) from exc
                    from botocore.config import Config as BotoConfig

                    boto_cfg = BotoConfig(
                        max_pool_connections=self.max_workers * 2,
                        retries={"max_attempts": self.max_retries, "mode": "adaptive"},
                        connect_timeout=self.timeout,
                        read_timeout=self.timeout,
                    )
                    self._s3_client = boto3.client("s3", config=boto_cfg)
                    logger.info(f"S3 client initialized for bucket={self.s3_bucket}")
        return self._s3_client

    def _s3_key_from_filename(self, filename: str) -> str:
        """Derive S3 object key from the warc_filename column value.

        Strips ``s3_key_prefix`` from the front of *filename* when present.
        E.g. prefix ``"crawl-data/"`` + filename ``"crawl-data/CC-MAIN-…"``
        → key ``"CC-MAIN-…"``.  With an empty prefix the filename is used
        as-is (the default for the AWS ``commoncrawl`` bucket).
        """
        if self.s3_key_prefix and filename.startswith(self.s3_key_prefix):
            return filename[len(self.s3_key_prefix) :]
        return filename

    def _read_warc_record_s3(self, row: pd.Series) -> bytes | None:
        """Fetch a single WARC record using S3 range request (boto3)."""
        filename = None
        try:
            filename = row[self.warc_filename_col]
            offset = int(row[self.warc_record_offset_col])
            length = int(row[self.warc_record_length_col])
            end_byte = offset + length - 1

            resp = self._get_s3_client().get_object(
                Bucket=self.s3_bucket,
                Key=self._s3_key_from_filename(filename),
                Range=f"bytes={offset}-{end_byte}",
            )
            raw_bytes = resp["Body"].read()

            try:
                decompressed = gzip.decompress(raw_bytes)
            except gzip.BadGzipFile:
                decompressed = raw_bytes

            try:
                stream = io.BytesIO(decompressed)
                archive_iterator = ArchiveIterator(stream)
                for record in archive_iterator:
                    if record.rec_type == "response":
                        return record.content_stream().read()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to parse WARC record {filename}: {e}, returning decompressed bytes")
                return decompressed
            else:
                logger.debug(f"No response record found in WARC for {filename}, returning raw content")
                return decompressed

        except RuntimeError:
            raise  # Propagate configuration errors (e.g. missing boto3)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"S3 fetch failed for {filename}: {e}")
            return None

    def _read_warc_record(self, row: pd.Series) -> bytes | None:  # noqa: C901, PLR0911
        """Fetch a single WARC record using HTTPS range request.

        This method:
        1. Fetches gzip-compressed WARC record bytes via HTTP range request
        2. Decompresses the gzip content
        3. Parses the WARC record format using warcio
        4. Extracts and returns the HTTP response body (the actual content)
        """
        filename = None
        offset = None
        try:
            filename = row[self.warc_filename_col]
            offset = int(row[self.warc_record_offset_col])
            length = int(row[self.warc_record_length_col])

            # Build the URL
            url = urljoin(CC_BASE_URL, filename)

            # HTTP Range header (inclusive end byte)
            end_byte = offset + length - 1
            headers = {"Range": f"bytes={offset}-{end_byte}"}

            response = self._get_session().get(
                url,
                headers=headers,
                timeout=self.timeout,
            )

            # 206 Partial Content is the expected response for range requests
            if response.status_code == HTTP_PARTIAL_CONTENT:
                raw_bytes = response.content
            elif response.status_code == HTTP_OK:
                # Server ignored range request, returned full file (unusual but handle it)
                logger.warning(f"Server returned full file instead of range for {filename}")
                raw_bytes = response.content[offset : offset + length]
            else:
                logger.warning(f"Failed to fetch WARC record {filename}: HTTP {response.status_code}")
                return None

            # Decompress gzip content (WARC files from CC are .warc.gz)
            try:
                decompressed = gzip.decompress(raw_bytes)
            except gzip.BadGzipFile:
                # Content might not be gzip-compressed, use as-is
                decompressed = raw_bytes

            # Parse the WARC record using warcio to extract HTTP response body
            try:
                stream = io.BytesIO(decompressed)
                archive_iterator = ArchiveIterator(stream)
                for record in archive_iterator:
                    if record.rec_type == "response":
                        # Return the HTTP response body (content after HTTP headers)
                        return record.content_stream().read()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to parse WARC record {filename}: {e}, returning decompressed bytes")
                return decompressed
            else:
                # If no response record found, return the decompressed bytes as-is
                logger.debug(f"No response record found in WARC for {filename}, returning raw content")
                return decompressed

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
        fetch_fn = self._read_warc_record_s3 if self.use_s3 else self._read_warc_record

        def fetch_row(row_data: tuple[int, pd.Series]) -> tuple[int, bytes | None]:
            idx, row = row_data
            return idx, fetch_fn(row)

        # Use a thread pool to parallelize the HTTP requests
        # Requests are IO bound, so threads work well here
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(fetch_row, (i, row)) for i, (_, row) in enumerate(rows)]

            for future in concurrent.futures.as_completed(futures):
                try:
                    i, result = future.result()
                    results[i] = result
                except RuntimeError:  # noqa: PERF203
                    # Propagate configuration errors (e.g. missing boto3)
                    for f in futures:
                        f.cancel()
                    raise
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
