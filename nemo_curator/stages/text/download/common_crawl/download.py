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

from loguru import logger

from nemo_curator.stages.text.download import DocumentDownloader


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
