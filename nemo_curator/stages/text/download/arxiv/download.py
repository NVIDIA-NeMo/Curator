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

from loguru import logger

from nemo_curator.stages.text.download import DocumentDownloader


class ArxivDownloader(DocumentDownloader):
    """Downloads Arxiv data from s3://arxiv/src/"""

    def __init__(self, download_dir: str, verbose: bool = False):
        super().__init__(download_dir, verbose)
        if not self._check_s5cmd_installed():
            msg = "s5cmd is not installed. Please install it from https://github.com/peak/s5cmd"
            raise RuntimeError(msg)

    def _get_output_filename(self, url: str) -> str:
        # Use tarfile name as output filename
        return url

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        s3path = os.path.join("s3://arxiv/src", url)

        if self._verbose:
            logger.info(f"Downloading {s3path} and writing to {path}")

        cmd = ["s5cmd", "--request-payer=requester", "cp", s3path, path]

        if self._verbose:
            stdout, stderr = None, None
        else:
            stdout, stderr = subprocess.DEVNULL, subprocess.DEVNULL

        p = subprocess.run(  # noqa: S603, PLW1510
            cmd,
            stdout=stdout,
            stderr=stderr,
        )

        if p.returncode != 0:
            if self._verbose:
                logger.error(f"Failed to download {s3path} to {path}")
            return False, f"Failed to download {s3path} to {path}"

        return True, None
