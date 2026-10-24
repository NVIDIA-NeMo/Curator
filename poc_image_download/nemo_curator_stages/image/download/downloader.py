# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Image download stage: downloads images from URLs to local storage.

This replaces cc-img-dl's downloader.py + worker.py download logic.
The async httpx approach is replaced with sync requests + ThreadPoolExecutor,
matching the pattern used by Curator's CommonCrawlWARCReader.
"""

from __future__ import annotations

import concurrent.futures
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask

from .manifest import validate_url


class ImageDownloader:
    """Downloads individual images from URLs with retry logic.

    Replaces cc-img-dl's download_image() async function with a synchronous
    implementation using requests (already a Curator dependency).

    Retry policy matches cc-img-dl:
    - 5xx: exponential backoff
    - 429: respect Retry-After header
    - 4xx (except 429): no retry
    - Network errors: exponential backoff
    """

    def __init__(
        self,
        output_dir: str,
        user_agent: str = "nemo-curator/1.0",
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
        max_retries: int = 3,
        max_image_bytes: int = 50_000_000,
    ):
        self._output_dir = output_dir
        self._user_agent = user_agent
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._max_retries = max_retries
        self._max_image_bytes = max_image_bytes
        self._session: requests.Session | None = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers["User-Agent"] = self._user_agent
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=32,
                pool_maxsize=64,
                max_retries=0,  # We handle retries ourselves
            )
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)
        return self._session

    def _url_to_filename(self, url: str) -> str:
        """Derive a local filename from a URL."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        path = parsed.path.strip("/").replace("/", "_")
        if not path:
            import hashlib
            path = hashlib.sha256(url.encode()).hexdigest()[:16]
        return path

    def download_one(self, url: str) -> str | None:
        """Download a single image URL to disk. Returns the local path or None on failure."""
        filename = self._url_to_filename(url)
        output_path = os.path.join(self._output_dir, filename)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path

        temp_path = output_path + ".tmp"
        os.makedirs(os.path.dirname(output_path) or self._output_dir, exist_ok=True)

        session = self._get_session()
        timeout = (self._connect_timeout, self._read_timeout)

        for attempt in range(self._max_retries):
            try:
                resp = session.get(url, timeout=timeout, allow_redirects=True)

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", "5"))
                    time.sleep(min(retry_after, 60))
                    continue

                if resp.status_code >= 500:
                    if attempt < self._max_retries - 1:
                        time.sleep(2**attempt)
                        continue
                    logger.warning(f"Server error {resp.status_code} for {url}")
                    return None

                if resp.status_code >= 400:
                    logger.debug(f"Client error {resp.status_code} for {url}")
                    return None

                content_type = resp.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    logger.debug(f"Non-image content type '{content_type}' for {url}")
                    return None

                data = resp.content
                if len(data) > self._max_image_bytes:
                    logger.debug(f"Image too large ({len(data)} bytes) for {url}")
                    return None

                # Validate redirect targets (SSRF protection)
                if resp.history:
                    final_url = str(resp.url)
                    if not validate_url(final_url):
                        logger.warning(f"Redirect to unsafe URL blocked: {final_url}")
                        return None

                # Atomic write via temp file
                with open(temp_path, "wb") as f:
                    f.write(data)
                os.rename(temp_path, output_path)
                return output_path

            except requests.exceptions.Timeout:
                if attempt < self._max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                logger.warning(f"Timeout downloading {url}")
                return None
            except requests.exceptions.RequestException as e:
                if attempt < self._max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                logger.warning(f"Network error downloading {url}: {e}")
                return None

        return None


@dataclass
class ImageDownloaderStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Stage that downloads images from URLs to local storage.

    Takes a FileGroupTask with URLs and returns a FileGroupTask with local file paths
    of successfully downloaded images. Failed downloads are logged and skipped.

    Uses ThreadPoolExecutor for concurrent downloads within a single task,
    matching Curator's CommonCrawlWARCReader pattern.
    """

    output_dir: str = ""
    user_agent: str = "nemo-curator/1.0"
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    max_retries: int = 3
    max_image_bytes: int = 50_000_000
    max_workers: int = 16
    name: str = "ImageDownloaderStage"
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5))

    def __post_init__(self) -> None:
        self._downloader: ImageDownloader | None = None

    def setup(self, worker_metadata: Any = None) -> None:
        """Initialize the downloader once per worker."""
        os.makedirs(self.output_dir, exist_ok=True)
        self._downloader = ImageDownloader(
            output_dir=self.output_dir,
            user_agent=self.user_agent,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
            max_retries=self.max_retries,
            max_image_bytes=self.max_image_bytes,
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: FileGroupTask) -> FileGroupTask | None:
        """Download all URLs in the task concurrently. Returns paths of successful downloads."""
        if self._downloader is None:
            self.setup()
        assert self._downloader is not None

        urls = task.data
        downloaded_paths: list[str] = []
        failed_count = 0

        def _download(url: str) -> str | None:
            return self._downloader.download_one(url)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {executor.submit(_download, url): url for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                result = future.result()
                if result:
                    downloaded_paths.append(result)
                else:
                    failed_count += 1

        if failed_count:
            logger.info(f"Task {task.task_id}: {len(downloaded_paths)} downloaded, {failed_count} failed")

        if not downloaded_paths:
            return None

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=downloaded_paths,
            _metadata={
                **task._metadata,
                "downloaded_count": len(downloaded_paths),
                "failed_count": failed_count,
                "total_urls": len(urls),
            },
            _stage_perf=task._stage_perf,
        )
