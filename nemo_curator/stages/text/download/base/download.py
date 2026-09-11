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
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask


class DocumentDownloader(ABC):
    """Abstract base class for document downloaders."""

    supports_remote_download_dir = False

    def __init__(self, download_dir: str, verbose: bool = False, storage_options: dict[str, Any] | None = None):
        """Initialize the downloader.

        Args:
            download_dir: Directory to store downloaded files, a local path or an fsspec URL
                (the latter only for subclasses that set ``supports_remote_download_dir``)
            verbose: If True, logs detailed download information
            storage_options: Options forwarded to the fsspec filesystem inferred from ``download_dir``
        """
        self._fs, _ = url_to_fs(download_dir, **(storage_options or {}))
        self._is_local = isinstance(self._fs, LocalFileSystem)
        if not self._is_local and not self.supports_remote_download_dir:
            msg = f"{type(self).__name__} needs a local download_dir, got {download_dir!r}"
            raise ValueError(msg)
        self._download_dir = download_dir.removeprefix("file://") if self._is_local else download_dir
        self._verbose = verbose
        # Object stores need no directory, and s3fs makedirs would try to create a missing bucket.
        if self._is_local:
            os.makedirs(self._download_dir, exist_ok=True)

    @abstractmethod
    def _get_output_filename(self, url: str) -> str:
        """Generate output filename from URL.

        Args:
            url: URL to download

        Returns:
            Output filename (without directory path)
        """
        ...

    @abstractmethod
    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        """Download URL to specified path.

        Args:
            url: URL to download
            path: Local path to save file

        Returns:
            Tuple of (success, error_message). If success is True, error_message should be None.
            If success is False, error_message should contain the error details.
        """
        ...

    def download(self, url: str) -> str | None:
        """Download a document from URL with temporary file handling.

        Downloads file to temporary location then atomically moves to final path.
        Checks for existing file to avoid re-downloading. Supports resumable downloads.
        Args:
            url: URL to download

        Returns:
            Path to downloaded file, or None if download failed
        """
        # Generate output filename
        output_name = self._get_output_filename(url)
        output_file = os.path.join(self._download_dir, output_name)
        temp_file = output_file + ".tmp"

        # If final file exists and is non-empty, assume it's complete
        if self._fs.exists(output_file) and self._fs.size(output_file) > 0:
            if self._verbose:
                logger.info(f"File: {output_file} exists. Not downloading")
            return output_file

        if self._is_local:
            # Download to temporary file, then atomically move it to the final location
            success, error_message = self._download_to_path(url, temp_file)
            if success:
                os.rename(temp_file, output_file)
        else:
            # No remote .tmp: a failed upload never leaves a partial final object, so there is nothing to
            # clean up (S3 https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutObject.html,
            # GCS https://cloud.google.com/storage/docs/consistency). Aborted multipart parts are out of scope.
            with tempfile.TemporaryDirectory() as staging_dir:
                staged_file = os.path.join(staging_dir, output_name)
                success, error_message = self._download_to_path(url, staged_file)
                if success:
                    self._fs.put_file(staged_file, output_file)

        if success:
            if self._verbose:
                file_size = self._fs.size(output_file)
                logger.info(f"Successfully downloaded to {output_file} ({file_size} bytes)")
            return output_file
        else:
            # Download failed
            logger.error(f"Failed to download to {output_file}: {error_message}")
            return None

    def num_workers_per_node(self) -> int | None:
        """Number of workers per node for Downloading. This is sometimes needed to ensure we are not overloading the network.

        Returns:
            Number of workers per node, or None if there is no limit and we can download as fast as possible
        """
        return None


@dataclass
class DocumentDownloadStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Stage that downloads files from URLs to local or fsspec storage.

    Takes a FileGroupTask with URLs and returns a FileGroupTask with local paths or fsspec URLs.
    This allows the download step to scale independently from iteration/extraction.
    """

    resources = Resources(cpus=0.5)
    downloader: DocumentDownloader
    batch_size = None

    def __post_init__(self):
        self.name = f"download_{self.downloader.__class__.__name__.lower()}"

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define input requirements - expects FileGroupTask with URLs."""
        return (["data"], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define output - produces FileGroupTask with local paths or fsspec URLs."""
        return (["data"], [])

    def process(self, task: FileGroupTask) -> FileGroupTask:
        """Download URLs to local or fsspec files.

        Args:
            task (FileGroupTask): Task containing URLs to download

        Returns:
            FileGroupTask: Task containing local file paths or fsspec URLs
        """
        local_files = []

        for url in task.data:
            downloaded_file = self.downloader.download(url)
            if downloaded_file:
                local_files.append(downloaded_file)

        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=local_files,
            _metadata={
                **task._metadata,
                "source_files": local_files,  # Add downloaded files for deterministic naming during write stage
            },
            _stage_perf=task._stage_perf,
        )

    def num_workers_per_node(self) -> float | None:
        return self.downloader.num_workers_per_node()
