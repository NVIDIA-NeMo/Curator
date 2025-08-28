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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask


class DocumentDownloader(ABC):
    """Abstract base class for document downloaders."""

    def __init__(self, download_dir: str, verbose: bool = False):
        """Initialize the downloader.

        Args:
            download_dir: Directory to store downloaded files
            verbose: If True, logs detailed download information
        """
        self._download_dir = download_dir
        self._verbose = verbose
        os.makedirs(download_dir, exist_ok=True)

    def _check_s5cmd_installed(self) -> bool:
        """Check if s5cmd is installed."""
        try:
            subprocess.run(["s5cmd", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)  # noqa: S603, S607
        except FileNotFoundError:
            return False
        else:
            return True

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
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            if self._verbose:
                logger.info(f"File: {output_file} exists. Not downloading")
            return output_file

        # Download to temporary file
        success, error_message = self._download_to_path(url, temp_file)

        if success:
            # Download successful, atomically move temp file to final location
            os.rename(temp_file, output_file)
            if self._verbose:
                file_size = os.path.getsize(output_file)
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
    """Stage that downloads files from URLs to local storage.

    Takes a FileGroupTask with URLs and returns a FileGroupTask with local file paths.
    This allows the download step to scale independently from iteration/extraction.
    """

    _resources = Resources(cpus=0.5)
    downloader: DocumentDownloader
    _batch_size = None

    def __post_init__(self):
        self._name = f"download_{self.downloader.__class__.__name__.lower()}"

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define input requirements - expects FileGroupTask with URLs."""
        return (["data"], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define output - produces FileGroupTask with local paths."""
        return (["data"], [])

    def process(self, task: FileGroupTask) -> FileGroupTask:
        """Download URLs to local files.

        Args:
            task (FileGroupTask): Task containing URLs to download

        Returns:
            FileGroupTask: Task containing local file paths
        """
        local_files = []

        for url in task.data:
            downloaded_file = self.downloader.download(url)
            if downloaded_file:
                local_files.append(downloaded_file)

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=local_files,
            _metadata={
                **task._metadata,
                "source_files": local_files,  # Add downloaded files for deterministic naming during write stage
            },
            _stage_perf=task._stage_perf,
        )

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {
            "num_workers_per_node": self.downloader.num_workers_per_node(),
        }
