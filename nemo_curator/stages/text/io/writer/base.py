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

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from fsspec.core import url_to_fs
from loguru import logger

import nemo_curator.stages.text.io.writer.utils as writer_utils
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.client_utils import is_remote_url
from nemo_curator.utils.file_utils import check_output_mode


@dataclass
class BaseWriter(ProcessingStage[DocumentBatch, FileGroupTask], ABC):
    """Base class for all writer stages.

    This abstract base class provides common functionality for writing DocumentBatch
    tasks to files, including file naming, metadata handling, and filesystem operations.
    """

    path: str
    file_extension: str
    write_kwargs: dict[str, Any] = field(default_factory=dict)
    fields: list[str] | None = None
    name: str = "BaseWriter"
    mode: Literal["ignore", "overwrite", "append", "error"] = "ignore"
    append_mode_implemented: bool = False

    def __post_init__(self):
        # Use fsspec's url_to_fs to get both filesystem and normalized path
        self.storage_options = (self.write_kwargs or {}).get("storage_options", {})
        self.fs, self._fs_path = url_to_fs(self.path, **self.storage_options)
        check_output_mode(self.mode, self.fs, self._fs_path, append_mode_implemented=self.append_mode_implemented)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def get_file_extension(self) -> str:
        """Return the file extension for this writer format."""
        return self.file_extension

    @abstractmethod
    def write_data(self, task: DocumentBatch, file_path: str) -> None:
        """Write data to file using format-specific implementation."""

    def process(self, task: DocumentBatch) -> FileGroupTask:
        """Process a DocumentBatch and write to files.

        Args:
            task (DocumentBatch): DocumentBatch containing data to write

        Returns:
            FileGroupTask: Task containing paths to written files
        """
        # Get source files from metadata for deterministic naming
        if source_files := task._metadata.get("source_files"):
            filename = writer_utils.get_deterministic_hash(source_files, task.task_id)
        else:
            logger.warning("The task does not have source_files in metadata, using UUID for base filename")
            filename = uuid.uuid4().hex

        # Generate filename with appropriate extension using normalized fs path
        file_extension = self.get_file_extension()
        file_path = self.fs.sep.join([self._fs_path, f"{filename}.{file_extension}"])

        # For remote URLs, restore the protocol prefix so downstream code can infer the filesystem
        file_path_with_protocol = self.fs.unstrip_protocol(file_path) if is_remote_url(self.path) else file_path

        if self.fs.exists(file_path):
            logger.debug(f"File {file_path_with_protocol} already exists, overwriting it")

        self.write_data(task, file_path_with_protocol)
        logger.debug(f"Written {task.num_items} records to {file_path_with_protocol}")

        # Create FileGroupTask with written files using the full protocol-prefixed path
        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[file_path_with_protocol],
            _metadata={
                **task._metadata,
                "format": self.get_file_extension(),
            },
            _stage_perf=task._stage_perf,
        )
