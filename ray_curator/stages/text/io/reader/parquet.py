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

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from nemo_curator.stages.base import CompositeStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import DocumentBatch, _EmptyTask
from nemo_curator.utils.file_utils import FILETYPE_TO_DEFAULT_EXTENSIONS

from .base import BaseReader


@dataclass
class ParquetReaderStage(BaseReader):
    """
    Stage that processes a group of Parquet files into a DocumentBatch.
    This stage accepts FileGroupTasks created by FilePartitioningStage
    and reads the actual file contents into DocumentBatches.

    Args:
        fields (list[str], optional): If specified, only read these columns. Defaults to None.
        read_kwargs (dict[str, Any], optional): Keyword arguments for the underlying reader. Defaults to {}.
    """

    _name: str = "parquet_reader"

    def read_data(
        self,
        paths: list[str],
        read_kwargs: dict[str, Any] | None = None,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read Parquet files using Pandas. Raises an exception if reading fails."""

        # Normalize read_kwargs to a dict to avoid TypeError when None
        # Work on a copy to avoid mutating caller's dict
        read_kwargs = {} if read_kwargs is None else dict(read_kwargs)

        update_kwargs = {}
        if fields is not None:
            update_kwargs["columns"] = fields
        if "engine" not in read_kwargs:
            update_kwargs["engine"] = "pyarrow"
        if "dtype_backend" not in read_kwargs:
            update_kwargs["dtype_backend"] = "pyarrow"
        read_kwargs.update(update_kwargs)
        return pd.read_parquet(paths, **read_kwargs)


@dataclass
class ParquetReader(CompositeStage[_EmptyTask, DocumentBatch]):
    """Composite stage for reading Parquet files.

    This high-level stage decomposes into:
    1. FilePartitioningStage - partitions files into groups
    2. ParquetReaderStage - reads file groups into DocumentBatches
    """

    file_paths: str | list[str]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    fields: list[str] | None = None  # If specified, only read these columns
    read_kwargs: dict[str, Any] | None = None
    file_extensions: list[str] = field(default_factory=lambda: FILETYPE_TO_DEFAULT_EXTENSIONS["parquet"])
    task_type: Literal["document", "image", "video", "audio"] = "document"
    _generate_ids: bool = False
    _assign_ids: bool = False
    _name: str = "parquet_reader"

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()
        if self.read_kwargs is not None:
            self.storage_options = self.read_kwargs.get("storage_options", {})

    def decompose(self) -> list[ParquetReaderStage]:
        """Decompose into file partitioning and processing stages."""
        if self.task_type != "document":
            msg = f"Converting DocumentBatch to {self.task_type} is not supported yet."
            raise NotImplementedError(msg)

        return [
            # First stage: partition files into groups
            FilePartitioningStage(
                file_paths=self.file_paths,
                files_per_partition=self.files_per_partition,
                blocksize=self.blocksize,
                file_extensions=self.file_extensions,
                storage_options=self.read_kwargs.get("storage_options", {}) if self.read_kwargs is not None else None,
            ),
            # Second stage: process file groups into document batches
            ParquetReaderStage(
                fields=self.fields,
                read_kwargs=self.read_kwargs or {},
                _generate_ids=self._generate_ids,
                _assign_ids=self._assign_ids,
            ),
        ]

    def get_description(self) -> str:
        """Get a description of this composite stage."""

        parts = [f"Read Parquet files from {self.file_paths}"]

        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")

        if self.fields:
            parts.append(f"reading columns: {self.fields}")

        return ", ".join(parts)
