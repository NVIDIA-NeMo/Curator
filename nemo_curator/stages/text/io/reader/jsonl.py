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
from loguru import logger

from nemo_curator.stages.base import CompositeStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import DocumentBatch, _EmptyTask
from nemo_curator.utils.file_utils import FILETYPE_TO_DEFAULT_EXTENSIONS, pandas_select_columns

from .base import BaseReader


@dataclass
class JsonlReaderStage(BaseReader):
    """
    Stage that processes a group of JSONL files into a DocumentBatch.
    This stage accepts FileGroupTasks created by FilePartitioningStage
    and reads the actual file contents into DocumentBatches.

    Args:
        fields (list[str], optional): If specified, only read these fields (columns). Defaults to None.
        read_kwargs (dict[str, Any], optional): Keyword arguments for the reader. Defaults to {}.
        _generate_ids (bool): Whether to generate monotonically increasing IDs across all files.
            This uses IdGenerator actor, which needs to be instantiated before using this stage.
            This can be slow, so it is recommended to use AddId stage instead, unless monotonically increasing IDs
            are required.
        _assign_ids (bool): Whether to assign monotonically increasing IDs from an IdGenerator.
            This uses IdGenerator actor, which needs to be instantiated before using this stage.
            This can be slow, so it is recommended to use AddId stage instead, unless monotonically increasing IDs
            are required.
    """

    _name: str = "jsonl_reader"

    def read_data(
        self,
        paths: list[str],
        read_kwargs: dict[str, Any] | None = None,
        fields: list[str] | None = None,
    ) -> pd.DataFrame | None:
        """Read JSONL files using Pandas."""

        # Normalize read_kwargs to a dict to avoid TypeError when None
        # Work on a copy to avoid mutating caller's dict
        read_kwargs = {} if read_kwargs is None else dict(read_kwargs)
        # Default to lines=True if not specified
        if "lines" in read_kwargs and read_kwargs["lines"] is False:
            msg = "lines=False is not supported for JSONL reader"
            raise ValueError(msg)
        else:
            read_kwargs["lines"] = True

        dfs = []
        for file_path in paths:
            df = pd.read_json(file_path, **read_kwargs)
            if fields is not None:
                df = pandas_select_columns(df, fields, file_path)
            dfs.append(df)
        # Concatenate all dataframes
        if not dfs:
            msg = f"No data read from files in task {paths} with read_kwargs {read_kwargs} in JSONL reader"
            logger.error(msg)
            raise ValueError(msg)
        return pd.concat(dfs, ignore_index=True)


@dataclass
class JsonlReader(CompositeStage[_EmptyTask, DocumentBatch]):
    """Composite stage for reading JSONL files.

    This high-level stage decomposes into:
    1. FilePartitioningStage - partitions files into groups
    2. JsonlReaderStage - reads file groups into DocumentBatches
    """

    file_paths: str | list[str]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    fields: list[str] | None = None  # If specified, only read these columns
    read_kwargs: dict[str, Any] | None = None
    task_type: Literal["document", "image", "video", "audio"] = "document"
    file_extensions: list[str] = field(default_factory=lambda: FILETYPE_TO_DEFAULT_EXTENSIONS["jsonl"])
    _generate_ids: bool = False
    _assign_ids: bool = False
    _name: str = "jsonl_reader"

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()
        if self.read_kwargs is not None:
            self.storage_options = self.read_kwargs.get("storage_options", {})

    def decompose(self) -> list[JsonlReaderStage]:
        """Decompose into file partitioning and processing stages."""
        if self.task_type != "document":
            msg = f"Converting DocumentBatch to {self.task_type} is not supported yet."
            raise NotImplementedError(msg)

        return [
            FilePartitioningStage(
                file_paths=self.file_paths,
                files_per_partition=self.files_per_partition,
                blocksize=self.blocksize,
                file_extensions=self.file_extensions,
                storage_options=self.read_kwargs.get("storage_options", None)
                if self.read_kwargs is not None
                else None,
            ),
            JsonlReaderStage(
                fields=self.fields,
                read_kwargs=(self.read_kwargs or {}),
                _generate_ids=self._generate_ids,
                _assign_ids=self._assign_ids,
            ),
        ]

    def get_description(self) -> str:
        """Get a description of this composite stage."""

        parts = [f"Read JSONL files from {self.file_paths}"]

        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")

        if self.fields:
            parts.append(f"reading columns: {self.fields}")

        return ", ".join(parts)
