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

from dataclasses import dataclass, field
from typing import Any, Literal

import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.json as paj
from loguru import logger

from nemo_curator.stages.base import CompositeStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import DocumentBatch, EmptyTask
from nemo_curator.utils.client_utils import is_remote_url
from nemo_curator.utils.file_utils import FILETYPE_TO_DEFAULT_EXTENSIONS, pandas_select_columns

from .base import BaseFileReader

PANDAS_ENGINE = "pandas"
PYARROW_DIRECT_ENGINE = "pyarrow_direct"
# Read 8 MiB chunks to accommodate most rows, but retry up to 256 MiB for
# rows containing large binary payloads such as base64-encoded images or PDFs.
DEFAULT_PYARROW_BLOCK_SIZE = 8 * 1024 * 1024
DEFAULT_PYARROW_MAX_BLOCK_SIZE = 256 * 1024 * 1024


def _validate_jsonl_read_kwargs(read_kwargs: dict[str, Any] | None) -> None:
    if read_kwargs is not None and read_kwargs.get("lines", True) is False:
        msg = "JsonlReader only supports lines=True"
        raise RuntimeError(msg)


def _pyarrow_select_columns(table: pa.Table, fields: list[str] | None, file_path: str) -> pa.Table | None:
    if fields is None:
        return table

    existing_fields = [column for column in fields if column in table.column_names]
    missing_fields = [column for column in fields if column not in table.column_names]
    if missing_fields:
        logger.warning(f"Columns {missing_fields} not found in {file_path}")
    if existing_fields:
        return table.select(existing_fields)

    logger.error(f"None of the requested columns found in {file_path}")
    return None


def _read_jsonl_file_with_pyarrow(
    file_path: str,
    block_size: int,
    max_block_size: int,
    storage_options: dict[str, Any],
    compression: str | None,
) -> pa.Table:
    """Read one JSONL file, growing the parser block for an oversized record.

    PyArrow reports ``straddling object`` when a JSON object is too large for
    its current parsing window. Each retry doubles the block size up to
    ``max_block_size``; the error is re-raised at the ceiling, so this loop is
    bounded. Remote paths and custom storage options are opened through
    ``fsspec`` and passed to PyArrow as a file-like stream.
    """
    while True:
        try:
            read_options = paj.ReadOptions(block_size=block_size, use_threads=False)
            if not is_remote_url(file_path) and not storage_options and compression == "infer":
                return paj.read_json(file_path, read_options=read_options)
            with fsspec.open(
                file_path,
                mode="rb",
                compression=compression,
                **storage_options,
            ) as stream:
                return paj.read_json(stream, read_options=read_options)
        except pa.ArrowInvalid as error:
            if "straddling object" not in str(error) or block_size >= max_block_size:
                raise
            block_size = min(block_size * 2, max_block_size)


def _read_jsonl_with_pyarrow(
    paths: list[str],
    read_kwargs: dict[str, Any],
    fields: list[str] | None,
) -> pa.Table:
    """Read JSONL paths directly with PyArrow."""
    read_kwargs = dict(read_kwargs)
    read_kwargs.pop("engine", None)
    read_kwargs.pop("lines", None)

    block_size = read_kwargs.pop("pyarrow_block_size", DEFAULT_PYARROW_BLOCK_SIZE)
    max_block_size = read_kwargs.pop("pyarrow_max_block_size", DEFAULT_PYARROW_MAX_BLOCK_SIZE)
    storage_options = read_kwargs.pop("storage_options", {}) or {}
    compression = read_kwargs.pop("compression", "infer")
    if block_size <= 0 or max_block_size < block_size:
        msg = "pyarrow block sizes must be positive and max block size must be at least the initial size"
        raise ValueError(msg)
    if read_kwargs:
        unsupported = ", ".join(sorted(read_kwargs))
        msg = f"Unsupported read_kwargs for engine={PYARROW_DIRECT_ENGINE!r}: {unsupported}"
        raise TypeError(msg)

    tables = []
    for file_path in paths:
        table = _read_jsonl_file_with_pyarrow(
            file_path,
            block_size,
            max_block_size,
            storage_options,
            compression,
        )
        table = _pyarrow_select_columns(table, fields, file_path)
        if table is not None:
            tables.append(table)
    if not tables:
        msg = f"No data read from files in task {paths} with direct PyArrow JSONL reader"
        logger.error(msg)
        raise ValueError(msg)

    return pa.concat_tables(tables, promote_options="permissive")


def _read_jsonl_with_pandas(
    paths: list[str],
    read_kwargs: dict[str, Any],
    fields: list[str] | None,
) -> pd.DataFrame:
    read_kwargs = dict(read_kwargs)
    if read_kwargs.get("engine") == PANDAS_ENGINE:
        read_kwargs.pop("engine")
    read_kwargs["lines"] = True

    dfs = []
    for file_path in paths:
        df = pd.read_json(file_path, **read_kwargs)
        if fields is not None:
            df = pandas_select_columns(df, fields, file_path)
        dfs.append(df)
    if not dfs:
        msg = f"No data read from files in task {paths} with read_kwargs {read_kwargs} in JSONL reader"
        logger.error(msg)
        raise ValueError(msg)
    return pd.concat(dfs, ignore_index=True)


@dataclass
class JsonlReaderStage(BaseFileReader):
    """
    Stage that processes a group of JSONL files into a DocumentBatch.
    This stage accepts FileGroupTasks created by FilePartitioningStage
    and reads the actual file contents into DocumentBatches.

    Args:
        fields (list[str], optional): If specified, only read these fields (columns). Defaults to None.
        read_kwargs (dict[str, Any], optional): Reader options. ``engine="pyarrow_direct"``
            uses the direct PyArrow reader; all other engines, including ``"pyarrow"``,
            are passed to ``pd.read_json``. Defaults to {}.
        _generate_ids (bool): Whether to generate monotonically increasing IDs across all files.
            This uses IdGenerator actor, which needs to be instantiated before using this stage.
            This can be slow, so it is recommended to use AddId stage instead, unless monotonically increasing IDs
            are required.
        _assign_ids (bool): Whether to assign monotonically increasing IDs from an IdGenerator.
            This uses IdGenerator actor, which needs to be instantiated before using this stage.
            This can be slow, so it is recommended to use AddId stage instead, unless monotonically increasing IDs
            are required.
    """

    name: str = "jsonl_reader"

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_jsonl_read_kwargs(self.read_kwargs)

    def read_data(
        self,
        paths: list[str],
        read_kwargs: dict[str, Any] | None = None,
        fields: list[str] | None = None,
    ) -> pd.DataFrame | pa.Table:
        """Read JSONL files using the selected engine."""

        read_kwargs = {} if read_kwargs is None else dict(read_kwargs)
        engine = read_kwargs.get("engine", PYARROW_DIRECT_ENGINE)
        if engine == PYARROW_DIRECT_ENGINE:
            return _read_jsonl_with_pyarrow(paths, read_kwargs, fields)
        return _read_jsonl_with_pandas(paths, read_kwargs, fields)


@dataclass
class JsonlReader(CompositeStage[EmptyTask, DocumentBatch]):
    """Composite stage for reading JSONL files.

    This high-level stage decomposes into:
    1. FilePartitioningStage - partitions files into groups
    2. JsonlReaderStage - reads file groups into DocumentBatches

    Args:
        file_paths: File paths, directories, or glob patterns to read.
        files_per_partition: Number of files grouped into each reader task.
            When set, this takes precedence over ``blocksize``.
        blocksize: Target storage size for each file-group task.
        fields: Optional columns to retain.
        read_kwargs: Options passed to ``pd.read_json``, or to the direct
            PyArrow reader when ``engine="pyarrow_direct"``.
        task_type: Output task modality. Only ``"document"`` is supported.
        file_extensions: File extensions considered during partitioning.
        _generate_ids: Generate stable, monotonically increasing document IDs.
        _assign_ids: Assign IDs previously registered for the same reader task.
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
    name: str = "jsonl_reader"

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()
        _validate_jsonl_read_kwargs(self.read_kwargs)
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
