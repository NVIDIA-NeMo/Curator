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

"""JSONL reader composite stage."""

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.json as pj
from loguru import logger

from ray_curator.backends.base import WorkerMetadata
from ray_curator.backends.experimental.utils import RayStageSpecKeys
from ray_curator.stages.base import CompositeStage, ProcessingStage
from ray_curator.stages.file_partitioning import FilePartitioningStage
from ray_curator.tasks import DocumentBatch, FileGroupTask, _EmptyTask


@dataclass
class JsonlReaderStage(ProcessingStage[FileGroupTask, DocumentBatch]):
    """
    Stage that processes a group of JSONL files into a DocumentBatch.
    This stage accepts FileGroupTasks created by FilePartitioningStage
    and reads the actual file contents into DocumentBatches.

    Args:
        columns (list[str], optional): If specified, only read these columns. Defaults to None.
        reader (str, optional): Reader to use ("pyarrow" or "pandas"). Defaults to "pandas".
        reader_kwargs (dict[str, Any], optional): Keyword arguments for the reader. Defaults to {}.
        _generate_ids (bool): Whether to generate monotonically increasing IDs across all files.
            This uses IdGenerator actor, which needs to be instantiated before using this stage.
            This can be slow, so it is recommended to use AddId stage instead, unless monotonically increasing IDs
            are required.
        _assign_ids (bool): Whether to assign monotonically increasing IDs from an IdGenerator.
            This uses IdGenerator actor, which needs to be instantiated before using this stage.
            This can be slow, so it is recommended to use AddId stage instead, unless monotonically increasing IDs
            are required.
    """

    columns: list[str] | None = None  # If specified, only read these columns
    reader: str = "pandas"  # "pandas" or "pyarrow"
    reader_kwargs: dict[str, Any] = field(default_factory=dict)
    _name: str = "jsonl_reader"
    _generate_ids: bool = False
    _assign_ids: bool = False

    def __post_init__(self):
        if self._generate_ids and self._assign_ids:
            msg = "Cannot generate and assign IDs at the same time"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.columns or []

    def setup(self, _: WorkerMetadata | None = None) -> None:
        if self._generate_ids or self._assign_ids:
            from ray_curator.stages.deduplication.id_generator import get_id_generator_actor

            try:
                self.id_generator = get_id_generator_actor()
            except ValueError:
                msg = (
                    "ID generator is required when self._generate_ids or self._assign_ids is True, "
                    "and the actor 'id_generator' does not exist. Please start the id_generator actor."
                )
                raise RuntimeError(msg) from None

    def process(self, task: FileGroupTask) -> DocumentBatch:
        """
        Process a single group of JSONL files.

        Args:
            task (FileGroupTask): FileGroupTask containing file paths and configuration

        Raises:
            ValueError: If an unknown reader is provided

        Returns:
            DocumentBatch | None: DocumentBatch with the data from these files

        """
        # Get storage options from task metadata
        storage_options = task._metadata.get("storage_options", {})

        # Read the files
        if self.reader.lower() == "pandas":
            df = self._read_with_pandas(task.data, storage_options, self.reader_kwargs, self.columns)
        elif self.reader.lower() == "pyarrow":
            df = self._read_with_pyarrow(task.data, self.reader_kwargs, self.columns)
        else:
            msg = f"Unknown reader: {self.reader}"
            raise ValueError(msg)

        if df is None or (hasattr(df, "empty") and df.empty) or (hasattr(df, "num_rows") and df.num_rows == 0):
            msg = f"No data read from files in task {task.task_id}"
            raise ValueError(msg)

        # Create DocumentBatch
        return DocumentBatch(
            task_id=f"{task.task_id}_processed",
            dataset_name=task.dataset_name,
            data=df,
            _metadata=task._metadata,  # Pass through metadata from input task
        )

    def _read_with_pandas(
        self,
        file_paths: list[str],
        storage_options: dict[str, Any],
        reader_kwargs: dict[str, Any],
        columns: list[str] | None,
    ) -> pd.DataFrame | None:
        """Read JSONL files using Pandas."""

        dfs = []

        for file_path in file_paths:
            try:
                # Read the JSONL file
                df = pd.read_json(file_path, lines=True, storage_options=storage_options, **reader_kwargs)

                # Select only the specified columns if provided
                if columns is not None:
                    # Check which columns actually exist in the dataframe
                    existing_columns = [col for col in columns if col in df.columns]
                    missing_columns = [col for col in columns if col not in df.columns]

                    if missing_columns:
                        logger.warning(f"Columns {missing_columns} not found in {file_path}")

                    if existing_columns:
                        df = df[existing_columns]
                    else:
                        logger.error(f"None of the requested columns found in {file_path}")
                        continue

                dfs.append(df)
                logger.debug(f"Read {len(df)} records from {file_path}")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to read {file_path}: {e}")
                continue

        if not dfs:
            return None

        # Concatenate all dataframes
        df = pd.concat(dfs, ignore_index=True)
        if self._generate_ids:
            return self._generate_ids_func(file_paths, df)
        if self._assign_ids:
            return self._assign_ids_func(file_paths, df)
        return df

    def _assign_ids_func(self, filepath: str | list[str], df: pd.DataFrame) -> pd.DataFrame:
        import numpy as np
        import ray

        from ray_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

        if CURATOR_DEDUP_ID_STR not in df.columns:
            # Get the ID generator actor and retrieve the previously registered ID range
            min_id, max_id = ray.get(self.id_generator.get_batch_range.remote(filepath, None))
            df[CURATOR_DEDUP_ID_STR] = np.arange(min_id, max_id + 1)
        else:
            logger.warning(f"Column {CURATOR_DEDUP_ID_STR} already exists in {filepath}, not re-assigning IDs")
        return df

    def _generate_ids_func(self, filepath: str | list[str], df: pd.DataFrame) -> pd.DataFrame:
        import numpy as np
        import ray

        from ray_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

        if CURATOR_DEDUP_ID_STR not in df.columns:
            num_rows = len(df)
            min_id = ray.get(self.id_generator.register_batch.remote(filepath, num_rows))
            df[CURATOR_DEDUP_ID_STR] = np.arange(min_id, min_id + num_rows)
        else:
            logger.warning(f"Column {CURATOR_DEDUP_ID_STR} already exists in {filepath}, not generating new IDs")
        return df

    def _read_with_pyarrow(
        self, file_paths: list[str], reader_kwargs: dict[str, Any], columns: list[str] | None
    ) -> pa.Table | None:
        """Read JSONL files using PyArrow."""

        tables = []
        if self._generate_ids or self._assign_ids:
            msg = "Generating or assigning IDs is not supported for PyArrow reader"
            raise NotImplementedError(msg)

        for file_path in file_paths:
            try:
                # PyArrow JSON reader doesn't support column selection during read,
                # so we read all and then select
                table = pj.read_json(file_path, **reader_kwargs)

                # Select only the specified columns if provided
                if columns is not None:
                    # Check which columns actually exist in the table
                    existing_columns = [col for col in columns if col in table.column_names]
                    missing_columns = [col for col in columns if col not in table.column_names]

                    if missing_columns:
                        logger.warning(f"Columns {missing_columns} not found in {file_path}")

                    if existing_columns:
                        table = table.select(existing_columns)
                    else:
                        logger.error(f"None of the requested columns found in {file_path}")
                        continue

                tables.append(table)
                logger.debug(f"Read {len(table)} records from {file_path}")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to read {file_path}: {e}")
                continue

        if not tables:
            return None

        # Concatenate all tables
        return pa.concat_tables(tables)

    def ray_stage_spec(self) -> None:
        # Explicitly set this to false, otherwise due to the setup method, the stage will be treated as an actor stage
        return {RayStageSpecKeys.IS_ACTOR_STAGE: False}


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
    columns: list[str] | None = None  # If specified, only read these columns
    reader: str = "pandas"  # "pandas" or "pyarrow"
    reader_kwargs: dict[str, Any] | None = None
    storage_options: dict[str, Any] | None = None
    task_type: Literal["document", "image", "video", "audio"] = "document"
    _generate_ids: bool = False
    _assign_ids: bool = False
    _name: str = "jsonl_reader"

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()

    def decompose(self) -> list[ProcessingStage]:
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
                file_extensions=[".jsonl", ".json"],
                storage_options=self.storage_options,
            ),
            # Second stage: process file groups into document batches
            JsonlReaderStage(
                columns=self.columns,
                reader=self.reader,
                reader_kwargs=self.reader_kwargs or {},
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

        if self.columns:
            parts.append(f"reading columns: {self.columns}")

        return ", ".join(parts)
