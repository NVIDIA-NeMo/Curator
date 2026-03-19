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

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import ray
from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata

from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, FileGroupTask

# Enforce contiguous buffers under ~2 GiB
_MAX_IN_MEMORY_BYTES = 2**31


def get_file_size(path: str) -> int:
    """
    Returns the file size in bytes, works for local or remote paths.
    """
    # Treat as local if it looks like an absolute or relative path
    if os.path.exists(path):
        return os.path.getsize(path)

    # Otherwise treat as remote via fsspec
    fs, relative_path = fsspec.core.url_to_fs(path)
    info = fs.info(relative_path)
    return info["size"]


def _dataframe_memory_bytes(df: pd.DataFrame) -> int:
    """Total in-RAM footprint including Python str/object columns."""
    return int(df.memory_usage(deep=True).sum())


@dataclass
class BaseReader(ProcessingStage[FileGroupTask, DocumentBatch]):
    """Common base for tabular file readers.

    Subclasses must implement the read_data method.
    """

    fields: list[str] | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    _generate_ids: bool = False
    _assign_ids: bool = False

    def __post_init__(self) -> None:
        if self._generate_ids and self._assign_ids:
            msg = "Cannot generate and assign IDs at the same time"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        output_fields = self.fields or []
        if self._generate_ids or self._assign_ids:
            from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

            output_fields.append(CURATOR_DEDUP_ID_STR)
        return ["data"], output_fields

    def setup(self, _: WorkerMetadata | None = None) -> None:
        if self._generate_ids or self._assign_ids:
            from nemo_curator.stages.deduplication.id_generator import get_id_generator_actor

            try:
                self.id_generator = get_id_generator_actor()
            except ValueError:
                msg = (
                    "ID generator is required when self._generate_ids or self._assign_ids is True, "
                    "and the actor 'id_generator' does not exist. Please start the id_generator actor."
                )
                raise RuntimeError(msg) from None

    def process(self, task: FileGroupTask) -> DocumentBatch:
        # Verify storage size of input files is not greater than 2 GiB
        # This should be a very quick check per file, so we do it first before reading the data
        total_storage_size = 0
        for file_path in task.data:
            file_size = get_file_size(file_path)
            total_storage_size += file_size
        if total_storage_size > _MAX_IN_MEMORY_BYTES:
            msg = (
                f"Error reading data from files: {task.data}. "
                f"Total storage size is {total_storage_size} bytes (limit {_MAX_IN_MEMORY_BYTES} bytes). "
                "Large input files should be split into smaller chunks using nemo_curator.utils.split_large_files."
            )
            raise ValueError(msg)

        # Merge read kwargs with storage options precedence: task.storage_options > self.read_kwargs
        effective_read_kwargs: dict[str, Any] = {}
        if self.read_kwargs:
            effective_read_kwargs.update(self.read_kwargs)

        # Read the files
        result = self.read_data(task.data, effective_read_kwargs, self.fields)

        # Validate the result
        if (
            (result is None)
            or (hasattr(result, "empty") and result.empty)
            or (hasattr(result, "num_rows") and result.num_rows == 0)
        ):
            msg = f"No data read from files in task {task.task_id}"
            raise ValueError(msg)

        # Even though we checked the storage size of the input files, the total in-memory size of the DataFrame can still be too large
        # This is a more expensive but more accurate check than the storage size check
        if isinstance(result, pa.Table):
            total_bytes = _dataframe_memory_bytes(result.to_pandas())
        elif hasattr(result, "memory_usage"):
            total_bytes = _dataframe_memory_bytes(result)
        else:
            raise ValueError(f"Unsupported result type: {type(result)}")

        if total_bytes > _MAX_IN_MEMORY_BYTES:
            msg = (
                f"Error reading data from files: {task.data}. "
                f"Estimated in-memory size is {total_bytes} bytes (limit {_MAX_IN_MEMORY_BYTES} bytes). "
                "Large input files should be split into smaller chunks using nemo_curator.utils.split_large_files."
            )
            raise ValueError(msg)

        # Apply IDs only for Pandas DataFrames
        if isinstance(result, pd.DataFrame):
            if self._generate_ids:
                result = self._generate_ids_func(task.data, result)
            elif self._assign_ids:
                result = self._assign_ids_func(task.data, result)

        return DocumentBatch(
            task_id=f"{task.task_id}_processed",
            dataset_name=task.dataset_name,
            data=result,
            _metadata=task._metadata,
        )

    # Subclass responsibilities -------------------------------------------------
    def read_data(
        self,
        file_paths: list[str],
        read_kwargs: dict[str, Any] | None,
        fields: list[str] | None,
    ) -> pd.DataFrame | None:  # pragma: no cover - abstract
        raise NotImplementedError

    # ID helpers ----------------------------------------------------------------
    def _assign_ids_func(self, filepath: str | list[str], df: pd.DataFrame) -> pd.DataFrame:
        from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

        if CURATOR_DEDUP_ID_STR not in df.columns:
            min_id, max_id = ray.get(self.id_generator.get_batch_range.remote(filepath, None))
            df[CURATOR_DEDUP_ID_STR] = np.arange(min_id, max_id + 1)
        else:
            logger.warning(f"Column {CURATOR_DEDUP_ID_STR} already exists in {filepath}, not re-assigning IDs")
        return df

    def _generate_ids_func(self, filepath: str | list[str], df: pd.DataFrame) -> pd.DataFrame:
        from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

        if CURATOR_DEDUP_ID_STR not in df.columns:
            num_rows = len(df)
            min_id = ray.get(self.id_generator.register_batch.remote(filepath, num_rows))
            df[CURATOR_DEDUP_ID_STR] = np.arange(min_id, min_id + num_rows)
        else:
            logger.warning(f"Column {CURATOR_DEDUP_ID_STR} already exists in {filepath}, not generating new IDs")
        return df

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_ACTOR_STAGE: self._generate_ids or self._assign_ids}
