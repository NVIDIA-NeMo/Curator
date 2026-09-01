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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd
import pyarrow as pa
import ray
from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, FileGroupTask, LanceReadTask

ReaderTask: TypeAlias = FileGroupTask | LanceReadTask
ReaderData: TypeAlias = pd.DataFrame | pa.Table


@dataclass(frozen=True)
class ReaderOutput:
    data: ReaderData
    metadata: dict[str, Any] | None = None


@dataclass
class BaseReader(ProcessingStage[ReaderTask, DocumentBatch]):
    """Common base for tabular readers.

    Subclasses must implement read_task for their input task type.
    """

    fields: list[str] | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    _generate_ids: bool = False
    _assign_ids: bool = False
    # Permit valid zero-row results.
    allow_empty: bool = False

    def __post_init__(self) -> None:
        if self._generate_ids and self._assign_ids:
            msg = "Cannot generate and assign IDs at the same time"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        output_fields = list(self.fields or [])
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

    def process(self, task: ReaderTask) -> DocumentBatch:
        output = self.read_task(task, dict(self.read_kwargs or {}), self.fields)
        self._validate_result(task, output.data)
        return self._document_batch(task, output)

    def _document_batch(self, task: ReaderTask, output: ReaderOutput) -> DocumentBatch:
        batch = DocumentBatch(
            dataset_name=task.dataset_name,
            data=output.data,
            _metadata=output.metadata if output.metadata is not None else task._metadata,
        )
        if self._generate_ids or self._assign_ids:
            batch_key = self._id_generator_key(task)
            if self._generate_ids:
                self._generate_ids_func(batch_key, batch)
            else:
                self._assign_ids_func(batch_key, batch)

        return batch

    def _validate_result(self, task: ReaderTask, result: ReaderData) -> None:
        if self.allow_empty:
            return
        if (
            (result is None)
            or (isinstance(result, pd.DataFrame) and result.empty)
            or (isinstance(result, pa.Table) and result.num_rows == 0)
        ):
            msg = f"No data read from files in task {task.task_id}"
            raise ValueError(msg)

    # Subclass responsibilities -------------------------------------------------
    def read_task(
        self,
        task: ReaderTask,
        read_kwargs: dict[str, Any] | None,
        fields: list[str] | None,
    ) -> ReaderOutput:  # pragma: no cover - abstract
        raise NotImplementedError

    # ID helpers ----------------------------------------------------------------
    @staticmethod
    def _id_generator_key(task: ReaderTask) -> str | list[str]:
        # TODO(NMCUR-315): Use the deterministic task ID for FileGroupTask as well.
        # Keep returning file paths for backward compatibility until existing ID registries are migrated.
        if isinstance(task, FileGroupTask):
            return task.data
        return task.get_deterministic_id()

    @staticmethod
    def _append_ids(batch: DocumentBatch, start_id: int, count: int) -> None:
        from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

        ids = np.arange(start_id, start_id + count)
        if isinstance(batch.data, pd.DataFrame):
            batch.data[CURATOR_DEDUP_ID_STR] = ids
        else:
            batch.data = batch.data.append_column(CURATOR_DEDUP_ID_STR, pa.array(ids, type=pa.int64()))

    def _assign_ids_func(self, batch_key: str | list[str], batch: DocumentBatch) -> None:
        from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

        if CURATOR_DEDUP_ID_STR not in batch.get_columns():
            min_id, max_id = ray.get(self.id_generator.get_batch_range.remote(batch_key, None))
            self._append_ids(batch, min_id, max_id - min_id + 1)
        else:
            logger.warning(f"Column {CURATOR_DEDUP_ID_STR} already exists in {batch_key}, not re-assigning IDs")

    def _generate_ids_func(self, batch_key: str | list[str], batch: DocumentBatch) -> None:
        from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

        if CURATOR_DEDUP_ID_STR not in batch.get_columns():
            min_id = ray.get(self.id_generator.register_batch.remote(batch_key, batch.num_items))
            self._append_ids(batch, min_id, batch.num_items)
        else:
            logger.warning(f"Column {CURATOR_DEDUP_ID_STR} already exists in {batch_key}, not generating new IDs")

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_ACTOR_STAGE: self._generate_ids or self._assign_ids}


@dataclass
class BaseFileReader(BaseReader):
    """Base reader for file-group readers that consume lists of paths."""

    def read_task(
        self,
        task: FileGroupTask,
        read_kwargs: dict[str, Any] | None,
        fields: list[str] | None,
    ) -> ReaderOutput:
        return ReaderOutput(self.read_data(task.data, read_kwargs, fields))

    def read_data(
        self,
        file_paths: list[str],
        read_kwargs: dict[str, Any] | None,
        fields: list[str] | None,
    ) -> ReaderData:  # pragma: no cover - abstract
        raise NotImplementedError
