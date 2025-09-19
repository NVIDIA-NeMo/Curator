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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import ray
from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata

from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, FileGroupTask


@dataclass
class BaseReader(ProcessingStage[FileGroupTask, DocumentBatch]):
    """Common base for tabular file readers.

    Subclasses must implement the read_data method.
    """

    fields: list[str] | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    _name: str = ""
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
        # Merge read kwargs with storage options precedence: task.storage_options > self.read_kwargs
        effective_read_kwargs: dict[str, Any] = {}
        if self.read_kwargs:
            effective_read_kwargs.update(self.read_kwargs)
        # Read the files
        result = self.read_data(task.data, effective_read_kwargs, self.fields)
        # Validate
        if (
            result is None
            or (hasattr(result, "empty") and result.empty)
            or (hasattr(result, "num_rows") and result.num_rows == 0)
        ):
            msg = f"No data read from files in task {task.task_id}"
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
