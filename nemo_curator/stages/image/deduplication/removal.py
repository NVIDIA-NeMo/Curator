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
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import ImageBatch


@dataclass
class ImageDuplicatesRemovalStage(ProcessingStage[ImageBatch, ImageBatch]):
    """Filter stage that removes images whose IDs appear in a Parquet file.

    The Parquet file must contain a column with image identifiers; by default this
    column is assumed to be ``id`` to match writer metadata. You can change
    the column name via ``duplicate_id_field``.

    Args:
        removal_parquets_dir: Directory containing Parquet files with image IDs to remove
        duplicate_id_field: Name of the column containing image IDs to remove
        verbose: Whether to log verbose output
        num_workers_per_node: Number of workers per node for the stage. This is sometimes needed
            to avoid OOM when concurrently running actors on one node loading the same removal
            parquet files into memory.
    """

    removal_parquets_dir: str
    duplicate_id_field: str = "id"
    verbose: bool = False
    num_workers_per_node: int | None = None

    _name: str = "image_dedup_filter"

    # Internal cache
    _ids_to_remove: set[str] = field(default_factory=set)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, _worker_metadata=None) -> None:  # noqa: ANN001
        removal_parquets = [os.path.join(self.removal_parquets_dir, f) for f in os.listdir(self.removal_parquets_dir) if f.endswith(".parquet")]
        if not removal_parquets:
            msg = f"No parquet files found in {self.removal_parquets_dir}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Read entire directory using a Dataset with explicit schema for the ID column.
        # This avoids Arrow attempting to unify string and null types from empty files.
        schema = pa.schema([pa.field(self.duplicate_id_field, pa.string())])
        dataset = ds.dataset(self.removal_parquets_dir, format="parquet", schema=schema)
        table = dataset.to_table(columns=[self.duplicate_id_field])
        ids_array = table[self.duplicate_id_field].to_pylist()
        self._ids_to_remove.update(ids_array)

        if self.verbose:
            logger.debug(
                f"Loaded {len(self._ids_to_remove)} IDs to remove from '{self.removal_parquets_dir}'"
            )

    def process(self, task: ImageBatch) -> ImageBatch:
        original_count = len(task.data)
        ids_to_remove = self._ids_to_remove  # local reference for faster lookups
        filtered_images = [img for img in task.data if img.image_id not in ids_to_remove]

        removed_count = original_count - len(filtered_images)
        if self.verbose:
            logger.debug(
                f"Dedup filtering: kept {len(filtered_images)}/{original_count} images, "
                f"removed {removed_count} by ID"
            )

        return ImageBatch(
            data=filtered_images,
            dataset_name=task.dataset_name,
            task_id=f"{task.task_id}_{self._name}",
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_per_node is not None:
            spec["num_workers_per_node"] = self.num_workers_per_node
        return spec
