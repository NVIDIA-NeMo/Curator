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
import os

from loguru import logger
import pyarrow.parquet as pq

from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import ImageBatch


@dataclass
class DedupFilterStage(ProcessingStage[ImageBatch, ImageBatch]):
    """Filter stage that removes images whose IDs appear in a Parquet file.

    The Parquet file must contain a column with image identifiers; by default this
    column is assumed to be ``id`` to match writer metadata. You can change
    the column name via ``id_column``.
    """

    removal_parquets_dir: str
    id_column: str = "id"
    verbose: bool = False

    _name: str = "image_dedup_filter"

    # Internal cache
    _ids_to_remove: set[str] = field(default_factory=set)

    @property
    def resources(self) -> Resources:
        # CPU-only filter
        return Resources()

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

        for removal_parquet in removal_parquets:
            table = pq.read_table(removal_parquet, columns=[self.id_column])

            # Convert to set of strings
            ids_array = table[self.id_column].to_pylist()
            # Add to set of ids to remove
            self._ids_to_remove.update(ids_array)

        if self.verbose:
            logger.info(
                f"Loaded {len(self._ids_to_remove)} IDs to remove from '{self.removal_parquets_dir}'"
            )

    def process(self, task: ImageBatch) -> ImageBatch:
        original_count = len(task.data)
        ids_to_remove = self._ids_to_remove  # local reference for faster lookups
        filtered_images = [img for img in task.data if img.image_id not in ids_to_remove]

        removed_count = original_count - len(filtered_images)
        if self.verbose:
            logger.info(
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
