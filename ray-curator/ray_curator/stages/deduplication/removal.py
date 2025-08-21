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

"""
Removal stage for distributed deduplication pipeline.

This stage implements the removal phase of the distributed deduplication approach:
1. Takes a DocumentBatch and determines the min/max ID range
2. Filters the parquet files for IDs to remove within this range
3. Filters out documents based on the removal list
4. Returns the filtered DocumentBatch
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from ray_curator.stages.resources import Resources
from ray_curator.tasks import DocumentBatch


@dataclass
class RemovalStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Stage for removing duplicate documents based on pre-computed removal lists.

    Args:
        ids_to_remove_path: Path to parquet files containing IDs to remove
        verbose: Whether to print verbose output
        read_kwargs: Additional arguments for reading parquet files
    """

    # Required parameters
    ids_to_remove_path: str

    id_field: str = CURATOR_DEDUP_ID_STR

    # Optional parameters
    verbose: bool = False
    read_kwargs: dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()
        self._name = "RemovalStage"

        # CPU-only stage for maximal parallelism
        self._resources = Resources(cpus=1.0, gpus=0.0)
        self._batch_size = 1  # Process one batch at a time

        # Storage options
        self.read_kwargs = self.read_kwargs if self.read_kwargs is not None else {}

    def process(self, task: DocumentBatch) -> DocumentBatch:
        """Process a DocumentBatch to remove duplicates."""
        df = task.to_pandas()

        min_id = df[self.id_field].min()
        max_id = df[self.id_field].max()

        # Filter the parquet files for IDs to remove within this range
        removal_df = pd.read_parquet(
            self.ids_to_remove_path,
            filters=[("id", ">=", min_id), ("id", "<=", max_id)],
            columns=["id"],
            **self.read_kwargs,
            storage_options=self.read_kwargs.get("storage_options") if self.read_kwargs else None,
        )
        removal_ids = set(removal_df["id"].tolist())

        # Filter out documents with IDs in the removal set using pandas
        df = df[~df[self.id_field].isin(removal_ids)]

        # Create output batch with filtered data
        return DocumentBatch(
            task_id=f"removal_{task.task_id}",
            dataset_name=task.dataset_name,
            data=df,
            _metadata={**task._metadata, "num_removed": len(removal_ids)},
            _stage_perf=task._stage_perf,
        )
