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

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.tasks import DocumentBatch


@dataclass
class TextDuplicatesRemovalStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Stage for removing duplicate documents based on pre-computed removal lists.

    Args:
        ids_to_remove_path: Path to parquet files containing IDs to remove
        id_field: Field to use for deduplication within the input dataframe. Defaults to CURATOR_DEDUP_ID_STR.
        duplicate_id_field: Field to use for deduplication within the removal dataframe. Defaults to "id".
        read_kwargs: Additional arguments for reading parquet files
    """

    ids_to_remove_path: str
    id_field: str = CURATOR_DEDUP_ID_STR
    duplicate_id_field: str = "id"

    # Optional parameters
    read_kwargs: dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()
        self._name = "DuplicatesRemovalStage"
        self.read_kwargs = self.read_kwargs.copy() if self.read_kwargs else {}

    def process(self, task: DocumentBatch) -> DocumentBatch:
        """
        Our deduplicator should've written out a parquet file with the IDs to remove.
        We read that file, filter the input dataframe to only include the IDs to remove,
        and return the filtered dataframe.
        We optimize by not loading the whole ids to remove into memory, but only loading the ids that are in the range of the input dataframe.
        """
        df = task.to_pandas()
        input_df_t0 = time.perf_counter()
        min_id = df[self.id_field].min()
        max_id = df[self.id_field].max()
        input_df_min_max_time = time.perf_counter() - input_df_t0
        # Filter the parquet files for IDs to remove within this range
        read_dupes_t0 = time.perf_counter()
        removal_df = pd.read_parquet(
            self.ids_to_remove_path,
            filters=[(self.duplicate_id_field, ">=", min_id), (self.duplicate_id_field, "<=", max_id)],
            columns=[self.duplicate_id_field],
            **self.read_kwargs,
        )
        read_dupes_time = time.perf_counter() - read_dupes_t0

        # Filter out documents with IDs in the removal set using pandas
        time_to_remove_t0 = time.perf_counter()
        removal_ids = set(removal_df[self.duplicate_id_field].tolist())
        df = df[~df[self.id_field].isin(removal_ids)]
        removal_ids_time = time.perf_counter() - time_to_remove_t0
        self._log_metrics(
            {
                "input_df_min_max_time": input_df_min_max_time,
                "read_dupes_time": read_dupes_time,
                "id_removal_time": removal_ids_time,
            }
        )

        # Create output batch with filtered data
        return DocumentBatch(
            task_id=f"removal_{task.task_id}",
            dataset_name=task.dataset_name,
            data=df,
            _metadata={**task._metadata, "num_removed": len(removal_ids)},
            _stage_perf=task._stage_perf,
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.id_field]
