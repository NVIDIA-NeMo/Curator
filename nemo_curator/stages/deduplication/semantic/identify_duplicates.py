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
from dataclasses import dataclass
from typing import Any

import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.io.writer.utils import get_deterministic_hash
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import check_disallowed_kwargs


@dataclass
class IdentifyDuplicatesStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Stage for batch removal of similar documents with optional ID-based partitioning.
    It is a CPU-only stage.
    """

    # Required parameters
    output_path: str
    eps: float

    # Optional ID parameters
    _num_row_groups_hint: int | None = None

    # Optional parameters
    verbose: bool = False
    read_kwargs: dict[str, Any] | None = None
    write_kwargs: dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()
        self._name = "RemovalStage"

        self._batch_size = 10  # We want to load multiple clusters at once

        # Storage options
        self.read_kwargs = self.read_kwargs.copy() if self.read_kwargs is not None else {}
        self.write_kwargs = self.write_kwargs.copy() if self.write_kwargs is not None else {}

        check_disallowed_kwargs(self.read_kwargs, ["filters", "engine"])
        check_disallowed_kwargs(self.write_kwargs, ["index", "row_group_size"])

        self.input_storage_options = self.read_kwargs.pop("storage_options", None) if self.read_kwargs else None
        self.output_storage_options = self.write_kwargs.pop("storage_options", None) if self.write_kwargs else None

        # break each file into 10 row groups by default
        self._num_row_groups_hint = self._num_row_groups_hint if self._num_row_groups_hint is not None else 10

    def process(self, task: FileGroupTask) -> FileGroupTask:
        msg = "RemovalStage does not support single task processing"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[FileGroupTask]) -> list[FileGroupTask]:
        """Process a batch of tasks and combine results into fewer output files.

        This allows processing multiple clusters together and optionally partitioning
        by ID ranges for more efficient reading.

        Args:
            tasks: List of FileGroupTask containing pairwise similarity results

        Returns:
            List of FileGroupTask with combined filtered results
        """
        # Validate all input tasks
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task!s} failed validation for stage {self}"
                raise ValueError(msg)

        if len(tasks) == 0:
            return []

        all_files = [file for task in tasks for file in task.data]
        # Read using filters
        df: pd.DataFrame = pd.read_parquet(
            all_files,
            storage_options=self.input_storage_options,
            **self.read_kwargs,
            filters=[("cosine_sim_score", ">=", 1.0 - self.eps)],
            engine="pyarrow",
        )[["id"]]  # TODO: If we want we can add other columns
        # Write out sorted and with multiple row groups
        df.sort_values("id", inplace=True)  # noqa: PD002

        output_file = os.path.join(self.output_path, get_deterministic_hash(all_files, tasks[0].task_id) + ".parquet")

        df.to_parquet(
            output_file,
            storage_options=self.output_storage_options,
            index=False,
            row_group_size=max(1, len(df) // self._num_row_groups_hint),
            **self.write_kwargs,
        )

        # Create output task
        return [
            FileGroupTask(
                task_id=f"identify_duplicates_{get_deterministic_hash(all_files, tasks[0].task_id)}",
                dataset_name=tasks[0].dataset_name,
                data=[output_file],
                _metadata={**tasks[0]._metadata, "num_removed": len(df)},
                _stage_perf=tasks[0]._stage_perf,
            )
        ]
