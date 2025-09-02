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

from typing import Any, Literal

from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.deduplication.shuffle_utils.rapidsmpf_shuffler import BulkRapidsMPFShuffler
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import create_or_overwrite_dir


class ShuffleStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """
    Stage that performs generic shuffling on specified columns from a FileGroupTask.
    This stage uses the BulkRapidsMPFShuffler with cuDF I/O for efficient GPU-based shuffling.

    Parameters
    ----------
    shuffle_on
        List of column names to shuffle on.
    total_nparts
        Total number of output partitions. If None, will be set automatically by the executor.
    output_path
        Path to write output files.
    read_kwargs
        Keyword arguments for cudf.read_parquet method.
    write_kwargs
        Keyword arguments for cudf.to_parquet method.
    rmm_pool_size
        Size of the RMM GPU memory pool in bytes.
        If "auto", the memory pool is set to 90% of the free GPU memory.
        If None, the memory pool is set to 50% of the free GPU memory that can expand if needed.
    spill_memory_limit
        Device memory limit in bytes for spilling to host.
        If "auto", the limit is set to 80% of the RMM pool size.
        If None spilling is disabled.
    enable_statistics
        Whether the underlying rapidsmpf shuffler should collect shuffle statistics.
    """

    _name = "ShuffleStage"
    _resources = Resources(gpus=1.0)

    # Use BulkRapidsMPFShuffler directly
    actor_class = BulkRapidsMPFShuffler

    def __init__(  # noqa: PLR0913
        self,
        shuffle_on: list[str],
        total_nparts: int | None = None,
        output_path: str = "./",
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        rmm_pool_size: int | Literal["auto"] | None = "auto",
        spill_memory_limit: int | Literal["auto"] | None = "auto",
        enable_statistics: bool = False,
    ):
        super().__init__()

        # Initialize instance variables
        self.shuffle_on = shuffle_on
        self.total_nparts = total_nparts
        self.output_path = output_path
        self.rmm_pool_size = rmm_pool_size
        self.spill_memory_limit = spill_memory_limit
        self.enable_statistics = enable_statistics

        self.read_kwargs = read_kwargs if read_kwargs is not None else {}
        self.write_kwargs = write_kwargs if write_kwargs is not None else {}

        self.actor_kwargs = {
            "shuffle_on": self.shuffle_on,
            "total_nparts": self.total_nparts,  # Can be None, executor will set it
            "output_path": self.output_path,
            "rmm_pool_size": self.rmm_pool_size,
            "spill_memory_limit": self.spill_memory_limit,
            "enable_statistics": self.enable_statistics,
            "read_kwargs": self.read_kwargs,
            "write_kwargs": self.write_kwargs,
        }
        # Handle output path
        create_or_overwrite_dir(self.output_path, storage_options=self.write_kwargs.get("storage_options", {}))

    def process(self, task: FileGroupTask) -> FileGroupTask:
        """Not implemented for actor-based stages."""
        msg = "ShufflerStage does not support the process method. Use with an actor-based executor."
        raise NotImplementedError(msg)

    def ray_stage_spec(self) -> dict[str, Any]:
        """Ray stage specification for this stage."""
        return {
            RayStageSpecKeys.IS_SHUFFLE_STAGE: True,
        }

    def _check_actor_obj(self) -> None:
        """Verify the actor object is properly initialized."""
        if not hasattr(self, "_actor_obj") or not isinstance(self._actor_obj, self.actor_class):
            msg = (
                "Actor object not initialized. This might be because an incorrect executor "
                "was used or it failed to setup the stage properly."
            )
            raise RuntimeError(msg)

    def read_and_insert(self, task: FileGroupTask) -> FileGroupTask:
        """Read files and insert into shuffler."""
        self._check_actor_obj()
        self.output_columns = self._actor_obj.read_and_insert(task.data)
        self.dataset_name = task.dataset_name
        return task

    def insert_finished(self) -> None:
        self._check_actor_obj()
        self._actor_obj.insert_finished()

    def extract_and_write(self) -> list[FileGroupTask]:
        self._check_actor_obj()
        partition_paths = self._actor_obj.extract_and_write(column_names=self.output_columns)
        return [
            FileGroupTask(
                task_id=partition_id,
                dataset_name=self.dataset_name + f"{self.name}",
                data=[path],
                _metadata={
                    "partition_index": partition_id,
                    "total_partitions": len(partition_paths),
                    "output_columns": self.output_columns,
                },
            )
            for partition_id, path in partition_paths
        ]

    def teardown(self) -> None:
        self._check_actor_obj()
        self._actor_obj.cleanup()
