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

from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask, _EmptyTask
from nemo_curator.utils.file_utils import get_all_file_paths_under, get_fs, infer_dataset_name_from_path

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


class ClusterWiseFilePartitioningStage(ProcessingStage[_EmptyTask, FileGroupTask]):
    """Stage that partitions input files into PairwiseFileGroupTasks for deduplication.

    This stage takes an EmptyTask as input and outputs partition-aware file groups.
    It reads parquet files partitioned by centroid (from kmeans output) and creates
    one PairwiseFileGroupTask per centroid partition.
    """

    def __init__(
        self,
        input_path: str,
        storage_options: dict[str, Any] | None = None,
    ):
        """Initialize the partitioning stage.

        Args:
            input_path: Path to the kmeans output directory containing centroid partitions
            storage_options: Storage options for reading files
            limit: Maximum number of partitions to process
        """
        self.input_path = input_path
        self.storage_options = storage_options
        self._name = "pairwise_file_partitioning"
        self._resources = Resources(cpus=0.5)
        self.fs: AbstractFileSystem | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self.fs = get_fs(self.input_path, storage_options=self.storage_options)

    def ray_stage_spec(self) -> dict[str, Any]:
        """Ray stage specification for this stage."""
        return {
            RayStageSpecKeys.IS_FANOUT_STAGE: True,
        }

    def process(self, _: _EmptyTask) -> list[FileGroupTask]:
        """Process the EmptyTask to create PairwiseFileGroupTasks.

        Args:
            task: EmptyTask input (ignored, used for triggering the stage)

        Returns:
            List of PairwiseFileGroupTask, each containing partitioned file groups per centroid
        """
        # Get centroid directories from kmeans output
        centroid_dirs = {}
        for entry in self.fs.ls(self.input_path):
            # Extract centroid ID from directory name (e.g., "centroid=0" -> 0)
            if "centroid=" in entry:
                centroid_id = int(entry.split("centroid=")[-1])
                centroid_dirs[centroid_id] = entry

        logger.debug(
            f"Found {len(centroid_dirs)} centroid directories e.g. {next(iter(centroid_dirs.values())) if centroid_dirs else None}"
        )

        if not centroid_dirs:
            logger.warning(f"No centroid directories found in: {self.input_path}")
            return []

        tasks = []
        dataset_name = infer_dataset_name_from_path(self.input_path)

        for centroid_id, centroid_dir in centroid_dirs.items():
            partition_files = get_all_file_paths_under(
                centroid_dir,
                recurse_subdirectories=True,
                keep_extensions=[".parquet"],
                fs=self.fs,
            )
            pairwise_task = FileGroupTask(
                task_id=f"pairwise_centroid_{centroid_id}",
                dataset_name=dataset_name,
                data=partition_files,
                _metadata={
                    "centroid_id": centroid_id,
                    "filetype": "parquet",
                },
            )
            tasks.append(pairwise_task)

        logger.debug(f"Created {len(tasks)} pairwise tasks")
        return tasks
