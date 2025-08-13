"""File partitioning stage for deduplication workflows."""

from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger

from ray_curator.backends.base import WorkerMetadata
from ray_curator.backends.experimental.utils import RayStageSpecKeys, get_available_cpu_gpu_resources
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import _EmptyTask
from ray_curator.utils.file_utils import get_all_files_paths_under

from .file_task import BatchedFileGroupTask


class KMeansFilePartitioningStage(ProcessingStage[_EmptyTask, BatchedFileGroupTask]):
    """Stage that partitions input files into KMeansFileGroupTasks for deduplication.

    This stage takes an EmptyTask as input and outputs num_output_partitions KMeansFileGroupTasks.
    Each KMeansFileGroupTask contains a list[list[str]] where each inner list contains
    N/num_output_partitions files.

    For JSON files: Each partition contains exactly one group with N/num_output_partitions files.
    For Parquet files: Currently behaves like JSON, but will be enhanced in the future to
    ensure no single group has more than a dynamically computed number of rows.
    """

    # Input parameters
    def __init__(
        self,
        file_paths: str | list[str],
        filetype: Literal["jsonl", "parquet"] = "parquet",
        embedding_dim: int | None = None,
        num_output_partitions: int | None = None,
        file_extensions: list[str] | None = None,
        storage_options: dict[str, Any] | None = None,
        limit: int | None = None,
    ):
        # Initialize attributes that would have been set by FilePartitioningStage
        self.file_paths = file_paths
        self.storage_options = storage_options if storage_options is not None else {}
        self.limit = limit
        self.filetype = filetype
        self.num_output_partitions = num_output_partitions
        self._name = "file_partitioning"
        self.embedding_dim = embedding_dim

        if self.filetype == "parquet":
            self.file_extensions = [".parquet"]
        else:
            self.file_extensions = [".jsonl", ".json"]

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, _: WorkerMetadata | None = None) -> None:
        """Setup the stage."""
        if self.num_output_partitions is None:
            # Compute the number of output partitions based on the number of gpus
            self.num_output_partitions = get_available_cpu_gpu_resources()[1]

    @property
    def resources(self) -> Resources:
        """Resource requirements for this stage."""
        return Resources(cpus=0.5)

    def ray_stage_spec(self) -> dict[str, Any]:
        """Ray stage specification for this stage."""
        return {
            RayStageSpecKeys.IS_FANOUT_STAGE: True,
        }

    def process(self, _: _EmptyTask) -> list[BatchedFileGroupTask]:
        """Process the EmptyTask to create KMeansFileGroupTasks.

        Args:
            task: EmptyTask input (ignored, used for triggering the stage)

        Returns:
            List of KMeansFileGroupTask, each containing partitioned file groups
        """
        # Get list of files
        files = self._get_file_list()
        logger.info(f"Found {len(files)} {self.filetype} files")

        if not files:
            logger.warning(f"No files found matching pattern: {self.file_paths}")
            return []

        # Handle case where we have more partitions than files
        if len(files) < self.num_output_partitions:
            logger.warning(
                f"Number of files ({len(files)}) is less than num_output_partitions "
                f"({self.num_output_partitions}). Will create {len(files)} partitions instead."
            )
            self.num_output_partitions = len(files)

        partitioned_groups = [partition.tolist() for partition in np.array_split(files, self.num_output_partitions)]
        if self.filetype == "parquet":
            partitioned_groups = self._break_parquet_partitions_into_groups(
                self.embedding_dim, partitioned_groups, self.storage_options
            )

        # Create KMeansFileGroupTask for each partition
        tasks = []
        dataset_name = self._get_dataset_name(files)

        for i, file_groups in enumerate(partitioned_groups):
            if self.limit is not None and len(tasks) >= self.limit:
                logger.info(f"Reached limit of {self.limit} partitions")
                break

            kmeans_task = BatchedFileGroupTask(
                task_id=f"kmeans_partition_{i}",
                dataset_name=dataset_name,
                data=file_groups,
                filetype=self.filetype,
            )
            tasks.append(kmeans_task)

        return tasks

    def _get_file_list(self) -> list[str]:
        """Get the list of files to process."""
        logger.debug(f"Getting file list for {self.file_paths}")
        if isinstance(self.file_paths, str):
            # TODO: This needs to change for fsspec
            path = Path(self.file_paths)
            if path.is_file():
                out = [str(path)]
            else:
                # Directory or pattern
                out = get_all_files_paths_under(
                    self.file_paths,
                    recurse_subdirectories=True,
                    keep_extensions=self.file_extensions,
                    storage_options=self.storage_options,
                )
        else:
            # List of files
            out = self.file_paths

        return out

    def _get_dataset_name(self, files: list[str]) -> str:
        """Extract dataset name from file paths."""
        if files:
            # Use the parent directory name or first file stem
            # TODO: This needs to change for fsspec
            first_file = Path(files[0])
            if first_file.parent.name and first_file.parent.name != ".":
                return first_file.parent.name
            else:
                return first_file.stem
        return "dataset"

    @staticmethod
    def _break_parquet_partitions_into_groups(
        embedding_dim: int | None,
        partitioned_groups: list[list[str]],
        storage_options: dict[str, Any],
    ) -> list[list[list[str]]]:
        if embedding_dim is None:
            # As a default we just go with an aggressive default of 1024 dimensional embedding
            embedding_dim = 1024

        cudf_max_num_rows = 2_000_000_000  # cudf only allows 2bn rows
        cudf_max_num_elements = (
            cudf_max_num_rows / embedding_dim
        )  # cudf considers each element in an array to be a row

        # for simiplicity we just load the first file and get the number of rows
        # to adjust for skew we multiply this by 1.5
        import pyarrow.parquet as pq
        from fsspec.parquet import open_parquet_file

        with open_parquet_file(partitioned_groups[0][0], storage_options=storage_options) as f:
            avg_num_rows = pq.read_metadata(f).num_rows * 1.5

        max_files_per_subgroup = int(cudf_max_num_elements / avg_num_rows)
        max_files_per_subgroup = max(1, max_files_per_subgroup)  # Ensure at least 1 file per subgroup

        out = []
        for partition in partitioned_groups:
            subgroups = [
                partition[i : i + max_files_per_subgroup] for i in range(0, len(partition), max_files_per_subgroup)
            ]
            out.append(subgroups)

        logger.info(
            f"Broke {len(partitioned_groups)} partitions into subgroups with max {max_files_per_subgroup} files per subgroup"
        )
        return out
