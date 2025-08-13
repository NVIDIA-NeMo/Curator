from typing import TYPE_CHECKING, Any

from loguru import logger

from ray_curator.backends.base import WorkerMetadata
from ray_curator.backends.experimental.utils import RayStageSpecKeys
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import _EmptyTask
from ray_curator.utils.file_utils import get_all_files_paths_under, get_fs, infer_dataset_name_from_path

from .file_task import BatchedFileGroupTask
from .utils import break_parquet_partition_into_groups

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


class PairwiseFilePartitioningStage(ProcessingStage[_EmptyTask, BatchedFileGroupTask]):
    """Stage that partitions input files into PairwiseFileGroupTasks for deduplication.

    This stage takes an EmptyTask as input and outputs partition-aware file groups.
    It reads parquet files partitioned by centroid (from kmeans output) and creates
    one PairwiseFileGroupTask per centroid partition.
    """

    def __init__(
        self,
        input_path: str,
        embedding_dim: int | None = None,
        storage_options: dict[str, Any] | None = None,
    ):
        """Initialize the partitioning stage.

        Args:
            input_path: Path to the kmeans output directory containing centroid partitions
            embedding_dim: Dimension of embeddings for memory calculation
            storage_options: Storage options for reading files
            limit: Maximum number of partitions to process
        """
        self.input_path = input_path
        self.embedding_dim = embedding_dim
        self.storage_options = storage_options if storage_options is not None else {}
        self._name = "pairwise_file_partitioning"
        self.fs: AbstractFileSystem | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self.fs = get_fs(self.input_path, storage_options=self.storage_options)

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
            partition_files = get_all_files_paths_under(
                centroid_dir,
                recurse_subdirectories=True,
                keep_extensions=[".parquet"],
                fs=self.fs,
            )
            # Break files into subgroups to avoid 2bn row limit
            file_groups = break_parquet_partition_into_groups(
                partition_files, embedding_dim=self.embedding_dim, storage_options=self.storage_options
            )

            pairwise_task = BatchedFileGroupTask(
                task_id=f"pairwise_centroid_{centroid_id}",
                dataset_name=dataset_name,
                data=file_groups,
                filetype="parquet",
                _metadata={
                    "centroid_id": centroid_id,
                },
            )
            tasks.append(pairwise_task)

        logger.debug(f"Created {len(tasks)} pairwise tasks")
        return tasks
