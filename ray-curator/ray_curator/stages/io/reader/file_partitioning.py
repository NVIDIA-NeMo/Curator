"""JSONL reader composite stage."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from ray_curator.backends.experimental.ray_data.utils import RayStageSpecKeys
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import FileGroupTask, _EmptyTask
from ray_curator.utils.file_utils import get_all_files_paths_under


@dataclass
class FilePartitioningStage(ProcessingStage[_EmptyTask, FileGroupTask]):
    """Stage that partitions input file paths into FileGroupTasks.

    This stage runs as a dedicated processing stage (not on the driver)
    and creates file groups based on the partitioning strategy.
    """

    file_paths: str | list[str]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    file_extensions: list[str] | None = None
    storage_options: dict[str, Any] | None = None
    limit: int | None = None
    _name: str = "file_partitioning"

    def __post_init__(self):
        """Initialize default values."""
        if self.file_extensions is None:
            self.file_extensions = [".jsonl", ".json"]
        if self.storage_options is None:
            self.storage_options = {}

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    @property
    def resources(self) -> Resources:
        """Resource requirements for this stage."""
        return Resources(cpus=0.5)

    def ray_stage_spec(self) -> dict[str, Any]:
        """Ray stage specification for this stage."""
        return {
            RayStageSpecKeys.IS_FANOUT_STAGE: True,
        }

    def process(self, _: _EmptyTask) -> list[FileGroupTask]:
        """Process the initial task to create file group tasks.

        This stage expects a simple Task with file paths information
        and outputs multiple FileGroupTasks for parallel processing.
        """
        # Get list of files
        files = self._get_file_list()
        logger.info(f"Found {len(files)} files")
        if not files:
            logger.warning(f"No files found matching pattern: {self.file_paths}")
            return []

        # Partition files
        if self.files_per_partition:
            partitions = self._partition_by_count(files, self.files_per_partition)
        elif self.blocksize:
            partitions = self._partition_by_size(files, self.blocksize)
        else:
            # All files in one group
            partitions = [files]

        # Create FileGroupTask for each partition
        tasks = []
        dataset_name = self._get_dataset_name(files)

        for i, file_group in enumerate(partitions):
            if self.limit is not None and len(tasks) >= self.limit:
                logger.info(f"Reached limit of {self.limit} file groups")
                break
            file_task = FileGroupTask(
                task_id=f"file_group_{i}",
                dataset_name=dataset_name,
                data=file_group,
                _metadata={
                    "partition_index": i,
                    "total_partitions": len(partitions),
                    "storage_options": self.storage_options,
                    "source_files": file_group,  # Add source files for deterministic naming during write stage
                },
                reader_config={},  # Empty - will be populated by reader stage
            )
            tasks.append(file_task)

        logger.info(f"Created {len(tasks)} file groups from {len(files)} files")
        return tasks

    def _get_file_list(self) -> list[str]:
        """Get the list of files to process."""
        logger.info(f"Getting file list for {self.file_paths}")
        if isinstance(self.file_paths, str):
            # TODO: This needs to change for fsspec
            path = Path(self.file_paths)
            if path.is_file():
                return [str(path)]
            else:
                # Directory or pattern
                return get_all_files_paths_under(
                    self.file_paths,
                    recurse_subdirectories=True,
                    keep_extensions=self.file_extensions,
                    storage_options=self.storage_options,
                )
        else:
            # List of files
            return self.file_paths

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

    def _partition_by_count(self, files: list[str], count: int) -> list[list[str]]:
        """Partition files by count."""
        partitions = []
        for i in range(0, len(files), count):
            partitions.append(files[i : i + count])
        return partitions

    def _partition_by_size(self, files: list[str], blocksize: int | str) -> list[list[str]]:
        """Partition files by target size.

        Note: This is a simplified implementation. A full implementation
        would check actual file sizes and create balanced partitions.
        """
        # Convert blocksize to bytes if string
        if isinstance(blocksize, str):
            blocksize = self._parse_size(blocksize)

        # TODO: We should calculate the average file size
        avg_file_size = 100 * 1024 * 1024  # 100MB
        files_per_block = max(1, blocksize // avg_file_size)

        return self._partition_by_count(files, files_per_block)

    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '128MB' to bytes."""
        size_str = size_str.upper().strip()

        # Check units in order from longest to shortest to avoid partial matches
        units = [
            ("TB", 1024 * 1024 * 1024 * 1024),
            ("GB", 1024 * 1024 * 1024),
            ("MB", 1024 * 1024),
            ("KB", 1024),
            ("B", 1),
        ]

        for unit, multiplier in units:
            if size_str.endswith(unit):
                number = float(size_str[: -len(unit)])
                return int(number * multiplier)

        # If no unit, assume bytes
        return int(size_str)
