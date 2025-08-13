from dataclasses import dataclass, field
from typing import Literal

from loguru import logger

from ray_curator.tasks import Task


@dataclass
class BatchedFileGroupTask(Task[list[list[str]]]):
    """Task representing groups of files.

    This task contains multiple groups of files, where each group is a list of file paths.
    The data structure is list[list[str]] where:
    - Outer list: Contains multiple file groups
    - Inner list: Contains file paths within each group

    This is typically used in deduplication pipelines where all of files within a partition can't be read
    together and need to be read in batches.
    """

    data: list[list[str]] = field(default_factory=list)
    filetype: Literal["jsonl", "parquet"] = field(default="parquet")

    @property
    def num_items(self) -> int:
        """Number of file groups in this task."""
        return len(self.data)

    @property
    def total_files(self) -> int:
        """Total number of files across all groups."""
        return sum(len(group) for group in self.data)

    def validate(self) -> bool:
        """Validate the task data."""
        if len(self.data) == 0:
            logger.warning(f"No file groups to process in task {self.task_id}")
            return False

        # Check that all groups contain at least one file
        for i, group in enumerate(self.data):
            if len(group) == 0:
                logger.warning(f"Empty file group {i} in task {self.task_id}")
                return False

        return True
