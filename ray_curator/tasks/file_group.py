from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .tasks import Task


@dataclass
class FileGroupTask(Task[list[str]]):
    """Task representing a group of files to be read.
    This is created during the planning phase and passed to reader stages.
    """

    reader_config: dict[str, Any] = field(default_factory=dict)
    data: list[str] = field(default_factory=list)

    @property
    def num_items(self) -> int:
        """Number of files in this group."""
        return len(self.data)

    def validate(self) -> bool:
        """Validate the task data."""
        # TODO: We should fsspec checks for that file paths do exist
        # Handle both Python lists and numpy arrays by checking length
        # instead of boolean evaluation (which is ambiguous for numpy arrays)
        if len(self.data) == 0:
            logger.warning(f"No files to process in task {self.task_id}")
            return False
        return True
