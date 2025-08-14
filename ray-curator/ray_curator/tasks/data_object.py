import os
from dataclasses import dataclass

from loguru import logger

from .tasks import Task


@dataclass
class DataObject(Task[dict]):
    """A single data dict with filepath check."""

    data: dict | None = None
    filepath_key: str | None = None
    task_id: str = ""
    dataset_name: str = ""

    @property
    def num_items(self) -> int:
        return 1

    def validate(self) -> bool:
        """Validate the task data."""
        if self.filepath_key is not None and not os.path.exists(self.data[self.filepath_key]):
            logger.warning(f"File {self.data[self.filepath_key]} does not exist")
            return False
        return True
