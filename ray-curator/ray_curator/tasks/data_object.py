import os
from dataclasses import dataclass

from loguru import logger

from .tasks import Task


@dataclass
class DataObject(Task[dict]):
    """A single data entry with filepath check."""

    def __init__(
        self,
        data: dict | None = None,
        filepath_key: str | None = None,
        dataset_name: str = "",
        task_id: int = 0,
        **kwargs,
    ):
        self.data = data  # data can be None to drop the entry
        self.filepath_key = filepath_key
        super().__init__(data=data, task_id=task_id, dataset_name=dataset_name, **kwargs)

    @property
    def num_items(self) -> int:
        return 1

    def validate(self) -> bool:
        """Validate the task data."""
        if self.filepath_key is not None and not os.path.exists(self.data[self.filepath_key]):
            logger.warning(f"File {self.data[self.filepath_key]} does not exist")
            return False
        return True
