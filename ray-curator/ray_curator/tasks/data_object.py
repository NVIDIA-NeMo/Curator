import os
from dataclasses import dataclass

from loguru import logger

from .tasks import Task


@dataclass
class DataObject(Task[dict]):
    """A single data dict with filepath check."""

    def __init__(
        self,
        data: dict | list[dict] | None = None,
        filepath_key: str | None = None,
        task_id: str = "",
        dataset_name: str = "",
        **kwargs,
    ):
        if isinstance(data, dict):
            self.data = [data]
        elif isinstance(data, list) or data is None:
            self.data = data
        else:
            raise ValueError(str(data))
        self.filepath_key = filepath_key
        super().__init__(data=self.data, task_id=task_id, dataset_name=dataset_name, **kwargs)

    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate_item(self, item: dict) -> bool:
        if self.filepath_key is not None and not os.path.exists(item[self.filepath_key]):
            logger.warning(f"File {item[self.filepath_key]} does not exist")
            return False
        else:
            return True

    def validate(self) -> bool:
        """Validate the task data."""
        return all(self.validate_item(item) for item in self.data)
