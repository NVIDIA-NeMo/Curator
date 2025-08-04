import os
from dataclasses import dataclass

from .tasks import Task


@dataclass
class SpeechObject(Task[dict]):
    """A wrapper for data entry + any additional metrics."""

    def __init__(
        self, data: dict | None = None, metrics: dict | None = None, dataset_name: str = "", task_id: int = 0, **kwargs
    ):
        self.data = data  # data can be None to drop the entry
        self.metrics = metrics
        self.filepath_key = "audio_filepath"
        super().__init__(data=data, task_id=task_id, dataset_name=dataset_name, **kwargs)

    @property
    def num_items(self) -> int:
        return 1

    def validate(self) -> bool:
        """Validate the task data."""
        if not os.path.exists(self.data[self.filepath_key]):
            print(f"Video {self.data[self.filepath_key]} does not exist")
            return False
        return True
