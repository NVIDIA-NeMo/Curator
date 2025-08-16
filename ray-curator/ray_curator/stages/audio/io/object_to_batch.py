import pandas as pd

from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import DataObject, DocumentBatch


class ObjectToBatchStage(ProcessingStage[DataObject, DocumentBatch]):
    """
    Stage to conver DocumentObject to DocumentBatch

    """

    def process_batch(self, tasks: list[DataObject]) -> list[DocumentBatch]:
        data = []
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task!s} failed validation for stage {self}"
                raise ValueError(msg)
            data.append(task.data)

        return [
            DocumentBatch(
                data=pd.DataFrame(data),
                task_id="",
                dataset_name="ObjectToBatch",
            )
        ]

    def process(self, _: DataObject) -> None:
        pass
