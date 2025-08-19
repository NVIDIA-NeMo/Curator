import pandas as pd

from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import DataObject, DocumentBatch


class ObjectToBatchStage(ProcessingStage[DataObject, DocumentBatch]):
    """
    Stage to conver DocumentObject to DocumentBatch

    """

    def process(self, task: DataObject) -> list[DocumentBatch]:
        return [
            DocumentBatch(
                data=pd.DataFrame(task.data),
                task_id="",
                dataset_name="ObjectToBatch",
            )
        ]
