import pandas as pd
from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import AudioBatch, DocumentBatch


class AudioToDocumentStage(ProcessingStage[AudioBatch, DocumentBatch]):
    """
    Stage to conver DocumentObject to DocumentBatch

    """

    def process(self, task: AudioBatch) -> list[DocumentBatch]:
        return [
            DocumentBatch(
                data=pd.DataFrame(task.data),
                task_id=task.task_id,
                dataset_name=task.dataset_name,
            )
        ]
