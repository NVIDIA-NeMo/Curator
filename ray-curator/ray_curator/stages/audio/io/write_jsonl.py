import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ray_curator.backends.base import NodeInfo, WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import DocumentBatch, DocumentObject


class ObjectToBatchStage(ProcessingStage[DocumentObject, DocumentBatch]):
    """
    Stage to conver DocumentObject to DocumentBatch

    """

    def process_batch(self, tasks: list[DocumentObject]) -> list[DocumentBatch]:
        return [
            DocumentBatch(
                data=pd.DataFrame([task.data for task in tasks]),
                task_id="",
                dataset_name="ObjectToBatch",
            )
        ]

    def process(self, _: DocumentObject) -> None:
        pass


@dataclass
class WriteJsonlStage(ProcessingStage[DocumentObject, DocumentObject]):
    """
    Stage for saving tasks as a one JSONL file.

    """

    output_manifest_file: str
    ensure_ascii: bool = False
    encoding: str = "utf8"

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        Path(self.output_manifest_file).parent.mkdir(parents=True, exist_ok=True)
        open(self.output_manifest_file, "w").close()

    def process(self, tasks: DocumentObject) -> DocumentObject:
        with open(self.output_manifest_file, "a", encoding=self.encoding) as f:
            f.write(json.dumps(tasks.data, ensure_ascii=self.ensure_ascii) + "\n")
        return tasks
