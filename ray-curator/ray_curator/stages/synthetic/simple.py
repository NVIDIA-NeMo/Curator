"""
This module contains a simple stage for generating synthetic data. It takes in Empty task and a prompt and produces the output in form of a DocumentBatch.
"""

import pandas as pd

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.services.model_client import LLMClient
from ray_curator.tasks import DocumentBatch, _EmptyTask


class SimpleSyntheticStage(ProcessingStage[_EmptyTask, DocumentBatch]):
    """
    A simple stage for generating synthetic data. It takes in Empty task and a prompt and produces the output in form of a DocumentBatch.
    """

    def __init__(self, prompt: str, client: LLMClient, model_name: str):
        self.prompt = prompt
        self.client = client
        self.model_name = model_name

    @property
    def name(self) -> str:
        return "SimpleSyntheticStage"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["text"]

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self.client.setup()

    def process(self, _: _EmptyTask) -> DocumentBatch:
        response = self.client.query_model(
            model=self.model_name,
            messages=[{"role": "user", "content": self.prompt}],
        )

        return DocumentBatch(data=pd.DataFrame({"text": response}), dataset_name="simple_synthetic_data", task_id=1)
