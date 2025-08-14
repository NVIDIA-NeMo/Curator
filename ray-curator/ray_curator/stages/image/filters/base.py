from dataclasses import dataclass
from typing import Generator

import numpy as np
import torch
from loguru import logger

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import ImageBatch, ImageObject


@dataclass
class BaseFilterStage(ProcessingStage[ImageBatch, ImageBatch]):
    """Base class for image filtering stages.

    This class provides a base class for image filtering stages.
    """
    model_dir: str = None
    num_gpus_per_worker: float = 0.25
    model_inference_batch_size: int = 32  # Number of images to process through model at once
    score_threshold: float = 0.5
    verbose: bool = False

    @property
    def name(self) -> str:
        return "image_filter"

    @property
    def resources(self) -> Resources:
        return Resources(gpus=self.num_gpus_per_worker)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        """Initialize the base filter stage."""
        pass

    def yield_next_batch(self, task: ImageBatch) -> Generator[list[ImageObject], None, None]:
        """
        Yields a generator of model inputs for the next batch.

        Args:
            task (ImageBatch): The ImageBatch to process.

        Yields:
            Generator[dict[str, torch.Tensor]]: A generator of model inputs for the next batch.

        """
        for i in range(0, len(task.data), self.model_inference_batch_size):
            yield task.data[i : i + self.model_inference_batch_size]

    def process(self, task: ImageBatch) -> ImageBatch:
        """Process an image batch to generate scores and filter by threshold.

        Args:
            task: ImageBatch containing list of ImageObject instances with pre-computed embeddings

        Returns:
            ImageBatch with filtered images that have scores below the threshold
        """
        pass


# Explicitly export the class
__all__ = ["BaseFilterStage"]
