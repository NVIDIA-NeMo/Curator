# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Generator
from dataclasses import dataclass

import torch

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import ImageBatch, ImageObject


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
    _name: str = "image_filter"

    def __post_init__(self) -> None:
        if torch.cuda.is_available():
            self._resources = Resources(gpus=self.num_gpus_per_worker)
        else:
            self._resources = Resources()

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Initialize the base filter stage."""
        raise NotImplementedError

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
        raise NotImplementedError


# Explicitly export the class
__all__ = ["BaseFilterStage"]
