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

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import ImageBatch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np

    from nemo_curator.backends.base import WorkerMetadata


@dataclass
class ImageScoringStage(ProcessingStage[ImageBatch, ImageBatch]):
    """Annotate scalar model scores without filtering or losing failed inputs.

    ``predictor_factory`` constructs a callable once per worker. The callable
    receives a list of image arrays and returns one finite scalar per array.
    It owns model preprocessing and device/precision choices. Model exceptions
    propagate so the executor can retry; they are not disguised as bad images.

    Batches are formed before missing pixels are excluded, preserving inference
    boundaries even when decoding failed. Configure CPU/GPU allocation with
    ``with_(resources=Resources(...))``; the default reserves one CPU only.
    """

    predictor_factory: Callable[[], Callable[[list[np.ndarray]], Sequence[float]]]
    score_field: str = "image_score"
    status_field: str = "image_score_status"
    model_inference_batch_size: int = 128
    remove_image_data: bool = False
    name: str = "image_scoring"
    resources = Resources(cpus=1.0)

    def __post_init__(self) -> None:
        if self.model_inference_batch_size <= 0:
            message = "model_inference_batch_size must be positive"
            raise ValueError(message)
        if not self.score_field or not self.status_field or self.score_field == self.status_field:
            message = "Score and status fields must be distinct nonempty names"
            raise ValueError(message)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        self._predictor = self.predictor_factory()

    def process(self, task: ImageBatch) -> ImageBatch:
        for start in range(0, len(task.data), self.model_inference_batch_size):
            batch = task.data[start : start + self.model_inference_batch_size]
            valid = [image for image in batch if image.image_data is not None]
            scores = (
                [float(score) for score in self._predictor([image.image_data for image in valid])] if valid else []
            )
            if len(scores) != len(valid) or not all(math.isfinite(score) for score in scores):
                message = "Image predictor must return one finite scalar per valid image"
                raise ValueError(message)
            for image in batch:
                image.metadata.update({self.score_field: None, self.status_field: "decode_error"})
            for image, score in zip(valid, scores, strict=True):
                image.metadata.update({self.score_field: score, self.status_field: "ok"})
            if self.remove_image_data:
                for image in batch:
                    image.image_data = None
        return task
