# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.image.filters.base import BaseFilterStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import ImageBatch


def _sharpness_score(image: np.ndarray) -> float:
    """Compute Laplacian variance as sharpness score; higher is sharper."""
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


@dataclass
class ImageBlurFilterStage(BaseFilterStage):
    """Stage for filtering out blurry images using Laplacian variance sharpness.

    Expects ImageObject.image_data to be set (e.g. by the image reader stage).
    Computes a sharpness score per image (higher = sharper). Images with scores
    below the threshold are filtered out.
    """

    model_dir: str = None
    num_gpus_per_worker: float = 0.01
    model_inference_batch_size: int = 32
    score_threshold: float = 100.0
    verbose: bool = False
    name: str = "image_blur_filter"

    def __post_init__(self) -> None:
        self.resources = Resources()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self.verbose:
            logger.info("ImageBlurFilterStage ready (CPU sharpness detection)")

    def process(self, task: ImageBatch) -> ImageBatch:
        for batch in self.yield_next_batch(task):
            for image_obj in batch:
                image_obj.sharpness_score = _sharpness_score(image_obj.image_data)
                logger.info(f"Sharpness score: {image_obj.sharpness_score}")

            if self.verbose:
                logger.info(f"Computed sharpness for {len(batch)} images in batch")

        filtered_images = []
        filtered_count = 0
        for image_obj in task.data:
            if image_obj.sharpness_score >= self.score_threshold:
                filtered_images.append(image_obj)
            else:
                filtered_count += 1
                if self.verbose:
                    logger.info(
                        f"Image {image_obj.image_id} (path: {image_obj.image_path}) has sharpness "
                        f"{image_obj.sharpness_score:.1f} below threshold "
                        f"{self.score_threshold}, filtered out as blurry."
                    )

        if self.verbose:
            logger.info(
                f"Blur filtering: {len(filtered_images)}/{len(task.data)} images passed, {filtered_count} filtered out"
            )

        return ImageBatch(
            data=filtered_images,
            dataset_name=task.dataset_name,
            task_id=f"{task.task_id}_{self.name}",
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


__all__ = ["ImageBlurFilterStage"]
