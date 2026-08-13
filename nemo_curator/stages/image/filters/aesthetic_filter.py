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

from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.models.aesthetics import AestheticScorer
from nemo_curator.stages.image.filters.base import BaseFilterStage
from nemo_curator.tasks import ImageBatch, ImageObject


@dataclass
class ImageAestheticFilterStage(BaseFilterStage):
    """Stage for filtering out images based on aesthetic scores.

    This class processes image batches through an aesthetic scoring model to generate
    aesthetic scores for each image. Images with scores below the threshold will be filtered out.
    """

    model_dir: str = None
    num_gpus_per_worker: float = 0.25
    model_inference_batch_size: int = 32  # Number of images to process through model at once
    score_threshold: float = 0.5
    verbose: bool = False
    batch_size: int = 1  # Number of ImageBatch tasks processed per executor call
    name: str = "image_aesthetic_filter"

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        """Download aesthetic model weights from HF"""
        AestheticScorer.download_weights_on_node(self.model_dir)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Initialize the aesthetic filtering model."""
        self.model = AestheticScorer(model_dir=self.model_dir)
        self.model.setup()

        if self.verbose:
            logger.info("Initialized aesthetic scoring model")

    def _score_images(self, images: list[ImageObject]) -> None:
        """Score images across task boundaries using full model batches."""
        for start in range(0, len(images), self.model_inference_batch_size):
            batch = images[start : start + self.model_inference_batch_size]
            embeddings = [img_obj.embedding for img_obj in batch]
            batch_tensor = np.stack(embeddings, axis=0)

            with torch.no_grad():
                scores = self.model(batch_tensor).cpu().numpy()

            for i, image_obj in enumerate(batch):
                image_obj.aesthetic_score = float(scores[i])

    def _filter_task(self, task: ImageBatch) -> ImageBatch:
        """Filter one task after scores have been assigned."""
        filtered_images = []
        for image_obj in task.data:
            if image_obj.aesthetic_score >= self.score_threshold:
                filtered_images.append(image_obj)
            elif self.verbose:
                logger.info(
                    f"Image {image_obj.image_id} (path: {image_obj.image_path}) has aesthetic score "
                    f"{image_obj.aesthetic_score:.3f} below threshold {self.score_threshold}, filtered out."
                )
        filtered_count = len(task.data) - len(filtered_images)

        if self.verbose:
            logger.info(
                f"Aesthetic filtering: {len(filtered_images)}/{len(task.data)} images passed, "
                f"{filtered_count} filtered out"
            )

        return ImageBatch(
            data=filtered_images,
            dataset_name=task.dataset_name,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

    def process(self, task: ImageBatch) -> ImageBatch:
        """Process an image batch to filter by aesthetic score threshold.

        Args:
            task: ImageBatch containing list of ImageObject instances with aesthetic scores

        Returns:
            ImageBatch with filtered images that meet the aesthetic score threshold.
        """

        self._score_images(task.data)
        return self._filter_task(task)

    def process_batch(self, tasks: list[ImageBatch]) -> list[ImageBatch]:
        """Score images from multiple transport tasks in full inference batches."""
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task!s} failed validation for stage {self}"
                raise ValueError(msg)

        self._score_images([image for task in tasks for image in task.data])
        return [self._filter_task(task) for task in tasks]
