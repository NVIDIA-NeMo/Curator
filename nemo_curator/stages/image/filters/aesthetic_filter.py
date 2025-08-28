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

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.models.aesthetics import AestheticScorer
from nemo_curator.stages.image.filters.base import BaseFilterStage
from nemo_curator.tasks import ImageBatch


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
    _name: str = "image_aesthetic_filter"

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Initialize the aesthetic filtering model."""
        self.model = AestheticScorer(model_dir=self.model_dir)
        self.model.setup()

        if self.verbose:
            logger.info("Initialized aesthetic scoring model")

    def process(self, task: ImageBatch) -> ImageBatch:
        """Process an image batch to filter by aesthetic score threshold.

        Args:
            task: ImageBatch containing list of ImageObject instances with aesthetic scores

        Returns:
            ImageBatch with filtered images that meet the aesthetic score threshold.
        """

        # Process images in batches to generate scores
        for batch in self.yield_next_batch(task):
            # Stack embeddings into batch tensor (N, embedding_dim)
            embeddings = [img_obj.embedding for img_obj in batch]
            batch_tensor = np.stack(embeddings, axis=0)

            # Generate aesthetic scores
            with torch.no_grad():
                scores = self.model(batch_tensor).cpu().numpy()

            # Store scores in ImageObject.aesthetic_score
            for i, image_obj in enumerate(batch):
                image_obj.aesthetic_score = float(scores[i])

        # Filter images based on aesthetic score threshold
        filtered_images = []
        filtered_count = 0

        for image_obj in task.data:
            if image_obj.aesthetic_score >= self.score_threshold:
                filtered_images.append(image_obj)
            else:
                filtered_count += 1
                if self.verbose:
                    logger.info(
                        f"Image {image_obj.image_id} (path: {image_obj.image_path}) has aesthetic score {image_obj.aesthetic_score:.3f} "
                        f"below threshold {self.score_threshold}, filtered out."
                    )

        if self.verbose:
            logger.info(
                f"Aesthetic filtering: {len(filtered_images)}/{len(task.data)} images passed, "
                f"{filtered_count} filtered out"
            )

        # Return new ImageBatch with filtered images
        return ImageBatch(
            data=filtered_images,
            dataset_name=task.dataset_name,
            task_id=f"{task.task_id}_{self.name}",
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
