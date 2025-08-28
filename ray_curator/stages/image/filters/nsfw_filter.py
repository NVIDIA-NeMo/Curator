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
from nemo_curator.models.nsfw import NSFWScorer
from nemo_curator.stages.image.filters.base import BaseFilterStage
from nemo_curator.tasks import ImageBatch


@dataclass
class ImageNSFWFilterStage(BaseFilterStage):
    """Stage for filtering out NSFW images using NSFWScorer model.

    This class processes image batches through an NSFW scoring model to generate
    NSFW probability scores for each image. Images with scores above the threshold
    will be filtered out as NSFW content.
    """
    weights_path: str = None
    _name: str = "image_nsfw_filter"

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Download NSFW model weights from LAION repository."""
        NSFWScorer.download_weights_on_node(self.model_dir)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Initialize the NSFW filtering model."""
        self.model = NSFWScorer(model_dir=self.model_dir)
        self.model.setup()

        if self.verbose:
            logger.info("Initialized NSFW scoring model")

    def process(self, task: ImageBatch) -> ImageBatch:
        """Process an image batch to generate NSFW scores and filter by threshold.

        Args:
            task: ImageBatch containing list of ImageObject instances with pre-computed embeddings

        Returns:
            ImageBatch with filtered images that have NSFW scores below the threshold
        """

        # Process images in batches to generate scores
        for batch in self.yield_next_batch(task):
            # Stack embeddings into batch tensor (N, embedding_dim)
            embeddings = [img_obj.embedding for img_obj in batch]
            batch_tensor = np.stack(embeddings, axis=0)

            # Generate NSFW scores
            with torch.no_grad():
                scores = self.model(batch_tensor).cpu().numpy()

            # Store scores in ImageObject.nsfw_score
            for i, image_obj in enumerate(batch):
                image_obj.nsfw_score = float(scores[i])

            if self.verbose:
                logger.info(
                    f"Generated NSFW scores for {len(batch)} images "
                    f"in batch {i}-{i + self.model_inference_batch_size}"
                )

        # Filter images based on NSFW score threshold
        filtered_images = []
        filtered_count = 0

        for image_obj in task.data:
            if image_obj.nsfw_score < self.score_threshold:
                filtered_images.append(image_obj)
            else:
                filtered_count += 1
                if self.verbose:
                    logger.info(
                        f"Image {image_obj.image_id} (path: {image_obj.image_path}) has NSFW score {image_obj.nsfw_score:.3f} "
                        f"above threshold {self.score_threshold}, filtered out as NSFW."
                    )

        if self.verbose:
            logger.info(
                f"NSFW filtering: {len(filtered_images)}/{len(task.data)} images passed, "
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


# Explicitly export the class
__all__ = ["ImageNSFWFilterStage"]
