from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger

from ray_curator.backends.base import WorkerMetadata
from ray_curator.models.aesthetics import AestheticScorer
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import ImageBatch

# Constants
DEBUG_LOW_AESTHETIC_THRESHOLD = 0.1  # Threshold for debug logging of very low aesthetic scores


@dataclass
class ImageAestheticFilterStage(ProcessingStage[ImageBatch, ImageBatch]):
    """Stage for filtering out images based on aesthetic scores.

    This class processes image batches through an aesthetic scoring model to generate
    aesthetic scores for each image. Images with scores below the threshold will be filtered out.
    """
    model_dir: str = "models/aesthetics"
    num_gpus_per_worker: float = 0.25
    batch_size: int = 32
    score_threshold: float = 0.5
    verbose: bool = False

    @property
    def name(self) -> str:
        return "image_aesthetic_filter"

    @property
    def resources(self) -> Resources:
        return Resources(gpus=self.num_gpus_per_worker)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
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
        num_images = len(task.data)
        for batch_start in range(0, num_images, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_images)
            batch_images = task.data[batch_start:batch_end]

            # Stack embeddings into batch tensor (N, embedding_dim)
            embeddings = [img_obj.embedding for img_obj in batch_images]
            batch_tensor = np.stack(embeddings, axis=0)

            # Generate aesthetic scores
            with torch.no_grad():
                scores = self.model(batch_tensor).cpu().numpy()

            # Store scores in ImageObject.aesthetic_score
            for i, image_obj in enumerate(batch_images):
                image_obj.aesthetic_score = float(scores[i])

                # Debug: show paths of images with very low aesthetic scores
                if image_obj.aesthetic_score < DEBUG_LOW_AESTHETIC_THRESHOLD:
                    logger.info(
                        f"[DEBUG] Low aesthetic score: {image_obj.aesthetic_score:.3f} - "
                        f"Image {image_obj.image_id} (path: {image_obj.image_path})"
                    )

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
