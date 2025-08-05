from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger

from ray_curator.backends.base import WorkerMetadata
from ray_curator.models.nsfw import NSFWScorer
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import ImageBatch, ImageObject


@dataclass
class ImageNSFWFilterStage(ProcessingStage[ImageBatch, ImageBatch]):
    """Stage for filtering out NSFW images using NSFWScorer model.

    This class processes image batches through an NSFW scoring model to generate
    NSFW probability scores for each image. Images with scores above the threshold
    will be filtered out as NSFW content.
    """
    model_dir: str = "models/nsfw"
    num_gpus_per_worker: float = 0.25
    batch_size: int = 32
    score_threshold: float = 0.5
    verbose: bool = False

    @property
    def name(self) -> str:
        return "image_nsfw_filter"

    @property
    def resources(self) -> Resources:
        return Resources(gpus=self.num_gpus_per_worker)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
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
        num_images = len(task.data)
        for batch_start in range(0, num_images, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_images)
            batch_images = task.data[batch_start:batch_end]
            
            # Stack embeddings into batch tensor (N, embedding_dim)
            embeddings = [img_obj.embedding for img_obj in batch_images]
            batch_tensor = np.stack(embeddings, axis=0)
            
            # Generate NSFW scores
            with torch.no_grad():
                scores = self.model(batch_tensor).cpu().numpy()
            
            # Store scores in ImageObject.nsfw_score
            for i, image_obj in enumerate(batch_images):
                image_obj.nsfw_score = float(scores[i])
                
            if self.verbose:
                logger.info(
                    f"Generated NSFW scores for {len(batch_images)} images "
                    f"in batch {batch_start}-{batch_end}"
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
