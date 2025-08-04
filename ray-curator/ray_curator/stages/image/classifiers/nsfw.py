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
class ImageNSFWScoringStage(ProcessingStage[ImageBatch, ImageBatch]):
    """Stage for generating NSFW scores using NSFWScorer model.

    This class processes image batches through an NSFW scoring model to generate
    NSFW probability scores for each image. It assumes embeddings are already computed
    in ImageObject.embedding and stores scores in ImageObject.nsfw_score.
    """
    model_dir: str = "models/nsfw"
    num_gpus_per_worker: float = 0.25
    batch_size: int = 32
    verbose: bool = False

    @property
    def name(self) -> str:
        return "image_nsfw_scoring"

    @property
    def resources(self) -> Resources:
        return Resources(gpus=self.num_gpus_per_worker)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        """Initialize the NSFW scoring model."""
        self.model = NSFWScorer(model_dir=self.model_dir)
        self.model.setup()
        
        if self.verbose:
            logger.info("Initialized NSFW scoring model")

    def process(self, task: ImageBatch) -> ImageBatch:
        """Process an image batch to generate NSFW scores.
        
        Args:
            task: ImageBatch containing list of ImageObject instances with pre-computed embeddings
            
        Returns:
            ImageBatch with NSFW scores stored in ImageObject.nsfw_score
        """

        # Process images in batches
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
                
        return task
