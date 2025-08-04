from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger

from ray_curator.backends.base import WorkerMetadata
from ray_curator.models.aesthetics import AestheticScorer
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import ImageBatch, ImageObject


@dataclass
class ImageAestheticScoringStage(ProcessingStage[ImageBatch, ImageBatch]):
    """Stage for generating aesthetic scores using AestheticScorer model.

    This class processes image batches through an aesthetic scoring model to generate
    aesthetic scores for each image. It assumes embeddings are already computed
    in ImageObject.embedding and stores scores in ImageObject.aesthetic_score.
    """
    model_dir: str = "models/aesthetics"
    num_gpus_per_worker: float = 0.25
    batch_size: int = 32
    verbose: bool = False

    @property
    def name(self) -> str:
        return "image_aesthetic_scoring"

    @property
    def resources(self) -> Resources:
        return Resources(gpus=self.num_gpus_per_worker)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        """Initialize the aesthetic scoring model."""
        self.model = AestheticScorer(model_dir=self.model_dir)
        self.model.setup()
        
        if self.verbose:
            logger.info("Initialized aesthetic scoring model")

    def process(self, task: ImageBatch) -> ImageBatch:
        """Process an image batch to generate aesthetic scores.
        
        Args:
            task: ImageBatch containing list of ImageObject instances with pre-computed embeddings
            
        Returns:
            ImageBatch with aesthetic scores stored in ImageObject.aesthetic_score
        """

        # Process images in batches
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
                
            if self.verbose:
                logger.info(
                    f"Generated aesthetic scores for {len(batch_images)} images "
                    f"in batch {batch_start}-{batch_end}"
                )
                
        return task
