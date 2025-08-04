from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger

from ray_curator.backends.base import WorkerMetadata
from ray_curator.models.clip import CLIPImageEmbeddings
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import ImageBatch, ImageObject


@dataclass
class ImageEmbeddingStage(ProcessingStage[ImageBatch, ImageBatch]):
    """Stage for generating image embeddings using CLIP model.

    This class processes image batches through a CLIP model to generate
    embeddings for each image. It assumes image data is already loaded
    in ImageObject.image_data and stores embeddings in ImageObject.embedding.
    """
    model_dir: str = "models/clip"
    num_gpus_per_worker: float = 0.25
    batch_size: int = 32
    verbose: bool = False

    @property
    def name(self) -> str:
        return "image_embedding"

    @property
    def resources(self) -> Resources:
        return Resources(gpus=self.num_gpus_per_worker)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        """Initialize the CLIP image embedding model."""
        self.model = CLIPImageEmbeddings(model_dir=self.model_dir)
        self.model.setup()
        
        if self.verbose:
            logger.info("Initialized CLIP model for image embedding generation")

    def process(self, task: ImageBatch) -> ImageBatch:
        """Process an image batch to generate embeddings.
        
        Args:
            task: ImageBatch containing list of ImageObject instances with pre-loaded image_data
            
        Returns:
            ImageBatch with embeddings stored in ImageObject.embedding
        """

        # Process images in batches
        num_images = len(task.data)
        for batch_start in range(0, num_images, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_images)
            batch_images = task.data[batch_start:batch_end]
            
            # Stack images into batch tensor (N, H, W, C)
            loaded_images = [img_obj.image_data for img_obj in batch_images]

            # DEBUGGING
            from transformers import CLIPProcessor
            from PIL import Image
            batch_tensor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")(
                images=[Image.fromarray(img) for img in loaded_images], return_tensors="pt"
            )["pixel_values"].numpy()

            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model(batch_tensor).cpu().numpy()
            
            # Store embeddings in ImageObject.embedding
            for i, image_obj in enumerate(batch_images):
                image_obj.embedding = embeddings[i]
                
            if self.verbose:
                logger.info(
                    f"Generated embeddings for {len(batch_images)} images "
                    f"in batch {batch_start}-{batch_end}"
                )
        
        return task
