

from collections.abc import Generator
from dataclasses import dataclass

import torch
from loguru import logger

from ray_curator.backends.base import WorkerMetadata
from ray_curator.models.clip import CLIPImageEmbeddings
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import ImageBatch


@dataclass
class ImageEmbeddingStage(ProcessingStage[ImageBatch, ImageBatch]):
    """Stage for generating image embeddings using CLIP model.

    This class processes image batches through a CLIP model to generate
    embeddings for each image. It assumes image data is already loaded
    in ImageObject.image_data and stores embeddings in ImageObject.embedding.
    """
    model_dir: str = None
    num_gpus_per_worker: float = 0.25
    model_inference_batch_size: int = 32  # Number of images to process through model at once
    verbose: bool = False

    @property
    def name(self) -> str:
        return "image_embedding"

    @property
    def resources(self) -> Resources:
        if torch.cuda.is_available():
            return Resources(gpus=self.num_gpus_per_worker)
        else:
            return Resources()

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Initialize the CLIP image embedding model."""
        # Use positional arg for compatibility with test doubles that may
        # not accept keyword arguments
        self.model = CLIPImageEmbeddings(self.model_dir)
        self.model.setup()

        if self.verbose:
            logger.info("Initialized CLIP model for image embedding generation")

    def yield_next_batch(self, task: ImageBatch) -> Generator[ImageBatch, None, None]:
        """Yield batches of images from the task.

        Args:
            task: ImageBatch containing list of ImageObject instances with pre-loaded image_data

        Yields:
            Generator[dict[str, torch.Tensor]]: A generator of model inputs for the next batch.

        """
        for i in range(0, len(task.data), self.model_inference_batch_size):
            yield task.data[i : i + self.model_inference_batch_size]

    def process(self, task: ImageBatch) -> ImageBatch:
        """Process an image batch to generate embeddings.

        Args:
            task: ImageBatch containing list of ImageObject instances with pre-loaded image_data

        Returns:
            ImageBatch with embeddings stored in ImageObject.embedding
        """

        for batch in self.yield_next_batch(task):
            # Stack images into batch tensor (N, H, W, C)
            loaded_images = [img_obj.image_data for img_obj in batch]
            batch_tensor = loaded_images

            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model(batch_tensor).cpu().numpy()

            # Store embeddings in ImageObject.embedding
            for i, image_obj in enumerate(batch):
                image_obj.embedding = embeddings[i]

            if self.verbose:
                logger.info(
                    f"Generated embeddings for {len(batch)} images."
                )

        return task
