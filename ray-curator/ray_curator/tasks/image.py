from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .tasks import Task


@dataclass
class ImageObject:
    """Represents a single image with metadata.
    
    Attributes:
        image_path: Path to the image file on disk
        image_id: Unique identifier for the image
        metadata: Additional metadata associated with the image
        image_data: Raw image pixel data as numpy array (H, W, C) in RGB format
        embedding: Image embedding vector as numpy array
        aesthetic_score: Aesthetic quality score as float
        nsfw_score: NSFW probability score as float
    """

    image_path: str = ""
    image_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # raw data
    image_data: np.ndarray | None = None
    # embedding data
    embedding: np.ndarray | None = None
    # classification score
    aesthetic_score: float | None = None
    nsfw_score: float | None = None


@dataclass
class ImageBatch(Task):
    """Task for processing batches of images.
    Images are stored as a list of ImageObject instances, each containing
    the path to the image and associated metadata.
    """

    data: list[ImageObject] = field(default_factory=list)

    def validate(self) -> bool:
        """Validate the task data."""
        # TODO: Implement image validation which should ensure image_path exists
        return True

    @property
    def num_items(self) -> int:
        """Number of images in this batch."""
        return len(self.data)
