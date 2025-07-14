from dataclasses import dataclass, field
from typing import Any

from .tasks import Task


@dataclass
class ImageObject:
    """Represents a single image with metadata."""

    image_path: str = ""
    image_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


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

    def num_items(self) -> int:
        """Number of images in this batch."""
        return len(self.data)
