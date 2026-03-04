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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, Generic
import dataclasses
from pathlib import Path
T = TypeVar("T")

if TYPE_CHECKING:
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

@dataclass(kw_only=True)
class ImageTaskData:
    """Task data for image processing."""
    image_path: Path | str
    image_id: str | None = None
    is_valid: bool = True
    error: str | None = None

    @staticmethod
    def _to_dict_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if dataclasses.is_dataclass(value):
            if hasattr(value, "to_dict"):
                return value.to_dict()
            else:
                return dataclasses.asdict(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [ImageTaskData._to_dict_value(item) for item in value]
        if isinstance(value, dict):
            return {k: ImageTaskData._to_dict_value(v) for k, v in value.items()}
        return value
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            field.name: self._to_dict_value(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }

@dataclass(kw_only=True)
class SingleDataTask(Task[T], Generic[T]):
    """Task that contains a single data item."""
    data: T

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"data": self.data.to_dict()}
    
    def validate(self) -> bool:
        """Validate the task data."""
        return True
    
    @property
    def num_items(self) -> int:
        """Number of items in the task."""
        return 1