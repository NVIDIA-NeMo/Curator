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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from .tasks import Task


@dataclass
class WebDatasetSample:
    """Represents a single sample from a WebDataset tar shard.

    A WebDataset sample is identified by a key and contains one or more
    components (e.g. ``"jpg"``, ``"json"``, ``"txt"``, ``"cls"``).
    Each component maps an extension name to its decoded data.

    This abstraction is modality-agnostic: images, text, audio, or any
    other payload can be stored as components, making it suitable for
    multimodal datasets.

    Attributes:
        key: Unique sample key within the shard (typically the filename
            stem shared by all component files in the tar entry).
        components: Mapping from extension name (e.g. ``"jpg"``,
            ``"json"``) to the decoded data for that component.
            Image data is expected to be an ``np.ndarray`` in
            ``(H, W, C)`` RGB format; text data is expected to be
            ``str``; JSON data is expected to be ``dict``; raw binary
            is ``bytes``.
        metadata: Arbitrary metadata associated with the sample (e.g.
            scores, labels, provenance info).  Not stored inside the
            tar but may appear in sidecar Parquet files.
        shard_path: Path to the source tar shard this sample was read
            from.  Populated by the reader stage.
    """

    key: str = ""
    components: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    shard_path: str = ""

    # -- convenience helpers for common modalities --

    def get_image(self, ext: str = "jpg") -> np.ndarray | None:
        """Return the image component as a numpy array, or ``None``."""
        return self.components.get(ext)

    def set_image(self, data: np.ndarray, ext: str = "jpg") -> None:
        """Store decoded image data under the given extension key."""
        self.components[ext] = data

    def get_text(self, ext: str = "txt") -> str | None:
        """Return a text component, or ``None``."""
        val = self.components.get(ext)
        return val if isinstance(val, str) else None

    def set_text(self, text: str, ext: str = "txt") -> None:
        """Store a text component."""
        self.components[ext] = text

    def get_json(self, ext: str = "json") -> dict | None:
        """Return a JSON component, or ``None``."""
        val = self.components.get(ext)
        return val if isinstance(val, dict) else None

    def set_json(self, data: dict, ext: str = "json") -> None:
        """Store a JSON component."""
        self.components[ext] = data

    @property
    def extensions(self) -> list[str]:
        """Return the list of component extension keys present."""
        return list(self.components.keys())


@dataclass
class WebDatasetBatch(Task[list[WebDatasetSample]]):
    """Task for processing batches of WebDataset samples.

    This is the distributed-friendly unit of work for WebDataset
    pipelines.  Each batch holds a list of :class:`WebDatasetSample`
    objects decoded from one or more tar shards.
    """

    data: list[WebDatasetSample] = field(default_factory=list)

    def validate(self) -> bool:
        """Validate the task data."""
        return True

    @property
    def num_items(self) -> int:
        """Number of samples in this batch."""
        return len(self.data)
