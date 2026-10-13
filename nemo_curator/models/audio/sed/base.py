# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Stage-adapter contract for audio sound-event detection.

``SEDInferenceStage`` owns Curator-side glue: reading ``AudioTask.data``,
loading or normalizing audio, resampling, resume behavior, and writing output
fields or NPZ sidecars. ``SEDAdapter`` owns model construction, checkpoint
loading, model-specific batch padding, inference, and temporal metadata.

Keeping this boundary explicit lets a YAML pipeline replace PANNs with another
SED runtime by changing ``adapter_target`` while preserving the task schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@dataclass
class SEDResult:
    """Canonical result for one waveform returned by an SED adapter.

    Attributes:
        framewise_output: Two-dimensional ``(frames, classes)`` probability
            matrix. It may include a padded tail shared with other batch rows.
        fps: Number of output frames per second.
        valid_frames: Number of leading rows that correspond to real audio.
        original_num_samples: Real waveform length after stage resampling and
            before model-specific padding.
    """

    framewise_output: np.ndarray
    fps: float
    valid_frames: int
    original_num_samples: int


@runtime_checkable
class SEDAdapter(Protocol):
    """Structural protocol implemented by every sound-event adapter.

    Constructor contract: the stage creates an adapter as
    ``cls(checkpoint_path=..., sample_rate=..., **adapter_kwargs)``. A ``None``
    checkpoint path asks the adapter to resolve its registered default.

    ``infer_batch`` receives stage-normalized items in input order. Each item
    contains one contiguous mono float32 ``waveform``. The adapter must return
    exactly one ``SEDResult`` per item, in the same order.
    """

    checkpoint_path: str | None
    sample_rate: int

    def download_weights_on_node(self) -> None:
        """Cache model weights without allocating worker-local model state."""
        ...

    def load_model(self, *, num_gpus: int) -> None:
        """Load worker-local model state for the requested physical GPU count."""
        ...

    def unload_model(self) -> None:
        """Release worker-local model and accelerator state."""
        ...

    def infer_batch(self, items: list[dict[str, Any]]) -> list[SEDResult]:
        """Return one canonical result per prepared waveform, in order."""
        ...
