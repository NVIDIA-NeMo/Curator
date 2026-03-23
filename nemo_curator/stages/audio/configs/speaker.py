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

"""Speaker separation (diarization) configuration."""

from dataclasses import dataclass
from typing import Any

# Internal defaults (not exposed in config)
_GAP_THRESHOLD = 0.1
_MIN_DURATION = 0.8
_BUFFER_TIME = 0.5


@dataclass
class SpeakerSeparationConfig:
    """
    Configuration for Speaker Separation stage.

    Uses NeMo's speaker diarization to separate audio by speaker.
    Produces multiple outputs - one AudioBatch per detected speaker.

    Attributes:
        model_path: HuggingFace model ID (e.g. "nvidia/diar_sortformer_4spk-v1")
        exclude_overlaps: If True, exclude overlapping speech regions

    Example:
        config = SpeakerSeparationConfig(exclude_overlaps=True)
        config = SpeakerSeparationConfig(model_path="nvidia/diar_sortformer_4spk-v1")
    """

    model_path: str = "nvidia/diar_sortformer_4spk-v1"
    exclude_overlaps: bool = True

    @property
    def gap_threshold(self) -> float:
        return _GAP_THRESHOLD

    @property
    def min_duration(self) -> float:
        return _MIN_DURATION

    @property
    def buffer_time(self) -> float:
        return _BUFFER_TIME

    @classmethod
    def from_dict(cls, d: dict) -> "SpeakerSeparationConfig":
        """Create config from dictionary."""
        if d is None:
            return cls()
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'model_path': self.model_path,
            'exclude_overlaps': self.exclude_overlaps,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return getattr(self, key, default)
