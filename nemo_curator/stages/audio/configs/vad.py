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

"""VAD (Voice Activity Detection) configuration."""

from dataclasses import dataclass
from typing import Any

# Internal defaults (not exposed in config)
_MIN_INTERVAL_MS = 500
_THRESHOLD = 0.5
_SPEECH_PAD_MS = 300


@dataclass
class VADConfig:
    """
    Configuration for Voice Activity Detection stage.

    Controls how audio is segmented based on speech activity.
    Uses Silero VAD model which supports both CPU and GPU execution.

    Attributes:
        min_duration_sec: Minimum segment duration to keep (seconds)
        max_duration_sec: Maximum segment duration (seconds)

    Example:
        config = VADConfig(min_duration_sec=2.0, max_duration_sec=30.0)
    """

    min_duration_sec: float = 2.0
    max_duration_sec: float = 60.0

    @property
    def min_interval_ms(self) -> int:
        return _MIN_INTERVAL_MS

    @property
    def threshold(self) -> float:
        return _THRESHOLD

    @property
    def speech_pad_ms(self) -> int:
        return _SPEECH_PAD_MS

    @classmethod
    def from_dict(cls, d: dict) -> "VADConfig":
        """Create config from dictionary."""
        if d is None:
            return cls()
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'min_duration_sec': self.min_duration_sec,
            'max_duration_sec': self.max_duration_sec,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return getattr(self, key, default)
