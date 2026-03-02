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

"""VAD (Voice Activity Detection) configuration."""

from dataclasses import dataclass
from typing import Any


@dataclass
class VADConfig:
    """
    Configuration for Voice Activity Detection stage.
    
    Controls how audio is segmented based on speech activity.
    Uses Silero VAD model which supports both CPU and GPU execution.
    
    Resource Allocation:
        cpus: Number of CPU cores for parallel processing
        gpus: Number of GPUs (0.0-1.0 for fractional)
        
        Silero VAD is lightweight, so gpus=0.1 is usually sufficient.
        When cpus > 1 and gpus == 0: Uses CPU parallel processing
        When gpus > 0: Uses GPU for VAD model inference
    
    Attributes:
        cpus: CPU cores for parallel processing (default: 1.0)
        gpus: GPU allocation for VAD inference (default: 0.1)
        min_interval_ms: Minimum silence interval between segments (ms)
        min_duration_sec: Minimum segment duration to keep (seconds)
        max_duration_sec: Maximum segment duration (seconds)
        threshold: VAD detection threshold (0.0-1.0)
        speech_pad_ms: Padding around detected speech (ms)
    
    Example:
        # GPU execution (default, lightweight)
        config = VADConfig(min_duration_sec=2.0, max_duration_sec=30.0)
        
        # CPU-only execution
        config = VADConfig(cpus=4.0, gpus=0.0, min_duration_sec=2.0)
    """
    
    # Resource allocation
    cpus: float = 1.0
    gpus: float = 0.1  # Silero VAD is lightweight
    
    min_interval_ms: int = 500
    min_duration_sec: float = 2.0
    max_duration_sec: float = 60.0
    threshold: float = 0.5
    speech_pad_ms: int = 300
    
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
            'cpus': self.cpus,
            'gpus': self.gpus,
            'min_interval_ms': self.min_interval_ms,
            'min_duration_sec': self.min_duration_sec,
            'max_duration_sec': self.max_duration_sec,
            'threshold': self.threshold,
            'speech_pad_ms': self.speech_pad_ms,
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return getattr(self, key, default)

