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

"""Speaker separation (diarization) configuration."""

from dataclasses import dataclass
from typing import Any


@dataclass
class SpeakerSeparationConfig:
    """
    Configuration for Speaker Separation stage.
    
    Uses NeMo's speaker diarization to separate audio by speaker.
    Produces multiple outputs - one AudioBatch per detected speaker.
    
    Resource Allocation:
        cpus: Number of CPU cores for parallel processing
        gpus: Number of GPUs (0.0-1.0 for fractional)
        
        Speaker diarization models typically require GPU.
        When gpus > 0: Uses GPU for NeMo diarization model
    
    Attributes:
        cpus: CPU cores for parallel processing (default: 1.0)
        gpus: GPU allocation for diarization (default: 0.5)
        model_path: Path to NeMo model (.nemo) or NGC model name
        gap_threshold: Minimum gap between segments to merge (seconds)
        exclude_overlaps: If True, exclude overlapping speech regions
        min_duration: Minimum duration for a speaker segment (seconds)
        buffer_time: Buffer time around overlap boundaries (seconds)
    
    Example:
        # GPU processing (default)
        config = SpeakerSeparationConfig(exclude_overlaps=True)
        
        # Custom GPU allocation
        config = SpeakerSeparationConfig(gpus=1.0, exclude_overlaps=True)
    """
    
    # Resource allocation
    cpus: float = 1.0
    gpus: float = 0.5  # Diarization typically needs GPU
    
    model_path: str = "model/diar_sortformer_4spk-v1.nemo"
    gap_threshold: float = 0.1
    exclude_overlaps: bool = True
    min_duration: float = 0.8
    buffer_time: float = 0.5
    
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
            'cpus': self.cpus,
            'gpus': self.gpus,
            'model_path': self.model_path,
            'gap_threshold': self.gap_threshold,
            'exclude_overlaps': self.exclude_overlaps,
            'min_duration': self.min_duration,
            'buffer_time': self.buffer_time,
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return getattr(self, key, default)

