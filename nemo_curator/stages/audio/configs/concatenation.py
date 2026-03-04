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

"""Configuration for Segment Concatenation stage."""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class SegmentConcatenationConfig:
    """
    Configuration for Segment Concatenation stage.
    
    Concatenates multiple audio segments with silence in between.
    Useful for combining quality-filtered segments before further processing.
    
    Attributes:
        silence_duration_sec: Duration of silence between segments (seconds)
        audio_key: Key in data dict containing audio
        batch_size: Number of tasks to collect before concatenating
    
    Example:
        # Default configuration (1 second silence)
        config = SegmentConcatenationConfig()
        
        # Custom silence duration and batch size
        config = SegmentConcatenationConfig(silence_duration_sec=0.5, batch_size=10)
    """
    
    silence_duration_sec: float = 1.0
    audio_key: str = "audio"
    batch_size: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SegmentConcatenationConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

