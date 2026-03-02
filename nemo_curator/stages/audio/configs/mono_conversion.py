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

"""Configuration for Mono Conversion stage."""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class MonoConversionConfig:
    """
    Configuration for Mono Conversion stage.
    
    Converts multi-channel audio to mono and verifies sample rate.
    Typically the first stage in an audio processing pipeline.
    
    Attributes:
        output_sample_rate: Expected/output sample rate in Hz
        audio_filepath_key: Key in data dict for audio file path
        strict_sample_rate: If True, reject audio with wrong sample rate
    
    Example:
        # Default configuration
        config = MonoConversionConfig()
        
        # Custom sample rate
        config = MonoConversionConfig(output_sample_rate=16000)
        
        # Allow any sample rate
        config = MonoConversionConfig(strict_sample_rate=False)
    """
    
    output_sample_rate: int = 48000
    audio_filepath_key: str = "audio_filepath"
    strict_sample_rate: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MonoConversionConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

