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

"""SIGMOS (Signal-based Mean Opinion Score) configuration."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SIGMOSConfig:
    """
    Configuration for SIGMOS quality assessment filter.
    
    SIGMOS predicts audio quality dimensions (1-5 scale):
    - NOISE: Background noise level (higher = less noisy)
    - OVRL: Overall quality
    - SIG: Signal quality
    - COL: Coloration
    - DISC: Discontinuity
    - LOUD: Loudness
    - REVERB: Reverberation (higher = less reverb)
    
    Set threshold to None to disable that check.
    
    Resource Allocation:
        cpus: Number of CPU cores for parallel processing
        gpus: Number of GPUs for parallel segment processing
        
        Processing Modes:
        - gpus == 0, cpus > 1: CPU parallel processing (ThreadPoolExecutor)
        - gpus > 0 and gpus < 1: Single GPU, sequential processing
        - gpus >= 1: Single GPU, sequential processing
        - gpus >= 2: Multi-GPU parallel processing (segments distributed across GPUs)
    
    Attributes:
        cpus: CPU cores for parallel processing (default: 1.0)
        gpus: GPU allocation for model inference (default: 0.3)
              Set to N (integer >= 2) for multi-GPU parallel processing
        model_path: Path to SIGMOS ONNX model
        noise_threshold: Minimum noise score (None to disable)
        ovrl_threshold: Minimum overall score (None to disable)
        sig_threshold: Minimum signal score (None to disable)
        col_threshold: Minimum coloration score (None to disable)
        disc_threshold: Minimum discontinuity score (None to disable)
        loud_threshold: Minimum loudness score (None to disable)
        reverb_threshold: Minimum reverb score (None to disable)
    
    Example:
        # Single GPU processing (default)
        config = SIGMOSConfig(noise_threshold=4.0, ovrl_threshold=3.5)
        
        # Multi-GPU parallel processing (8 GPUs)
        config = SIGMOSConfig(gpus=8.0, noise_threshold=4.0)
        
        # CPU parallel processing with 4 workers
        config = SIGMOSConfig(cpus=4.0, gpus=0.0, noise_threshold=4.0)
    """
    
    # Resource allocation
    cpus: float = 1.0
    gpus: float = 0.3
    
    model_path: str = "model/model-sigmos_1697718653_41d092e8-epo-200.onnx"
    noise_threshold: Optional[float] = 4.0
    ovrl_threshold: Optional[float] = 3.5
    sig_threshold: Optional[float] = None
    col_threshold: Optional[float] = None
    disc_threshold: Optional[float] = None
    loud_threshold: Optional[float] = None
    reverb_threshold: Optional[float] = None
    
    @classmethod
    def from_dict(cls, d: dict) -> "SIGMOSConfig":
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
            'noise_threshold': self.noise_threshold,
            'ovrl_threshold': self.ovrl_threshold,
            'sig_threshold': self.sig_threshold,
            'col_threshold': self.col_threshold,
            'disc_threshold': self.disc_threshold,
            'loud_threshold': self.loud_threshold,
            'reverb_threshold': self.reverb_threshold,
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return getattr(self, key, default)
    
    def get_active_thresholds(self) -> dict:
        """Return only thresholds that are not None."""
        thresholds = {}
        for name in ['noise', 'ovrl', 'sig', 'col', 'disc', 'loud', 'reverb']:
            val = getattr(self, f'{name}_threshold')
            if val is not None:
                thresholds[name] = val
        return thresholds

