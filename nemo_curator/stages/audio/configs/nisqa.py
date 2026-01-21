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

"""NISQA (Non-Intrusive Speech Quality Assessment) configuration."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class NISQAConfig:
    """
    Configuration for NISQA quality assessment filter.
    
    NISQA predicts speech quality dimensions (1-5 scale):
    - MOS: Overall Mean Opinion Score
    - NOI: Noisiness (higher = less noisy)
    - COL: Coloration/distortion
    - DIS: Discontinuity
    - LOUD: Loudness appropriateness
    
    Set threshold to None to disable that check.
    
    Resource Allocation:
        cpus: Number of CPU cores for parallel processing
        gpus: Number of GPUs (0.0-1.0 for fractional)
        
        When cpus > 1 and gpus == 0: Uses CPU parallel processing
        When gpus > 0: Uses GPU for NISQA model inference
    
    Attributes:
        cpus: CPU cores for parallel processing (default: 1.0)
        gpus: GPU allocation for model inference (default: 0.3)
        model_path: Path to NISQA model weights
        mos_threshold: Minimum MOS score (None to disable)
        noi_threshold: Minimum noisiness score (None to disable)
        col_threshold: Minimum coloration score (None to disable)
        dis_threshold: Minimum discontinuity score (None to disable)
        loud_threshold: Minimum loudness score (None to disable)
    
    Example:
        # GPU processing (default)
        config = NISQAConfig(mos_threshold=4.5, noi_threshold=4.3)
        
        # CPU parallel processing with 4 workers
        config = NISQAConfig(cpus=4.0, gpus=0.0, mos_threshold=4.0)
    """
    
    # Resource allocation
    cpus: float = 1.0
    gpus: float = 0.3
    
    model_path: str = "model/nisqa.tar"
    mos_threshold: Optional[float] = 4.5
    noi_threshold: Optional[float] = 4.3
    col_threshold: Optional[float] = None
    dis_threshold: Optional[float] = None
    loud_threshold: Optional[float] = None
    
    @classmethod
    def from_dict(cls, d: dict) -> "NISQAConfig":
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
            'mos_threshold': self.mos_threshold,
            'noi_threshold': self.noi_threshold,
            'col_threshold': self.col_threshold,
            'dis_threshold': self.dis_threshold,
            'loud_threshold': self.loud_threshold,
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return getattr(self, key, default)
    
    def get_active_thresholds(self) -> dict:
        """Return only thresholds that are not None."""
        thresholds = {}
        for name in ['mos', 'noi', 'col', 'dis', 'loud']:
            val = getattr(self, f'{name}_threshold')
            if val is not None:
                thresholds[name] = val
        return thresholds

