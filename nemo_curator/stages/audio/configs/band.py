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

"""Band filter (bandwidth classification) configuration."""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class BandFilterConfig:
    """
    Configuration for Band Filter stage.
    
    Classifies audio as "full_band" or "narrow_band" based on
    spectral characteristics. Useful for filtering low-quality
    telephone or compressed audio.
    
    Note: The band classifier uses a scikit-learn model (RandomForest) which
    runs on CPU. However, feature extraction uses PyTorch and can benefit
    from GPU acceleration and multi-GPU parallelization.
    
    Resource Allocation:
        cpus: Number of CPU cores for parallel processing
        gpus: Number of GPUs for parallel feature extraction
        
        Processing Modes:
        - gpus == 0, cpus > 1: CPU parallel processing (ThreadPoolExecutor)
        - gpus > 0 and gpus < 1: Single GPU for feature extraction
        - gpus >= 1: Single GPU for feature extraction
        - gpus >= 2: Multi-GPU parallel processing (segments distributed across GPUs)
    
    Attributes:
        cpus: CPU cores for parallel processing (default: 1.0)
        gpus: GPU allocation (default: 0.0, CPU-only by default)
              Set to N (integer >= 2) for multi-GPU parallel processing
        model_path: Path to band classifier model (.joblib)
        feature_group: Feature extraction group
        n_workers: Number of parallel workers for feature extraction
        feature_cache_size: Size of feature cache
        band_value: Which band type to pass ("full_band" or "narrow_band")
    
    Example:
        # CPU processing (default)
        config = BandFilterConfig(band_value="full_band")
        
        # Multi-GPU parallel processing (8 GPUs)
        config = BandFilterConfig(gpus=8.0, band_value="full_band")
        
        # CPU parallel processing with 4 workers
        config = BandFilterConfig(cpus=4.0, band_value="full_band")
    """
    
    # Resource allocation
    cpus: float = 1.0
    gpus: float = 0.0  # Band filter is CPU-only by default
    
    model_path: str = "model/band_classifier_model_band_7000_samples.joblib"
    feature_group: str = "band"
    n_workers: int = 4
    feature_cache_size: int = 100
    band_value: Literal["full_band", "narrow_band"] = "full_band"
    
    @classmethod
    def from_dict(cls, d: dict) -> "BandFilterConfig":
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
            'feature_group': self.feature_group,
            'n_workers': self.n_workers,
            'feature_cache_size': self.feature_cache_size,
            'band_value': self.band_value,
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return getattr(self, key, default)
