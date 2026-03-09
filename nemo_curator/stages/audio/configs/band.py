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
    runs on CPU. Feature extraction uses PyTorch and can benefit
    from GPU acceleration.

    Attributes:
        model_path: Path to band classifier model (.joblib)
        model_inference_batch_size: Number of items per batch for model inference.
        band_value: Which band type to pass ("full_band" or "narrow_band")

    Note:
        Worker count and feature cache size are internal (derived from resources
        or fixed default); not exposed in config (same pattern as image stages).

    Example:
        config = BandFilterConfig(band_value="full_band")
    """

    model_path: str = "model/band_classifier_model_band_7000_samples.joblib"
    model_inference_batch_size: int = 32
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
            'model_path': self.model_path,
            'model_inference_batch_size': self.model_inference_batch_size,
            'band_value': self.band_value,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return getattr(self, key, default)
