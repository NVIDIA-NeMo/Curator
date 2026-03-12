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

"""UTMOS (UTokyo-SaruLab MOS Prediction) configuration."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class UTMOSConfig:
    """
    Configuration for UTMOS quality assessment filter.

    UTMOS predicts a single Mean Opinion Score (1-5 scale) using a
    self-supervised learning based model (utmos22_strong).
    Audio is internally resampled to the target sample rate (default 16 kHz).

    Attributes:
        mos_threshold: Minimum MOS score to pass (None to disable)
        sample_rate: Target sample rate for UTMOS inference (default 16000)

    Example:
        config = UTMOSConfig(mos_threshold=3.5)
    """

    mos_threshold: Optional[float] = 3.5
    sample_rate: int = 16000

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "UTMOSConfig":
        if d is None:
            return cls()
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def to_dict(self) -> dict:
        return {
            'mos_threshold': self.mos_threshold,
            'sample_rate': self.sample_rate,
        }

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def get_active_thresholds(self) -> dict:
        thresholds = {}
        if self.mos_threshold is not None:
            thresholds['mos'] = self.mos_threshold
        return thresholds
