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

"""NISQA (Non-Intrusive Speech Quality Assessment) configuration."""

from dataclasses import dataclass
from typing import Any, Optional

_MODEL_PATH = "model/nisqa.tar"


@dataclass
class NISQAConfig:
    """
    Configuration for NISQA quality assessment filter.

    NISQA predicts speech quality dimensions (1-5 scale).
    Set threshold to None to disable that check.

    Attributes:
        mos_threshold: Minimum MOS score (None to disable)
        noi_threshold: Minimum noisiness score (None to disable)
        col_threshold: Minimum coloration score (None to disable)
        dis_threshold: Minimum discontinuity score (None to disable)
        loud_threshold: Minimum loudness score (None to disable)

    Example:
        config = NISQAConfig(mos_threshold=4.5, noi_threshold=4.3)
    """

    mos_threshold: Optional[float] = 4.5
    noi_threshold: Optional[float] = 4.3
    col_threshold: Optional[float] = None
    dis_threshold: Optional[float] = None
    loud_threshold: Optional[float] = None

    @property
    def model_path(self) -> str:
        return _MODEL_PATH

    @classmethod
    def from_dict(cls, d: dict) -> "NISQAConfig":
        if d is None:
            return cls()
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def to_dict(self) -> dict:
        return {
            'mos_threshold': self.mos_threshold,
            'noi_threshold': self.noi_threshold,
            'col_threshold': self.col_threshold,
            'dis_threshold': self.dis_threshold,
            'loud_threshold': self.loud_threshold,
        }

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def get_active_thresholds(self) -> dict:
        thresholds = {}
        for name in ['mos', 'noi', 'col', 'dis', 'loud']:
            val = getattr(self, f'{name}_threshold')
            if val is not None:
                thresholds[name] = val
        return thresholds
