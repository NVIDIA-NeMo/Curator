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

"""SIGMOS (Signal-based Mean Opinion Score) configuration."""

from dataclasses import dataclass
from typing import Any, Optional

_MODEL_PATH = "model/model-sigmos_1697718653_41d092e8-epo-200.onnx"


@dataclass
class SIGMOSConfig:
    """
    Configuration for SIGMOS quality assessment filter.

    SIGMOS predicts audio quality dimensions (1-5 scale).
    Set threshold to None to disable that check.

    Attributes:
        noise_threshold: Minimum noise score (None to disable)
        ovrl_threshold: Minimum overall score (None to disable)
        sig_threshold: Minimum signal score (None to disable)
        col_threshold: Minimum coloration score (None to disable)
        disc_threshold: Minimum discontinuity score (None to disable)
        loud_threshold: Minimum loudness score (None to disable)
        reverb_threshold: Minimum reverb score (None to disable)

    Example:
        config = SIGMOSConfig(noise_threshold=4.0, ovrl_threshold=3.5)
    """

    noise_threshold: Optional[float] = 4.0
    ovrl_threshold: Optional[float] = 3.5
    sig_threshold: Optional[float] = None
    col_threshold: Optional[float] = None
    disc_threshold: Optional[float] = None
    loud_threshold: Optional[float] = None
    reverb_threshold: Optional[float] = None

    @property
    def model_path(self) -> str:
        return _MODEL_PATH

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "SIGMOSConfig":
        if d is None:
            return cls()
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def to_dict(self) -> dict:
        return {
            'noise_threshold': self.noise_threshold,
            'ovrl_threshold': self.ovrl_threshold,
            'sig_threshold': self.sig_threshold,
            'col_threshold': self.col_threshold,
            'disc_threshold': self.disc_threshold,
            'loud_threshold': self.loud_threshold,
            'reverb_threshold': self.reverb_threshold,
        }

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def get_active_thresholds(self) -> dict:
        thresholds = {}
        for name in ['noise', 'ovrl', 'sig', 'col', 'disc', 'loud', 'reverb']:
            val = getattr(self, f'{name}_threshold')
            if val is not None:
                thresholds[name] = val
        return thresholds
