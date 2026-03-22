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
    spectral characteristics.

    Attributes:
        band_value: Which band type to pass ("full_band" or "narrow_band")

    Example:
        config = BandFilterConfig(band_value="full_band")
    """

    band_value: Literal["full_band", "narrow_band"] = "full_band"

    @classmethod
    def from_dict(cls, d: dict) -> "BandFilterConfig":
        if d is None:
            return cls()
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def to_dict(self) -> dict:
        return {'band_value': self.band_value}

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
