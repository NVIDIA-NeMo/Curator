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

"""Configuration for Mono Conversion stage."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MonoConversionConfig:
    """
    Configuration for Mono Conversion stage.

    Converts multi-channel audio to mono and verifies sample rate.

    Attributes:
        output_sample_rate: Expected/output sample rate in Hz
        strict_sample_rate: If True, reject audio with wrong sample rate

    Example:
        config = MonoConversionConfig(output_sample_rate=16000)
        config = MonoConversionConfig(strict_sample_rate=False)
    """

    output_sample_rate: int = 48000
    strict_sample_rate: bool = True

    @property
    def audio_filepath_key(self) -> str:
        return "audio_filepath"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'output_sample_rate': self.output_sample_rate,
            'strict_sample_rate': self.strict_sample_rate,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MonoConversionConfig":
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in config_dict.items() if k in valid_keys})

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
