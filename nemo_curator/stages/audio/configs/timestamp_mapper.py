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

"""TimestampMapper configuration."""

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class TimestampMapperConfig:
    """
    Configuration for TimestampMapper stage.

    Controls which metadata fields from upstream stages are included
    in the final output.  The TimestampMapper always produces five
    standard fields (original_file, original_start_ms, original_end_ms,
    duration_ms, duration_sec).  All other fields are controlled by
    passthrough_keys.

    Attributes:
        passthrough_keys: When None (default), every key that is not
            an internal pipeline field (waveform, audio, audio_filepath,
            start_ms, end_ms, segment_num) passes through to the output.
            When set to an explicit list, only those keys are copied.
            Use this to keep the output clean -- for example, to include
            only quality scores and speaker info while excluding internal
            fields like sample_rate, is_mono, num_segments, etc.

    Example:
        # Include everything (default)
        config = TimestampMapperConfig()

        # Only include quality scores and speaker info
        config = TimestampMapperConfig(passthrough_keys=[
            "band_prediction",
            "utmos_mos",
            "sigmos_noise",
            "sigmos_ovrl",
            "speaker_id",
            "num_speakers",
        ])
    """

    passthrough_keys: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "TimestampMapperConfig":
        if d is None:
            return cls()
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def to_dict(self) -> dict:
        return {"passthrough_keys": self.passthrough_keys}

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
