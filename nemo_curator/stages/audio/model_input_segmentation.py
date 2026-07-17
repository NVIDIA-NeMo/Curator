# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Payload-free model-input segmentation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioSegment:
    """One bounded model input represented only by time coordinates."""

    index: int
    count: int
    start_s: float
    duration_s: float


def plan_audio_segments(duration_s: float, max_duration_s: float) -> tuple[AudioSegment, ...]:
    """Split metadata duration into contiguous, model-safe descriptors."""
    duration = float(duration_s)
    maximum = float(max_duration_s)
    if duration < 0:
        msg = f"Audio duration must be non-negative, got {duration_s}"
        raise ValueError(msg)
    if maximum <= 0:
        msg = f"Maximum model-input duration must be positive, got {max_duration_s}"
        raise ValueError(msg)
    count = max(1, math.ceil(duration / maximum))
    return tuple(
        AudioSegment(
            index=index,
            count=count,
            start_s=index * maximum,
            duration_s=max(0.0, min(maximum, duration - index * maximum)),
        )
        for index in range(count)
    )
