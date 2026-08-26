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

"""Shared audio model-input segmentation helpers.

Segmentation creates model-safe work units. Duration-aware bucketing is a
separate packing step that consumes these bounded units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioSegment:
    """One contiguous audio model-input segment in sample coordinates."""

    index: int
    count: int
    start_sample: int
    stop_sample: int
    duration_s: float


def resolve_max_model_input_duration(
    *,
    max_duration_s: float,
    owner: str,
) -> float:
    """Validate and normalize the model-input duration ceiling."""

    maximum = float(max_duration_s)
    if maximum <= 0:
        msg = f"{owner}.max_inference_duration_s must be > 0 s, got {max_duration_s}"
        raise ValueError(msg)
    return maximum


def plan_audio_segments(
    *,
    num_samples: int,
    sample_rate: int,
    max_duration_s: float,
    owner: str,
) -> tuple[AudioSegment, ...]:
    """Create bounded contiguous segment specs for one audio input."""

    maximum = resolve_max_model_input_duration(
        max_duration_s=max_duration_s,
        owner=owner,
    )
    if sample_rate <= 0:
        msg = f"{owner}.sample_rate must be > 0, got {sample_rate}"
        raise ValueError(msg)
    if num_samples <= 0:
        return (
            AudioSegment(
                index=0,
                count=1,
                start_sample=0,
                stop_sample=0,
                duration_s=0.0,
            ),
        )

    max_samples = max(1, int(maximum * float(sample_rate)))
    starts = list(range(0, int(num_samples), max_samples))
    count = max(1, len(starts))
    segments: list[AudioSegment] = []
    for index, start in enumerate(starts):
        stop = min(start + max_samples, int(num_samples))
        duration_s = float(stop - start) / float(sample_rate)
        segments.append(
            AudioSegment(
                index=index,
                count=count,
                start_sample=start,
                stop_sample=stop,
                duration_s=duration_s,
            )
        )
    return tuple(segments)


def duration_to_num_samples(duration_s: float, sample_rate: int) -> int:
    """Return ceil(duration_s * sample_rate) with non-negative duration."""

    if sample_rate <= 0:
        msg = f"sample_rate must be > 0, got {sample_rate}"
        raise ValueError(msg)
    return math.ceil(max(float(duration_s), 0.0) * float(sample_rate))
