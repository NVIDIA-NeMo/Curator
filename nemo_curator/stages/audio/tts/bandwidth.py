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

"""Annotate audio bandwidth."""

from __future__ import annotations

from dataclasses import dataclass

from nemo_curator.stages.audio.metrics.bandwidth import BandwidthEstimationStage, estimate_bandwidth

__all__ = ["BandwidthAnnotationStage", "estimate_bandwidth"]


@dataclass
class BandwidthAnnotationStage(BandwidthEstimationStage):
    """Compatibility wrapper for the Granary annotation configuration."""

    output_key: str | None = "bandwidth"
    name: str = "BandwidthAnnotation"
    batch_size: int = 16
