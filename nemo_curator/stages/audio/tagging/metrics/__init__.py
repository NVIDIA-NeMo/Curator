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

"""Audio quality and error-rate metrics."""

from nemo_curator.stages.audio.tagging.metrics.bandwidth import BandwidthEstimationStage
from nemo_curator.stages.audio.tagging.metrics.squim import TorchSquimQualityMetricsStage
from nemo_curator.stages.audio.tagging.metrics.wer import ComputeWERStage

__all__ = [
    "BandwidthEstimationStage",
    "ComputeWERStage",
    "TorchSquimQualityMetricsStage",
]
