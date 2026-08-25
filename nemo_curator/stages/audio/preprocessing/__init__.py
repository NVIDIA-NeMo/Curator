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

"""
Audio preprocessing stages.

These stages prepare audio for further processing:
- ChannelCountStage: Record, select on, or convert the channel count (never resamples)
- SampleRateFilterStage: Keep only acceptable sample rates, recording each (header-only read)
- MonoConversionStage: Convert to mono and verify sample rate in one step
- SegmentConcatenationStage: Concatenate multiple audio segments

Channel policy and rate policy are separate stages so a pipeline can set one without the
other, and each says whether it measures, selects or converts rather than doing two at once.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.preprocessing import (
        ChannelCountStage,
        SampleRateFilterStage,
    )

    pipeline = Pipeline(name="preprocessing_pipeline")
    # 48 kHz mono, by selection: nothing is rewritten, the rest is dropped.
    pipeline.add_stage(SampleRateFilterStage(allowed_sample_rates=[48000]))
    pipeline.add_stage(ChannelCountStage(action="filter", allowed_channels=[1]))
    # ...or by conversion: every row is kept and made mono.
    pipeline.add_stage(ChannelCountStage(action="convert", target_channels=1))
"""

from .channel_count import ChannelCountStage
from .concatenation import SegmentConcatenationStage
from .mono_conversion import MonoConversionStage
from .sample_rate_filter import SampleRateFilterStage

__all__ = [
    "ChannelCountStage",
    "MonoConversionStage",
    "SampleRateFilterStage",
    "SegmentConcatenationStage",
]
