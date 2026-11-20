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
- AudioFormatStage: Record, select on, or convert sample rate and channel count in one stage
- PrepareIndicASRInputStage: Prepare Hindi audio and reference fields for Indic ASR
- SegmentConcatenationStage: Concatenate multiple audio segments

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.preprocessing import AudioFormatStage

    pipeline = Pipeline(name="preprocessing_pipeline")
    # 48 kHz mono, by selection: nothing is decoded, other rows are dropped.
    pipeline.add_stage(AudioFormatStage(allowed_sample_rates=[48000], allowed_channels=[1]))
    # ...or by conversion: every row is decoded and made 16 kHz mono.
    pipeline.add_stage(AudioFormatStage(target_sample_rate=16000, target_channels=1))
"""

from .audio_format import AudioFormatStage
from .concatenation import SegmentConcatenationStage
from .indic_asr import PrepareIndicASRInputStage

__all__ = [
    "AudioFormatStage",
    "PrepareIndicASRInputStage",
    "SegmentConcatenationStage",
]
