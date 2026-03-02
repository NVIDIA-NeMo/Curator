# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
- MonoConversionStage: Convert to mono and verify sample rate
- SegmentConcatenationStage: Concatenate multiple audio segments

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.preprocessing import MonoConversionStage
    
    pipeline = Pipeline(name="preprocessing_pipeline")
    pipeline.add_stage(MonoConversionStage(output_sample_rate=48000))
"""

from .mono_conversion import MonoConversionStage
from .concatenation import SegmentConcatenationStage

__all__ = ["MonoConversionStage", "SegmentConcatenationStage"]
