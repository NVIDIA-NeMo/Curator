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
Audio segmentation stages.

These stages segment audio into smaller chunks:
- VADSegmentationStage: Segment based on voice activity detection
- SpeakerSeparationStage: Separate audio by speaker using diarization

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.segmentation import VADSegmentationStage
    
    pipeline = Pipeline(name="segmentation_pipeline")
    pipeline.add_stage(VADSegmentationStage(min_duration_sec=2.0))
"""

from .vad_segmentation import VADSegmentationStage
from .speaker_separation import SpeakerSeparationStage

__all__ = ["VADSegmentationStage", "SpeakerSeparationStage"]
