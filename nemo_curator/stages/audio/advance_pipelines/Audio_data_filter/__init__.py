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
Audio Data Filter pipeline.

Composite pipeline stage for audio curation with VAD,
quality filtering, speaker separation, and timestamp tracking.

Example::

    from nemo_curator.stages.audio.advance_pipelines import (
        AudioDataFilterStage,
    )

    # Using default config
    pipeline.add_stage(AudioDataFilterStage())

    # Using custom YAML config
    pipeline.add_stage(AudioDataFilterStage(config_path="my_config.yaml"))
"""

from .audio_data_filter import AudioDataFilterStage

__all__ = [
    "AudioDataFilterStage",
]
