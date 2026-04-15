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

"""Vendored AudioSet-tagging CNN models (PANNs) for sound event detection.

Based on https://github.com/qiuqiangkong/audioset_tagging_cnn
Requires ``torchlibrosa`` (pip install torchlibrosa).
"""

from nemo_curator.stages.audio.inference.sed_models.cnn14 import (
    MODEL_REGISTRY,
    Cnn14DecisionLevelAtt,
    Cnn14DecisionLevelAvg,
    Cnn14DecisionLevelMax,
)

__all__ = [
    "MODEL_REGISTRY",
    "Cnn14DecisionLevelAtt",
    "Cnn14DecisionLevelAvg",
    "Cnn14DecisionLevelMax",
]
