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

"""Sound-event-detection model architectures and lazy PANNs registry.

Resolution is lazy so importing the SED stage does not require
``torchlibrosa`` until a model is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

# Checkpoint name -> class name in the vendored PANNs module. Names match the
# released PANNs checkpoints so a downloaded .pth maps to the right architecture.
_MODEL_CLASS_NAMES: dict[str, str] = {
    "Cnn14_DecisionLevelMax": "Cnn14DecisionLevelMax",
    "Cnn14_DecisionLevelAvg": "Cnn14DecisionLevelAvg",
    "Cnn14_DecisionLevelAtt": "Cnn14DecisionLevelAtt",
}

SUPPORTED_MODEL_TYPES: tuple[str, ...] = tuple(_MODEL_CLASS_NAMES)

__all__ = ["SUPPORTED_MODEL_TYPES", "get_model_class"]


def get_model_class(model_type: str) -> type[nn.Module]:
    """Resolve a checkpoint name to its vendored architecture class."""
    class_name = _MODEL_CLASS_NAMES.get(model_type)
    if class_name is None:
        msg = f"Unknown SED model_type {model_type!r}; expected one of {list(_MODEL_CLASS_NAMES)}"
        raise ValueError(msg)

    from nemo_curator.models.sed import cnn14

    return getattr(cnn14, class_name)
