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

"""Minimal tests for the vendored CNN14 SED model module.

Loads ``cnn14.py`` directly (bypassing the ``nemo_curator`` package import chain)
to test the model registry and framewise utility functions without pulling in the
full package. The real inference stage, its NPZ output, and its padding behaviour
are covered against the actual ``SEDInferenceStage`` in ``test_sed_stage.py``.

Run: pytest tests/stages/audio/inference/test_sed.py -v --noconftest
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from types import ModuleType


# ---------------------------------------------------------------------------
# Direct module import (bypasses nemo_curator.__init__ chain for Py3.9 compat)
# ---------------------------------------------------------------------------
def _import_from_path(module_name: str, file_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_cnn14_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "..",
    "nemo_curator",
    "stages",
    "audio",
    "inference",
    "sed_models",
    "cnn14.py",
)
_cnn14 = _import_from_path("_test_cnn14", os.path.abspath(_cnn14_path))

MODEL_REGISTRY = _cnn14.MODEL_REGISTRY
Cnn14DecisionLevelMax = _cnn14.Cnn14DecisionLevelMax
interpolate = _cnn14.interpolate
pad_framewise_output = _cnn14.pad_framewise_output

CLASSES_NUM = 527
SAMPLE_RATE = 16000
HOP_SIZE = 320


class TestModelRegistry:
    def test_three_variants(self) -> None:
        assert "Cnn14_DecisionLevelMax" in MODEL_REGISTRY
        assert "Cnn14_DecisionLevelAvg" in MODEL_REGISTRY
        assert "Cnn14_DecisionLevelAtt" in MODEL_REGISTRY

    @pytest.mark.skip(reason="Requires working librosa/numba (GLIBC compat)")
    def test_cnn14_forward_shape(self) -> None:
        model = Cnn14DecisionLevelMax(sample_rate=SAMPLE_RATE, hop_size=HOP_SIZE, classes_num=CLASSES_NUM)
        model.eval()
        x = torch.randn(1, SAMPLE_RATE)
        with torch.no_grad():
            out = model(x, None)
        assert "framewise_output" in out
        assert "clipwise_output" in out
        assert out["framewise_output"].shape[0] == 1
        assert out["framewise_output"].shape[2] == CLASSES_NUM
        assert out["clipwise_output"].shape == (1, CLASSES_NUM)


class TestUtilities:
    def test_interpolate(self) -> None:
        x = torch.rand(1, 4, 10)
        result = interpolate(x, ratio=32)
        assert result.shape == (1, 128, 10)

    def test_pad_framewise_output(self) -> None:
        fw = torch.rand(1, 50, 10)
        result = pad_framewise_output(fw, frames_num=60)
        assert result.shape == (1, 60, 10)
