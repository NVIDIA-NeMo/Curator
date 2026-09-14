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

"""Tests for the SED model registry and the vendored PANNs architectures."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from nemo_curator.models import sed

_CNN14_SOURCE = Path(sed.__file__).parent / "cnn14.py"


def test_the_three_decision_level_variants_are_supported() -> None:
    assert set(sed.SUPPORTED_MODEL_TYPES) == {
        "Cnn14_DecisionLevelMax",
        "Cnn14_DecisionLevelAvg",
        "Cnn14_DecisionLevelAtt",
    }


def test_an_unknown_model_type_is_rejected_by_name() -> None:
    with pytest.raises(ValueError, match="Unknown SED model_type 'NotAModel'"):
        sed.get_model_class("NotAModel")


def test_the_error_lists_what_is_available() -> None:
    with pytest.raises(ValueError, match="Cnn14_DecisionLevelMax"):
        sed.get_model_class("NotAModel")


def test_importing_the_sed_stage_does_not_pull_in_torchlibrosa() -> None:
    """The stage must import without audio_cuda12; only building a model needs torchlibrosa.

    Checked in a clean interpreter, since another test in this session may
    already have imported torchlibrosa.
    """
    probe = (
        "import sys;"
        "import nemo_curator.stages.audio.inference.sed.stage;"
        "import nemo_curator.models.sed as m;"
        "m.SUPPORTED_MODEL_TYPES;"
        "print('torchlibrosa' in sys.modules)"
    )
    result = subprocess.run(  # noqa: S603 - fixed probe run under the test interpreter
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_listing_supported_types_needs_no_heavy_dependency() -> None:
    assert tuple(sed._MODEL_CLASS_NAMES) == sed.SUPPORTED_MODEL_TYPES


@pytest.mark.gpu
@pytest.mark.parametrize("model_type", ["Cnn14_DecisionLevelMax", "Cnn14_DecisionLevelAvg", "Cnn14_DecisionLevelAtt"])
def test_each_variant_resolves_to_a_torch_module(model_type: str) -> None:
    assert issubclass(sed.get_model_class(model_type), nn.Module)


@pytest.mark.gpu
@pytest.mark.parametrize("model_type", ["Cnn14_DecisionLevelMax", "Cnn14_DecisionLevelAvg", "Cnn14_DecisionLevelAtt"])
def test_each_variant_emits_framewise_and_clipwise_output(model_type: str) -> None:
    """SED needs framewise output; the base Cnn14 gives clip-level only."""
    classes_num = 8
    model = sed.get_model_class(model_type)(
        sample_rate=16000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=7800,
        classes_num=classes_num,
    ).eval()

    with torch.no_grad():
        # The nkoluguri reference stage passes the unused mixup argument
        # positionally; preserve that exact callable contract.
        out = model(torch.zeros(2, 16000), None)

    assert out["framewise_output"].shape[0] == 2
    assert out["framewise_output"].shape[2] == classes_num
    assert out["clipwise_output"].shape == (2, classes_num)


@pytest.mark.gpu
def test_framewise_output_is_a_probability() -> None:
    model = sed.get_model_class("Cnn14_DecisionLevelMax")(
        sample_rate=16000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=7800, classes_num=8
    ).eval()

    with torch.no_grad():
        framewise = model(torch.zeros(1, 16000))["framewise_output"]

    assert bool((framewise >= 0).all())
    assert bool((framewise <= 1).all())


def test_a_missing_torchlibrosa_names_the_extra_to_install() -> None:
    """The import guard has to be actionable; 'no module named torchlibrosa' is not."""
    source = _CNN14_SOURCE.read_text(encoding="utf-8")
    assert "audio_cuda12" in source
    assert "raise ImportError" in source


def test_the_vendored_model_carries_its_upstream_mit_license() -> None:
    header = _CNN14_SOURCE.read_text(encoding="utf-8")[:2000]
    assert "The MIT License" in header
    assert "Qiuqiang Kong" in header
    assert "WITHOUT WARRANTY OF ANY KIND" in header


def test_the_vendored_model_names_its_upstream_source() -> None:
    source = _CNN14_SOURCE.read_text(encoding="utf-8")
    assert "audioset_tagging_cnn/blob/d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4/pytorch/models.py" in source
    assert "audioset_tagging_cnn/blob/d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4/pytorch/pytorch_utils.py" in source
