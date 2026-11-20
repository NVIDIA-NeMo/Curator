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

import numpy as np
import pytest
import torch

from nemo_curator.stages.audio.metrics._common import (
    metrics_mapping,
    resident_pair_is_complete,
    resident_pcm_to_mono_float32,
    resident_sample_rate,
    validate_metric_keys,
)


def test_validate_metric_keys_rejects_empty_and_duplicate_keys() -> None:
    validate_metric_keys("S", keys={"a": "x", "b": "y"}, strict_fields=("a", "b"))
    with pytest.raises(ValueError, match="'a' must be a non-empty string"):
        validate_metric_keys("S", keys={"a": " ", "b": "y"}, strict_fields=("a", "b"))
    with pytest.raises(ValueError, match="must be distinct"):
        validate_metric_keys("S", keys={"a": "x", "b": "x"}, strict_fields=("a", "b"))


@pytest.mark.parametrize(
    ("value", "expected"), [(16000, 16000), ("16000", 16000), (16000.0, 16000), (torch.tensor(8000), 8000)]
)
def test_resident_sample_rate_accepts_lossless_integers(value: object, expected: int) -> None:
    assert resident_sample_rate(value, sample_rate_key="sr", stage_name="S") == expected


@pytest.mark.parametrize("value", [True, 0, -1, 16000.5, "16k", None])
def test_resident_sample_rate_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="Resident sample rate 'sr'"):
        resident_sample_rate(value, sample_rate_key="sr", stage_name="S")


def test_resident_pair_is_complete() -> None:
    kwargs = {"waveform_key": "w", "sample_rate_key": "sr", "stage_name": "S"}
    full = {"w": np.zeros(4), "sr": 16000}
    assert resident_pair_is_complete(full, residency="auto", **kwargs) is True
    assert resident_pair_is_complete(full, residency="file", **kwargs) is False
    assert resident_pair_is_complete({}, residency="waveform", **kwargs) is False
    with pytest.raises(ValueError, match="'w' is present but 'sr' is missing"):
        resident_pair_is_complete({"w": np.zeros(4)}, residency="auto", **kwargs)


def test_resident_pcm_to_mono_float32_scales_pcm_and_downmixes() -> None:
    pcm = np.array([[16384, -32768], [16384, 0]], dtype=np.int16)
    np.testing.assert_allclose(resident_pcm_to_mono_float32(pcm, stage_name="S"), [0.5, -0.5])
    tensor = torch.tensor([[2147483648 // 2]], dtype=torch.int32)
    np.testing.assert_allclose(resident_pcm_to_mono_float32(tensor, stage_name="S"), [0.5])
    out = resident_pcm_to_mono_float32(np.ones(3, dtype=np.float64), stage_name="S")
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]


def test_resident_pcm_to_mono_float32_rejects_bad_input() -> None:
    with pytest.raises(TypeError, match="Unsupported resident waveform dtype"):
        resident_pcm_to_mono_float32(np.zeros(3, dtype=np.uint8), stage_name="S")
    with pytest.raises(ValueError, match="1-D or 2-D"):
        resident_pcm_to_mono_float32(torch.zeros(1, 1, 3), stage_name="S")


def test_metrics_mapping() -> None:
    existing = {"pesq": 3.0}
    assert metrics_mapping({"metrics": existing}, metrics_key="metrics", stage_name="S") is existing
    item: dict = {}
    assert metrics_mapping(item, metrics_key="metrics", stage_name="S") == {}
    assert "metrics" not in item
    with pytest.raises(TypeError, match="must be a MutableMapping"):
        metrics_mapping({"metrics": [1]}, metrics_key="metrics", stage_name="S")
