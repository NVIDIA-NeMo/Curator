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

"""Regression coverage for the gpu_memory_gb -> fractional-GPU derivation.

Resources(gpu_memory_gb=N) divides N by the device's total memory and rounds to
one decimal. On devices larger than ~200 GB a small request rounded to 0.0, so
the stage was scheduled CPU-only and failed with "No CUDA GPUs are available"
(NvBug 6783908, first seen on a 256 GB GB300). These tests pin the derived
fraction -- not just the stored gpu_memory_gb value -- across device sizes.
"""

from unittest.mock import patch

import pytest

import nemo_curator.stages.resources as resources_mod
from nemo_curator.stages.resources import Resources

_GET_MEM = "nemo_curator.stages.resources._get_gpu_memory_gb"

# Resolved via getattr so that, if the floor constant is ever removed, the tests
# below fail on their assertions (fraction == 0.0) instead of on an ImportError.
MIN_GPU_FRACTION = getattr(resources_mod, "MIN_GPU_FRACTION", 0.1)

# Device totals as reported by pynvml (bytes / 1024**3) on real hardware.
A100_80GB = 80.0
H100_80GB = 80.0
H200_NVL_141GB = 143.77
GB300_256GB = 250.69


@pytest.mark.parametrize(
    ("device_gb", "requested_gb"),
    [
        # Below 5% of the device: round(ratio, 1) == 0.0 before the fix.
        (GB300_256GB, 10.0),  # 0.0399 -- the quickstart.py request that failed on GB300
        (GB300_256GB, 12.0),  # 0.0479 -- just under the 12.5 GB threshold on GB300
        (GB300_256GB, 1.0),  # 0.0040
        (H200_NVL_141GB, 5.0),  # 0.0348 -- H200 is exposed too, not only GB300
        (A100_80GB, 3.0),  # 0.0375 -- even an 80 GB device with a tiny request
    ],
)
def test_small_request_on_large_device_floors_at_min_fraction(device_gb: float, requested_gb: float) -> None:
    """A positive memory request must never resolve to a zero-GPU reservation."""
    with patch(_GET_MEM, return_value=device_gb):
        res = Resources(gpu_memory_gb=requested_gb)
    assert res.gpus == MIN_GPU_FRACTION
    assert res.gpus > 0.0
    assert res.requires_gpu


@pytest.mark.parametrize(
    ("device_gb", "requested_gb", "expected_gpus"),
    [
        (A100_80GB, 10.0, 0.1),  # 0.125 -> 0.1  (unchanged from pre-fix behaviour)
        (H100_80GB, 40.0, 0.5),  # 0.5
        (A100_80GB, 72.0, 0.9),  # 0.9
        (GB300_256GB, 20.0, 0.1),  # 0.0798 -> 0.1 with or without the floor
        (GB300_256GB, 125.0, 0.5),  # 0.4986 -> 0.5
        (H200_NVL_141GB, 72.0, 0.5),  # 0.5008 -> 0.5
    ],
)
def test_rounding_preserved_when_ratio_already_at_or_above_floor(
    device_gb: float, requested_gb: float, expected_gpus: float
) -> None:
    """The floor must not change results that already rounded to >= 0.1."""
    with patch(_GET_MEM, return_value=device_gb):
        res = Resources(gpu_memory_gb=requested_gb)
    assert res.gpus == expected_gpus
    assert res.gpu_memory_gb == requested_gb  # original request is retained


def test_min_fraction_matches_rounding_granularity() -> None:
    """The floor is the smallest value round(x, 1) can produce above zero."""
    assert hasattr(resources_mod, "MIN_GPU_FRACTION"), "MIN_GPU_FRACTION removed from resources.py"
    assert MIN_GPU_FRACTION == 0.1
    assert round(MIN_GPU_FRACTION, 1) == MIN_GPU_FRACTION


@pytest.mark.parametrize(
    ("device_gb", "requested_gb"),
    [
        (A100_80GB, 100.0),  # 1.25 > 1
        (A100_80GB, 88.0),  # 1.1  > 1
        (GB300_256GB, 300.0),  # 1.2  > 1
    ],
)
def test_oversized_request_still_raises(device_gb: float, requested_gb: float) -> None:
    """The floor must not weaken the 'too large for a single GPU' guard."""
    with (
        patch(_GET_MEM, return_value=device_gb),
        pytest.raises(ValueError, match="too large for a single GPU"),
    ):
        Resources(gpu_memory_gb=requested_gb)


def test_exactly_one_full_gpu_is_allowed() -> None:
    with patch(_GET_MEM, return_value=A100_80GB):
        assert Resources(gpu_memory_gb=80.0).gpus == 1.0


@pytest.mark.parametrize("device_gb", [A100_80GB, GB300_256GB])
def test_explicit_gpus_bypass_memory_derivation(device_gb: float) -> None:
    """gpus= is honoured verbatim and never touched by the floor or the device size."""
    with patch(_GET_MEM, return_value=device_gb):
        assert Resources(gpus=2.0).gpus == 2.0
        assert Resources(gpus=0.5).gpus == 0.5
        assert Resources(gpus=0.05).gpus == 0.05  # below MIN_GPU_FRACTION on purpose: explicit wins


def test_cpu_only_stage_has_zero_gpus() -> None:
    with patch(_GET_MEM, return_value=GB300_256GB):
        res = Resources(cpus=2.0)
    assert res.gpus == 0.0
    assert res.gpu_memory_gb == 0.0
    assert not res.requires_gpu


def test_gpus_and_gpu_memory_gb_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="Cannot specify both gpus and gpu_memory_gb"):
        Resources(gpus=1.0, gpu_memory_gb=10.0)


def test_memory_detection_fallback_still_floors() -> None:
    """If pynvml is unavailable the module falls back to 24 GB; the floor still applies."""
    with patch("nemo_curator.stages.resources._get_gpu_memory_gb", return_value=24.0):
        assert Resources(gpu_memory_gb=1.0).gpus == MIN_GPU_FRACTION  # 0.0417 -> floor
        assert Resources(gpu_memory_gb=12.0).gpus == 0.5
