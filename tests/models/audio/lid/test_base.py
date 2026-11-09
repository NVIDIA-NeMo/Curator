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

import subprocess
import sys

import numpy as np
import pytest

from nemo_curator.models.audio.lid.base import (
    AudioLIDAdapter,
    AudioLIDResult,
    _validate_sample_rate,
    _validate_single_device_gpu_count,
    _waveform_from_item,
)
from nemo_curator.models.audio.lid.speechbrain import SpeechBrainLIDAdapter


def test_result_normalizes_language_and_confidence() -> None:
    result = AudioLIDResult("  EN-us ", 1)

    assert result == AudioLIDResult(language="en-us", confidence=1.0)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("inf"), float("nan")])
def test_result_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        AudioLIDResult("en", confidence)


def test_result_rejects_non_string_language() -> None:
    with pytest.raises(TypeError, match="language must be a string"):
        AudioLIDResult(None, 0.0)


@pytest.mark.parametrize("sample_rate", [True, 0, -1, 16_000.5, "16000"])
def test_sample_rate_rejects_lossy_or_invalid_values(sample_rate: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _validate_sample_rate(sample_rate, owner="test")


@pytest.mark.parametrize(("num_gpus", "gpu_required"), [(True, False), (0.5, False), (2, False), (0, True)])
def test_gpu_count_rejects_invalid_allocations(num_gpus: object, gpu_required: bool) -> None:
    with pytest.raises(ValueError, match="requires num_gpus"):
        _validate_single_device_gpu_count(num_gpus, owner="test", gpu_required=gpu_required)


def test_waveform_boundary_rejects_missing_or_non_mono_input() -> None:
    with pytest.raises(KeyError, match="contain 'waveform'"):
        _waveform_from_item({}, owner="test")
    with pytest.raises(ValueError, match="mono 1-D"):
        _waveform_from_item({"waveform": np.zeros((2, 10))}, owner="test")


def test_waveform_boundary_preserves_samples_in_contiguous_float32_output() -> None:
    waveform = np.arange(8, dtype=np.float64)[::2]

    result = _waveform_from_item({"waveform": waveform}, owner="test")

    np.testing.assert_array_equal(result, waveform)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous


def test_provider_adapter_conforms_to_shared_protocol() -> None:
    assert isinstance(SpeechBrainLIDAdapter(), AudioLIDAdapter)


def test_package_import_does_not_load_optional_model_runtimes() -> None:
    probe = (
        "import sys;"
        "import nemo_curator.models.audio.lid;"
        "blocked=('speechbrain','whisper','nemo.collections.asr','tensorrt_llm');"
        "print(','.join(name for name in blocked if name in sys.modules))"
    )
    result = subprocess.run(  # noqa: S603 - fixed probe under the test interpreter
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == ""
