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

"""Tests for adapter-independent ASR waveform conversion."""

import numpy as np
import pytest

from nemo_curator.models.asr.waveform import resample_waveform, to_mono_numpy_1d


def test_to_mono_accepts_channels_first_and_channels_last() -> None:
    channels_first = np.asarray([[1.0, 3.0, 5.0], [3.0, 5.0, 7.0]], dtype=np.float32)
    channels_last = channels_first.T

    expected = np.asarray([2.0, 4.0, 6.0], dtype=np.float32)
    np.testing.assert_array_equal(to_mono_numpy_1d(channels_first), expected)
    np.testing.assert_array_equal(to_mono_numpy_1d(channels_last), expected)


def test_to_mono_rejects_higher_rank_waveform() -> None:
    with pytest.raises(ValueError, match="Expected 1-D or 2-D waveform"):
        to_mono_numpy_1d(np.zeros((2, 2, 2), dtype=np.float32))


def test_resample_returns_same_object_when_rate_matches() -> None:
    waveform = np.zeros(16_000, dtype=np.float32)
    assert resample_waveform(waveform, 16_000, 16_000) is waveform


def test_resample_rejects_invalid_rates() -> None:
    with pytest.raises(ValueError, match="Sample rates must be positive"):
        resample_waveform(np.zeros(1, dtype=np.float32), 0, 16_000)
