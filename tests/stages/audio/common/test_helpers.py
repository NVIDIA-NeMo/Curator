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

from pathlib import Path
from unittest import mock

import numpy as np
import torch

from nemo_curator.stages.audio.common import (
    ensure_mono,
    ensure_waveform_2d,
    load_audio_file,
    resolve_model_path,
    resolve_waveform_from_item,
)


def test_ensure_waveform_2d_from_tensor() -> None:
    assert ensure_waveform_2d(torch.randn(16000)).shape == (1, 16000)


def test_ensure_waveform_2d_from_numpy() -> None:
    assert ensure_waveform_2d(np.random.default_rng(0).standard_normal(16000).astype(np.float32)).dim() == 2


def test_ensure_mono() -> None:
    assert ensure_mono(torch.randn(2, 16000)).shape == (1, 16000)


def test_load_audio_file(tmp_path: Path) -> None:
    fake_data = np.random.default_rng(0).standard_normal(32000).astype(np.float32)
    with mock.patch("nemo_curator.stages.audio.common.soundfile.read", return_value=(fake_data, 16000)):
        waveform, sr = load_audio_file(str(tmp_path / "test.wav"), mono=True)
        assert sr == 16000
        assert waveform.shape == (1, 32000)


def test_resolve_waveform_with_data() -> None:
    item = {"waveform": torch.randn(1, 16000), "sample_rate": 16000}
    result = resolve_waveform_from_item(item, "test")
    assert result is not None
    assert result[1] == 16000


def test_resolve_waveform_from_file(tmp_path: Path) -> None:
    wav_path = str(tmp_path / "audio.wav")
    Path(wav_path).write_bytes(b"\x00")
    with mock.patch("nemo_curator.stages.audio.common.load_audio_file", return_value=(torch.randn(1, 16000), 16000)):
        item = {"audio_filepath": wav_path}
        result = resolve_waveform_from_item(item, "test")
        assert result is not None
        assert item["waveform"] is not None


def test_resolve_waveform_returns_none_when_missing() -> None:
    assert resolve_waveform_from_item({}, "test") is None
    assert resolve_waveform_from_item({"audio_filepath": "/nonexistent.wav"}, "test") is None
    assert resolve_waveform_from_item({"waveform": torch.randn(16000)}, "test") is None


def test_resolve_model_path(tmp_path: Path) -> None:
    assert resolve_model_path("/abs/model.bin", __file__, "sub") == "/abs/model.bin"

    module_dir = tmp_path / "sub"
    module_dir.mkdir()
    (module_dir / "model.bin").write_bytes(b"\x00")
    result = resolve_model_path("model.bin", str(tmp_path / "ref.py"), "sub")
    assert result == str(module_dir / "model.bin")
