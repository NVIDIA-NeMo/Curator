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

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo_curator.stages.audio.common import (
    MAX_AUDIO_DURATION_SEC,
    AudioTooLongError,
    ensure_mono,
    ensure_waveform_2d,
    exceeds_max_duration,
    load_audio_file,
    resample_waveform,
    resolve_model_path,
)


def test_ensure_waveform_2d_from_tensor() -> None:
    assert ensure_waveform_2d(torch.randn(16000)).shape == (1, 16000)


def test_ensure_waveform_2d_from_numpy() -> None:
    assert ensure_waveform_2d(np.random.default_rng(0).standard_normal(16000).astype(np.float32)).dim() == 2


def test_ensure_mono() -> None:
    assert ensure_mono(torch.randn(2, 16000)).shape == (1, 16000)


def test_load_audio_file_reads_slices_downmixes_and_refuses_long_files(tmp_path: Path) -> None:
    stereo = np.stack([np.arange(8, dtype=np.float32), -np.arange(8, dtype=np.float32)], axis=1) / 8
    path = str(tmp_path / "test.wav")
    sf.write(path, stereo, 4, subtype="FLOAT")

    waveform, sr = load_audio_file(path, mono=False, start=2, stop=5)
    assert sr == 4
    assert torch.equal(waveform, torch.from_numpy(stereo[2:5]).T)
    assert torch.equal(load_audio_file(path)[0], torch.zeros(1, 8))
    with pytest.raises(AudioTooLongError):
        load_audio_file(path, max_duration_sec=1)
    assert load_audio_file(path, max_duration_sec=None)[0].shape == (1, 8)


def test_exceeds_max_duration() -> None:
    assert exceeds_max_duration(MAX_AUDIO_DURATION_SEC + 1)
    assert not exceeds_max_duration(MAX_AUDIO_DURATION_SEC)
    assert not exceeds_max_duration(10, None)
    assert not exceeds_max_duration("n/a")


def test_resample_waveform() -> None:
    waveform = torch.randn(1, 16000)
    assert resample_waveform(waveform, 16000, 16000) is waveform
    assert resample_waveform(waveform.numpy(), 16000, 8000).shape == (1, 8000)


def test_resolve_model_path(tmp_path: Path) -> None:
    assert resolve_model_path("/abs/model.bin", __file__, "sub") == "/abs/model.bin"

    module_dir = tmp_path / "sub"
    module_dir.mkdir()
    (module_dir / "model.bin").write_bytes(b"\x00")
    result = resolve_model_path("model.bin", str(tmp_path / "ref.py"), "sub")
    assert result == str(module_dir / "model.bin")
