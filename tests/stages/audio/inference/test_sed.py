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

"""Minimal tests for SED inference logic.

Tests the CNN14 model output format, NPZ saving, and padding behaviour
using a lightweight mock model.

Run: pytest tests/stages/audio/inference/test_sed.py -v --noconftest
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Direct module import (bypasses nemo_curator.__init__ chain for Py3.9 compat)
# ---------------------------------------------------------------------------
def _import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_cnn14_path = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "nemo_curator", "stages", "audio", "inference", "sed_models", "cnn14.py",
)
_cnn14 = _import_from_path("_test_cnn14", os.path.abspath(_cnn14_path))

MODEL_REGISTRY = _cnn14.MODEL_REGISTRY
Cnn14DecisionLevelMax = _cnn14.Cnn14DecisionLevelMax
interpolate = _cnn14.interpolate
pad_framewise_output = _cnn14.pad_framewise_output


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSES_NUM = 527
SAMPLE_RATE = 16000
HOP_SIZE = 320


# ---------------------------------------------------------------------------
# Mock model & helpers
# ---------------------------------------------------------------------------


class MockCnn14(nn.Module):
    """Tiny model returning random framewise output of correct shape."""

    def __init__(self, classes_num: int = CLASSES_NUM, hop_size: int = HOP_SIZE) -> None:
        super().__init__()
        self.classes_num = classes_num
        self.hop_size = hop_size
        self._dummy = nn.Linear(1, 1)

    def forward(self, input: torch.Tensor, mixup_lambda=None) -> dict[str, torch.Tensor]:
        batch, samples = input.shape
        frames = samples // self.hop_size + 1
        fw = torch.rand(batch, frames, self.classes_num)
        return {"framewise_output": fw, "clipwise_output": fw.mean(dim=1)}


def _make_wav(tmp_path: Path, duration_sec: float = 1.0, sr: int = SAMPLE_RATE) -> Path:
    num_samples = int(sr * duration_sec)
    data = struct.pack(f"<{num_samples}h", *([0] * num_samples))
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HHIIHH", 1, 1, sr, sr * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data)))
    buf.write(data)
    p = tmp_path / "test.wav"
    p.write_bytes(buf.getvalue())
    return p


def _run_sed(model: nn.Module, audio_path: str, output_dir: str, pad: bool = True) -> str:
    """Reproduce the core SED inference + NPZ save logic from SEDInferenceStage."""
    waveform, sr = sf.read(audio_path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    original_samples = waveform.shape[0]

    min_input = max(1024, HOP_SIZE * 32)
    was_padded = False
    if pad and original_samples < min_input:
        waveform = np.pad(waveform, (0, min_input - original_samples), mode="constant")
        was_padded = True

    x = torch.from_numpy(waveform[None, :]).float()
    with torch.no_grad():
        out = model(x, None)
    framewise = out["framewise_output"].cpu().numpy()[0]
    fps = float(SAMPLE_RATE) / HOP_SIZE
    valid_frames = min(int(np.ceil(original_samples / HOP_SIZE)), framewise.shape[0])

    fw = framewise.astype(np.float16)
    fw_dir = os.path.join(output_dir, "framewise")
    os.makedirs(fw_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    h = hashlib.md5(audio_path.encode("utf-8")).hexdigest()[:8]
    npz_path = os.path.join(fw_dir, f"{stem}__{h}.npz")
    np.savez_compressed(
        npz_path,
        framewise=fw,
        fps=np.float32(fps),
        audio_filepath=str(audio_path),
        original_num_samples=np.int32(original_samples),
        valid_frames=np.int32(valid_frames),
        was_padded=np.bool_(was_padded),
    )
    return npz_path


# ---------------------------------------------------------------------------
# Tests: NPZ output format
# ---------------------------------------------------------------------------


class TestNPZOutput:
    """1s audio -> NPZ with correct shape (T, 527) and metadata."""

    def test_framewise_shape(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        npz = _run_sed(MockCnn14(), str(wav), str(tmp_path / "out"))
        with np.load(npz) as d:
            assert d["framewise"].ndim == 2
            assert d["framewise"].shape[1] == CLASSES_NUM

    def test_metadata_fields(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        npz = _run_sed(MockCnn14(), str(wav), str(tmp_path / "out"))
        with np.load(npz) as d:
            assert "fps" in d
            assert "valid_frames" in d
            assert "audio_filepath" in d
            assert "was_padded" in d
            assert float(d["fps"]) == pytest.approx(SAMPLE_RATE / HOP_SIZE)

    def test_dtype_float16(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        npz = _run_sed(MockCnn14(), str(wav), str(tmp_path / "out"))
        with np.load(npz) as d:
            assert d["framewise"].dtype == np.float16

    def test_normal_audio_not_padded(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path, duration_sec=1.0)
        npz = _run_sed(MockCnn14(), str(wav), str(tmp_path / "out"))
        with np.load(npz) as d:
            assert bool(d["was_padded"]) is False


class TestPadding:
    """Very short audio is zero-padded, metadata records it."""

    def test_short_audio_padded(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path, duration_sec=0.005)
        npz = _run_sed(MockCnn14(), str(wav), str(tmp_path / "out"))
        with np.load(npz) as d:
            assert bool(d["was_padded"]) is True


# ---------------------------------------------------------------------------
# Tests: Model registry & forward shape
# ---------------------------------------------------------------------------


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
