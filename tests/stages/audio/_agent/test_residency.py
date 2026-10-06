# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from nemo_curator.stages.audio._agent import _residency


@pytest.mark.parametrize("explicit_dir", [False, True])
def test_failed_audio_write_removes_only_owned_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_dir: bool
) -> None:
    unrelated = tmp_path / "existing.wav"
    unrelated.write_bytes(b"keep")
    monkeypatch.setattr(_residency.tempfile, "tempdir", str(tmp_path))

    def fail_write(path: str, *_args: object) -> None:
        Path(path).write_bytes(b"partial")
        message = "disk write failed"
        raise OSError(message)

    monkeypatch.setattr(_residency.sf, "write", fail_write)
    with pytest.raises(OSError, match="disk write failed"):
        _residency.write_audio_stable(
            np.zeros((1, 16), dtype=np.float32), 16000, output_dir=str(tmp_path) if explicit_dir else None
        )
    assert list(tmp_path.iterdir()) == [unrelated]
    assert unrelated.read_bytes() == b"keep"


@pytest.mark.parametrize("explicit_dir", [False, True])
@pytest.mark.parametrize("stem", ["a" * 240, "音" * 80, "ordinary"])
def test_audio_export_bounds_filename_and_preserves_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_dir: bool, stem: str
) -> None:
    import os

    import soundfile as sf

    monkeypatch.setattr(_residency.tempfile, "tempdir", str(tmp_path))
    waveform = np.zeros((1, 160), dtype=np.float32)
    path = _residency.write_audio_stable(
        waveform, 16000, output_dir=str(tmp_path) if explicit_dir else None, stem=stem, tag="mono"
    )
    assert len(os.fsencode(Path(path).name)) <= os.pathconf(tmp_path, "PC_NAME_MAX")
    audio, rate = sf.read(path)
    assert rate == 16000
    np.testing.assert_array_equal(audio, waveform[0])
    if stem == "ordinary":
        assert Path(path).name.startswith("ordinary_mono_")
    if explicit_dir:
        assert path == _residency.write_audio_stable(waveform, 16000, output_dir=str(tmp_path), stem=stem, tag="mono")


def test_dtype_preservation_is_opt_in_for_normalizing_consumers() -> None:
    import torch

    data = {"waveform": np.array([32767, -32768], dtype=np.int16), "sample_rate": 16000}
    ordinary, _ = _residency.resolve_audio(data, residency="waveform")
    pcm, _ = _residency.resolve_audio(data, residency="waveform", preserve_pcm_dtype=True)
    assert ordinary.dtype == torch.float32
    assert pcm.dtype == torch.int16
    torch.testing.assert_close(ordinary, pcm.float())


def test_pcm_preservation_keeps_untyped_lists_as_float_samples() -> None:
    import torch

    waveform, rate = _residency.resolve_audio(
        {"waveform": [0, 1, -1], "sample_rate": 16000},
        residency="waveform",
        preserve_pcm_dtype=True,
    )
    assert rate == 16000
    assert waveform.dtype == torch.float32
    assert waveform.tolist() == [[0.0, 1.0, -1.0]]
