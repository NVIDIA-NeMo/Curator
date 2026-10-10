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


from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo_curator.stages.audio._agent import _residency
from nemo_curator.stages.audio._agent._residency import (
    cleanup_temp_files,
    resolve_audio,
    resolve_audio_path,
    validate_audio_key_configuration,
    validate_input_residency,
    write_audio_stable,
)


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


@pytest.mark.parametrize("residency", ["file", "waveform", "auto"])
def test_input_residency_validator_accepts_only_declared_modes(residency: str) -> None:
    validate_input_residency(residency, stage_name="Fixture")


def test_input_residency_validator_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="input_residency must be one of"):
        validate_input_residency("wavefrom", stage_name="Fixture")


@pytest.mark.parametrize(
    "sample_rate",
    [
        pytest.param(True, id="bool"),
        pytest.param(np.bool_(True), id="numpy-bool"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(16000.5, id="fractional-float"),
        pytest.param("16000.5", id="fractional-string"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
        pytest.param(torch.tensor([16000]), id="non-scalar-tensor"),
    ],
)
def test_resolve_audio_rejects_invalid_resident_sample_rates(sample_rate: object) -> None:
    with pytest.raises(ValueError, match="positive, losslessly integral, non-boolean"):
        resolve_audio({"waveform": torch.zeros(8), "sample_rate": sample_rate})


@pytest.mark.parametrize(
    "sample_rate",
    [
        pytest.param(16000, id="int"),
        pytest.param(np.int64(16000), id="numpy-int"),
        pytest.param(16000.0, id="integral-float"),
        pytest.param("16000", id="numeric-string"),
        pytest.param(torch.tensor(16000), id="scalar-tensor"),
    ],
)
def test_resolve_audio_preserves_lossless_sample_rate_coercions(sample_rate: object) -> None:
    resolved = resolve_audio({"waveform": torch.zeros(8), "sample_rate": sample_rate})

    assert resolved is not None
    assert resolved[1] == 16000
    assert isinstance(resolved[1], int)


def test_audio_key_validator_rejects_input_role_aliases() -> None:
    with pytest.raises(ValueError, match="Audio input keys must be distinct"):
        validate_audio_key_configuration(
            "Fixture",
            input_keys={"waveform_key": "audio", "sample_rate_key": "audio"},
            output_keys={"score_key": "score"},
        )


def test_file_audio_hydration_policies_are_opt_in_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    path.touch()
    loaded = torch.arange(8, dtype=torch.float32).unsqueeze(0)

    def loader(_path: str, *, mono: bool) -> tuple[torch.Tensor, int]:
        assert mono
        return loaded, 16000

    untouched = {"audio_filepath": str(path)}
    resolve_audio(untouched, residency="file", loader=loader)
    assert set(untouched) == {"audio_filepath"}

    always = {"audio_filepath": str(path)}
    resolve_audio(always, residency="file", loader=loader, file_audio_hydration="always")
    assert always["waveform"] is loaded
    assert always["sample_rate"] == 16000

    partial = {"audio_filepath": str(path), "sample_rate": 8000}
    resolve_audio(partial, residency="auto", loader=loader, file_audio_hydration="auto_partial")
    assert partial["waveform"] is loaded
    assert partial["sample_rate"] == 16000

    ordinary_auto = {"audio_filepath": str(path)}
    resolve_audio(ordinary_auto, residency="auto", loader=loader, file_audio_hydration="auto_partial")
    assert set(ordinary_auto) == {"audio_filepath"}


def test_failed_file_hydration_preserves_the_existing_pair(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    path.touch()
    stale = torch.ones(1, 4)
    item = {"audio_filepath": str(path), "waveform": stale, "sample_rate": 8000}

    def failing_loader(_path: str, *, mono: bool) -> tuple[torch.Tensor, int]:
        assert mono
        msg = "decode failed"
        raise OSError(msg)

    with pytest.raises(OSError, match="decode failed"):
        resolve_audio(
            item,
            residency="file",
            loader=failing_loader,
            file_audio_hydration="always",
        )

    assert item["waveform"] is stale
    assert item["sample_rate"] == 8000


def test_resolve_audio_path_auto_prefers_complete_resident_audio(tmp_path: Path) -> None:
    """Auto residency must not silently choose a stale file over a complete waveform."""
    file_path = tmp_path / "one_second.wav"
    sf.write(file_path, torch.zeros(16000).numpy(), 16000)
    resident = torch.ones(1, 32000)
    item = {
        "audio_filepath": str(file_path),
        "waveform": resident,
        "sample_rate": 16000,
    }
    temporary_paths: list[str] = []

    resolved = resolve_audio_path(item, residency="auto", temp_dir=str(tmp_path), register_temp=temporary_paths)

    assert resolved is not None
    assert resolved != str(file_path)
    assert temporary_paths == [resolved]
    loaded, sample_rate = sf.read(resolved)
    assert sample_rate == 16000
    assert len(loaded) == 32000
    assert loaded.mean() > 0.9
    assert resolve_audio_path(item, residency="file") == str(file_path)

    cleanup_temp_files(temporary_paths)
    assert not os.path.exists(resolved)


def test_resolve_audio_path_preserves_float_waveform_samples(tmp_path: Path) -> None:
    waveform = np.array([[1e-5, -1e-5, 1.25, -1.25]], dtype=np.float32)
    temporary_paths: list[str] = []

    resolved = resolve_audio_path(
        {"waveform": waveform, "sample_rate": 16000},
        residency="waveform",
        temp_dir=str(tmp_path),
        register_temp=temporary_paths,
    )

    observed, sample_rate = sf.read(resolved, dtype="float32", always_2d=True)
    assert sample_rate == 16000
    assert sf.info(resolved).subtype == "FLOAT"
    np.testing.assert_array_equal(observed[:, 0], waveform[0])
    cleanup_temp_files(temporary_paths)


def test_stable_audio_names_include_layout_and_written_short_stereo_shape(tmp_path: Path) -> None:
    """Different channel layouts with identical samples need distinct artifacts."""
    output_dir = str(tmp_path)
    mono = torch.zeros(1, 32000)
    stereo = torch.zeros(2, 16000)

    mono_path = write_audio_stable(mono, 16000, output_dir=output_dir, stem="audio")
    stereo_path = write_audio_stable(stereo, 16000, output_dir=output_dir, stem="audio")

    assert mono_path != stereo_path
    assert sf.info(mono_path).channels == 1
    assert sf.info(stereo_path).channels == 2

    short_stereo_path = write_audio_stable(
        torch.tensor([[0.25], [0.75]]),
        16000,
        output_dir=output_dir,
        stem="short",
    )
    short_info = sf.info(short_stereo_path)
    assert (short_info.frames, short_info.channels) == (1, 2)
