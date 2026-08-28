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

"""Characterization tests pinning the exact behavior of the audio-resolution
helpers BEFORE consolidating them onto ``_residency.resolve_audio``.

These lock the current return shapes/types/values so the refactor of
``utmos._load_waveform_tensor`` and ``sigmos._get_audio_numpy_sr`` into thin
wrappers around ``resolve_audio`` is provably byte-identical (no behavior change).
``common.resolve_waveform_from_item`` is included to document its unique
sample-rate-from-header behavior (kept as-is for now).
"""

from __future__ import annotations

import os
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
import torch

from nemo_curator.stages.audio._agent._residency import cleanup_temp_files, resolve_audio, resolve_audio_path

if TYPE_CHECKING:
    import pytest
from nemo_curator.stages.audio.common import resolve_waveform_from_item
from nemo_curator.stages.audio.filtering.sigmos import _get_audio_numpy_sr
from nemo_curator.stages.audio.filtering.utmos import _load_waveform_tensor

_SR = 16000


def _wav(path: Path, *, channels: int = 1, n: int = 1600) -> str:
    data = np.linspace(-0.5, 0.5, n, dtype="float32")
    arr = data if channels == 1 else np.stack([data, data * 0.5], axis=1)  # (n, ch) for soundfile
    sf.write(str(path), arr, _SR)
    return str(path)


# --------------------------------------------------------------------------- #
# utmos._load_waveform_tensor  ->  torch (1, N) mono
# --------------------------------------------------------------------------- #
def test_utmos_waveform_stereo_tensor_to_mono():  # noqa: ANN202
    wf = torch.ones(2, 1600)
    out = _load_waveform_tensor({"waveform": wf, "sample_rate": _SR}, "t")
    assert out is not None
    t, sr = out
    assert torch.is_tensor(t) and t.shape == (1, 1600) and sr == _SR  # noqa: PT018


def test_utmos_waveform_numpy_1d_to_mono():  # noqa: ANN202
    out = _load_waveform_tensor({"waveform": np.ones(1600, dtype="float32"), "sample_rate": _SR}, "t")
    t, _ = out
    assert torch.is_tensor(t) and t.shape == (1, 1600)  # noqa: PT018


def test_utmos_waveform_present_no_sr_returns_none():  # noqa: ANN202
    assert _load_waveform_tensor({"waveform": torch.ones(1, 1600)}, "t") is None


def test_utmos_file_path_loads_mono(tmp_path: Path):  # noqa: ANN202
    out = _load_waveform_tensor({"audio_filepath": _wav(tmp_path / "u.wav", channels=2)}, "t")
    t, sr = out
    assert t.shape[0] == 1 and sr == _SR  # noqa: PT018


def test_utmos_residency_waveform_no_data_returns_none(tmp_path: Path):  # noqa: ANN202
    out = _load_waveform_tensor({"audio_filepath": _wav(tmp_path / "u2.wav")}, "t", input_residency="waveform")
    assert out is None


def test_utmos_missing_returns_none():  # noqa: ANN202
    assert _load_waveform_tensor({}, "t") is None


# --------------------------------------------------------------------------- #
# sigmos._get_audio_numpy_sr  ->  np.float32 1D mono
# --------------------------------------------------------------------------- #
def test_sigmos_waveform_stereo_tensor_to_mono_numpy():  # noqa: ANN202
    out = _get_audio_numpy_sr({"waveform": torch.ones(2, 1600), "sample_rate": _SR}, "t")
    a, sr = out
    assert isinstance(a, np.ndarray) and a.dtype == np.float32 and a.ndim == 1 and a.shape == (1600,) and sr == _SR  # noqa: PT018


def test_sigmos_waveform_numpy_to_mono():  # noqa: ANN202
    out = _get_audio_numpy_sr({"waveform": np.ones((2, 1600), dtype="float32"), "sample_rate": _SR}, "t")
    a, _ = out
    assert a.ndim == 1 and a.shape == (1600,)  # noqa: PT018


def test_sigmos_file_path_loads_1d_numpy(tmp_path: Path):  # noqa: ANN202
    out = _get_audio_numpy_sr({"audio_filepath": _wav(tmp_path / "s.wav", channels=2)}, "t")
    a, sr = out
    assert isinstance(a, np.ndarray) and a.ndim == 1 and sr == _SR  # noqa: PT018


def test_sigmos_waveform_present_no_sr_falls_back_to_file(tmp_path: Path):  # noqa: ANN202
    # sigmos requires BOTH wf+sr for the in-memory branch; otherwise tries the file.
    out = _get_audio_numpy_sr({"waveform": torch.ones(1, 1600), "audio_filepath": _wav(tmp_path / "s2.wav")}, "t")
    assert out is not None and out[0].ndim == 1  # noqa: PT018


def test_sigmos_residency_waveform_no_data_returns_none(tmp_path: Path):  # noqa: ANN202
    out = _get_audio_numpy_sr({"audio_filepath": _wav(tmp_path / "s3.wav")}, "t", input_residency="waveform")
    assert out is None


def test_sigmos_missing_returns_none():  # noqa: ANN202
    assert _get_audio_numpy_sr({}, "t") is None


# --------------------------------------------------------------------------- #
# resolve_audio (unchanged) — pin its current contract for reference
# --------------------------------------------------------------------------- #
def test_resolve_audio_waveform_branch_does_not_force_mono():  # noqa: ANN202
    # resolve_audio keeps channels on the in-memory branch (mono only applies on file load).
    out = resolve_audio({"waveform": torch.ones(2, 1600), "sample_rate": _SR})
    t, sr = out
    assert t.shape == (2, 1600) and sr == _SR  # noqa: PT018


def test_resolve_audio_file_branch_applies_mono(tmp_path: Path):  # noqa: ANN202
    out = resolve_audio({"audio_filepath": _wav(tmp_path / "r.wav", channels=2)}, mono=True)
    t, _ = out
    assert t.shape[0] == 1


# --------------------------------------------------------------------------- #
# common.resolve_waveform_from_item — unique sr-from-header behavior
# --------------------------------------------------------------------------- #
def test_common_reads_sr_from_header_without_reloading_waveform(tmp_path: Path):  # noqa: ANN202
    wav = _wav(tmp_path / "c.wav")
    provided = torch.ones(1, 1600)
    item = {"waveform": provided, "audio_filepath": wav}  # waveform present, sample_rate MISSING
    out = resolve_waveform_from_item(item, "t")
    t, sr = out
    assert sr == _SR  # read from the file header
    assert t.shape == (1, 1600)  # kept the provided waveform (not reloaded from file)
    assert item["sample_rate"] == _SR  # written back into the item


# --------------------------------------------------------------------------- #
# resolve_audio_path — tilde expansion + missing-file pass-through
# (pre-residency stages fed url_to_fs-normalized or raw paths to their own
# machinery; the residency layer must not be stricter than they were)
# --------------------------------------------------------------------------- #
def test_resolve_audio_path_existing_file_returned_verbatim(tmp_path: Path):  # noqa: ANN202
    wav = _wav(tmp_path / "p.wav")
    assert resolve_audio_path({"audio_filepath": wav}, residency="file") == wav


def test_resolve_audio_path_expands_tilde_for_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN202
    _wav(tmp_path / "home_audio.wav")
    monkeypatch.setenv("HOME", str(tmp_path))
    resolved = resolve_audio_path({"audio_filepath": "~/home_audio.wav"}, residency="file")
    assert resolved == str(tmp_path / "home_audio.wav")


def test_resolve_audio_expands_tilde_for_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN202
    _wav(tmp_path / "home_audio.wav")
    monkeypatch.setenv("HOME", str(tmp_path))
    out = resolve_audio({"audio_filepath": "~/home_audio.wav"})
    assert out is not None
    _, sr = out
    assert sr == _SR


def test_resolve_audio_path_missing_file_passes_through(tmp_path: Path):  # noqa: ANN202
    missing = str(tmp_path / "not_there.wav")
    assert resolve_audio_path({"audio_filepath": missing}, residency="file") == missing
    # auto residency with no waveform fallback also passes the path through
    assert resolve_audio_path({"audio_filepath": missing}, residency="auto") == missing


def test_resolve_audio_path_missing_file_prefers_waveform_fallback(tmp_path: Path):  # noqa: ANN202
    missing = str(tmp_path / "not_there.wav")
    item = {"audio_filepath": missing, "waveform": torch.zeros(1, 1600), "sample_rate": _SR}
    temp: list[str] = []
    resolved = resolve_audio_path(item, residency="auto", register_temp=temp)
    assert resolved != missing
    assert temp == [resolved]
    assert os.path.exists(resolved)
    cleanup_temp_files(temp)
    assert not os.path.exists(resolved)


def test_resolve_audio_path_no_input_returns_none():  # noqa: ANN202
    assert resolve_audio_path({}, residency="file") is None
    assert resolve_audio_path({}, residency="auto") is None


# --------------------------------------------------------------------------- write_audio_stable
# The four stages that write in-memory audio, all going through the shared helper.


def _writers(directory: str) -> dict[str, object]:
    """One thunk per stage that writes in-memory audio, all sharing ``write_audio_stable``."""
    from nemo_curator.stages.audio.preprocessing.channel_count import ChannelCountStage
    from nemo_curator.stages.audio.preprocessing.concatenation import SegmentConcatenationStage
    from nemo_curator.stages.audio.preprocessing.mono_conversion import MonoConversionStage
    from nemo_curator.stages.audio.segmentation.speaker_separation import SpeakerSeparationStage
    from nemo_curator.tasks import AudioTask

    wav = torch.sin(torch.arange(0, 16000) * 0.01).unsqueeze(0)
    task = AudioTask(task_id="t", dataset_name="d", data={"audio_filepath": "/data/spk1/utt1.wav"})
    return {
        "mono": lambda: MonoConversionStage(output_dir=directory)._write_audio(wav, 16000, task),
        "channel": lambda: ChannelCountStage(action="convert", output_dir=directory)._write_audio(wav, 16000, task),
        "concat": lambda: SegmentConcatenationStage(write_to_disk=True, output_dir=directory)._write_wav(
            wav, 16000, "/data/spk1/utt1.wav"
        ),
        "speaker": lambda: SpeakerSeparationStage(separated_audio_dir=directory)._write_speaker_wav(
            wav, 16000, "/data/spk1/utt1.wav", "spk0"
        ),
    }


def test_in_memory_writers_do_not_accumulate_a_file_per_run(tmp_path: Path):  # noqa: ANN202
    """The same audio written three times is one file, not three."""
    for name in _writers(str(tmp_path)):
        directory = tmp_path / f"out_{name}"
        directory.mkdir()
        write_once = _writers(str(directory))[name]
        for _ in range(3):
            written = write_once()
        assert len(os.listdir(directory)) == 1, f"{name} wrote a file per run: {os.listdir(directory)}"
        assert os.path.basename(written).startswith("utt1_"), "the source stem must stay readable in the name"


def test_write_audio_stable_separates_audio_that_differs(tmp_path: Path):  # noqa: ANN202
    """Different audio, rate or tag must never resolve to the same name."""
    from nemo_curator.stages.audio._agent._residency import write_audio_stable

    out = str(tmp_path)
    a = torch.sin(torch.arange(0, 16000) * 0.01).unsqueeze(0)
    b = torch.sin(torch.arange(0, 16000) * 0.02).unsqueeze(0)
    names = {
        os.path.basename(write_audio_stable(a, 16000, output_dir=out, stem="x")),
        os.path.basename(write_audio_stable(b, 16000, output_dir=out, stem="x")),  # other audio
        os.path.basename(write_audio_stable(a, 8000, output_dir=out, stem="x")),  # other rate
        os.path.basename(write_audio_stable(a, 16000, output_dir=out, stem="x", tag="mono")),
    }
    assert len(names) == 4, f"distinct inputs collapsed onto one name: {names}"


def test_write_audio_stable_without_an_output_dir_keeps_mkstemp_privacy(tmp_path: Path):  # noqa: ANN202, ARG001
    """The no-output_dir default is the shared system temp dir, where a predictable name leaks.

    mkstemp was giving three things away for free there: an unguessable name, owner-only mode on
    raw speech audio, and an exclusive create. Content-addressing that directory would have made
    two processes agree on one world-readable path in a 1777 directory.
    """
    import stat

    from nemo_curator.stages.audio._agent._residency import write_audio_stable

    written = write_audio_stable(torch.zeros(1, 16000), 16000, output_dir=None, stem="x")
    try:
        assert stat.S_IMODE(os.stat(written).st_mode) == 0o600
    finally:
        os.unlink(written)
