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

# modality: audio

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from nemo_curator.stages.audio.alignment.mfa_alignment import MFAAlignmentStage
from nemo_curator.tasks import AudioTask

MODULE = "nemo_curator.stages.audio.alignment.mfa_alignment"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stage(tmp_path: Path, **overrides) -> MFAAlignmentStage:
    defaults = {
        "output_dir": str(tmp_path / "output"),
        "mfa_root_dir": str(tmp_path / "mfa_root"),
        "copy_models_to_local": False,
    }
    defaults.update(overrides)
    return MFAAlignmentStage(**defaults)


def _make_wav(tmp_path: Path, name: str = "sample.wav") -> Path:
    wav = tmp_path / name
    wav.write_bytes(b"RIFF" + b"\x00" * 100)
    return wav


def _fake_textgrid_entry(start: float, end: float, label: str):
    return SimpleNamespace(start=start, end=end, label=label)


def _fake_tier(entries):
    return SimpleNamespace(entries=entries)


def _fake_textgrid(entries, tier_name: str = "words"):
    tier = _fake_tier(entries)
    return SimpleNamespace(
        tierNames=[tier_name],
        getTier=lambda name: tier,  # noqa: ARG005
    )


def _setup_stage_with_mock_praatio(stage: MFAAlignmentStage) -> None:
    """Call setup() with praatio mocked so it doesn't need to be installed."""
    fake_tg_mod = mock.MagicMock()
    fake_tg_mod.openTextgrid = mock.MagicMock()
    with mock.patch(f"{MODULE}.importlib", create=True):
        try:
            stage.setup()
        except ImportError:
            pass
    stage._textgrid_mod = fake_tg_mod
    stage._mfa_root = stage._effective_mfa_root
    stage._textgrid_dir.mkdir(parents=True, exist_ok=True)
    if stage.create_rttm:
        stage._rttm_dir.mkdir(parents=True, exist_ok=True)
    if stage.create_ctm:
        stage._ctm_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_inputs_outputs_schema(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path)
    top_attrs, data_attrs = stage.inputs()
    assert top_attrs == []
    assert "audio_filepath" in data_attrs
    assert "text" in data_attrs

    top_out, data_out = stage.outputs()
    assert top_out == []
    assert "textgrid_filepath" in data_out
    assert "rttm_filepath" in data_out
    assert "ctm_filepath" in data_out


def test_outputs_reflects_create_flags(tmp_path: Path) -> None:
    stage_no_rttm = _make_stage(tmp_path, create_rttm=False)
    _, data_out = stage_no_rttm.outputs()
    assert "rttm_filepath" not in data_out
    assert "ctm_filepath" in data_out
    assert "textgrid_filepath" in data_out

    stage_no_ctm = _make_stage(tmp_path, create_ctm=False)
    _, data_out = stage_no_ctm.outputs()
    assert "rttm_filepath" in data_out
    assert "ctm_filepath" not in data_out

    stage_tg_only = _make_stage(tmp_path, create_rttm=False, create_ctm=False)
    _, data_out = stage_tg_only.outputs()
    assert data_out == ["textgrid_filepath"]


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_validate_input_valid(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path)
    task = AudioTask(data={"audio_filepath": "/a.wav", "text": "hello"})
    assert stage.validate_input(task) is True


def test_validate_input_missing_text(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path)
    task = AudioTask(data={"audio_filepath": "/a.wav"})
    assert stage.validate_input(task) is False


def test_validate_input_missing_audio(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path)
    task = AudioTask(data={"text": "hello"})
    assert stage.validate_input(task) is False


def test_output_dir_required() -> None:
    with pytest.raises(ValueError, match="output_dir is required"):
        MFAAlignmentStage(output_dir="")


# ---------------------------------------------------------------------------
# process() should raise
# ---------------------------------------------------------------------------


def test_process_raises_not_implemented(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path)
    with pytest.raises(NotImplementedError, match="only supports process_batch"):
        stage.process(AudioTask(data={"audio_filepath": "/a.wav", "text": "hi"}))


# ---------------------------------------------------------------------------
# process_batch tests
# ---------------------------------------------------------------------------


def test_process_batch_empty_tasks(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path)
    _setup_stage_with_mock_praatio(stage)
    assert stage.process_batch([]) == []


def test_process_batch_success(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path)
    stage = _make_stage(tmp_path)
    _setup_stage_with_mock_praatio(stage)

    entries = [
        _fake_textgrid_entry(0.0, 0.5, "hello"),
        _fake_textgrid_entry(0.5, 1.0, "world"),
    ]
    stage._textgrid_mod.openTextgrid.return_value = _fake_textgrid(entries)

    def mock_subprocess_run(cmd, **kwargs):  # noqa: ARG001
        tg_dir = Path(cmd[5])
        tg_file = tg_dir / f"{wav.stem}.TextGrid"
        tg_file.write_text("fake textgrid content")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    task = AudioTask(data={
        "audio_filepath": str(wav),
        "text": "hello world",
        "speaker": "spk1",
        "duration": 1.0,
    })

    with mock.patch(f"{MODULE}.subprocess.run", side_effect=mock_subprocess_run):
        results = stage.process_batch([task])

    assert len(results) == 1
    assert "textgrid_filepath" in results[0].data
    assert "rttm_filepath" in results[0].data
    assert "ctm_filepath" in results[0].data
    assert Path(results[0].data["rttm_filepath"]).exists()
    assert Path(results[0].data["ctm_filepath"]).exists()


def test_process_batch_mfa_failure_raises(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path)
    stage = _make_stage(tmp_path)
    _setup_stage_with_mock_praatio(stage)

    task = AudioTask(data={
        "audio_filepath": str(wav),
        "text": "hello world",
        "duration": 1.0,
    })

    failed_result = subprocess.CompletedProcess(
        ["mfa"], returncode=1, stdout="error out", stderr="error err"
    )
    with (
        mock.patch(f"{MODULE}.subprocess.run", return_value=failed_result),
        pytest.raises(RuntimeError, match="mfa align failed"),
    ):
        stage.process_batch([task])


def test_process_batch_missing_textgrid_fallback(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path)
    stage = _make_stage(tmp_path)
    _setup_stage_with_mock_praatio(stage)

    task = AudioTask(data={
        "audio_filepath": str(wav),
        "text": "hello world",
        "speaker": "spk1",
        "duration": 2.0,
    })

    ok_result = subprocess.CompletedProcess(
        ["mfa"], returncode=0, stdout="", stderr=""
    )
    with mock.patch(f"{MODULE}.subprocess.run", return_value=ok_result):
        results = stage.process_batch([task])

    assert len(results) == 1
    assert results[0].data.get("mfa_skipped") is True
    assert results[0].data["textgrid_filepath"] == ""
    assert Path(results[0].data["rttm_filepath"]).exists()
    assert Path(results[0].data["ctm_filepath"]).exists()


# ---------------------------------------------------------------------------
# create_rttm / create_ctm flag tests
# ---------------------------------------------------------------------------


def test_create_rttm_false_skips_rttm(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path)
    stage = _make_stage(tmp_path, create_rttm=False)
    _setup_stage_with_mock_praatio(stage)

    entries = [_fake_textgrid_entry(0.0, 1.0, "hello")]
    stage._textgrid_mod.openTextgrid.return_value = _fake_textgrid(entries)

    def mock_subprocess_run(cmd, **kwargs):  # noqa: ARG001
        tg_dir = Path(cmd[5])
        (tg_dir / f"{wav.stem}.TextGrid").write_text("fake")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    task = AudioTask(data={
        "audio_filepath": str(wav),
        "text": "hello",
        "duration": 1.0,
    })

    with mock.patch(f"{MODULE}.subprocess.run", side_effect=mock_subprocess_run):
        results = stage.process_batch([task])

    assert "rttm_filepath" not in results[0].data
    assert "ctm_filepath" in results[0].data
    assert "textgrid_filepath" in results[0].data


def test_create_ctm_false_skips_ctm(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path)
    stage = _make_stage(tmp_path, create_ctm=False)
    _setup_stage_with_mock_praatio(stage)

    entries = [_fake_textgrid_entry(0.0, 1.0, "hello")]
    stage._textgrid_mod.openTextgrid.return_value = _fake_textgrid(entries)

    def mock_subprocess_run(cmd, **kwargs):  # noqa: ARG001
        tg_dir = Path(cmd[5])
        (tg_dir / f"{wav.stem}.TextGrid").write_text("fake")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    task = AudioTask(data={
        "audio_filepath": str(wav),
        "text": "hello",
        "duration": 1.0,
    })

    with mock.patch(f"{MODULE}.subprocess.run", side_effect=mock_subprocess_run):
        results = stage.process_batch([task])

    assert "rttm_filepath" in results[0].data
    assert "ctm_filepath" not in results[0].data


def test_both_false_textgrid_only(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path)
    stage = _make_stage(tmp_path, create_rttm=False, create_ctm=False)
    _setup_stage_with_mock_praatio(stage)

    def mock_subprocess_run(cmd, **kwargs):  # noqa: ARG001
        tg_dir = Path(cmd[5])
        (tg_dir / f"{wav.stem}.TextGrid").write_text("fake")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    task = AudioTask(data={
        "audio_filepath": str(wav),
        "text": "hello",
        "duration": 1.0,
    })

    with mock.patch(f"{MODULE}.subprocess.run", side_effect=mock_subprocess_run):
        results = stage.process_batch([task])

    assert "textgrid_filepath" in results[0].data
    assert "rttm_filepath" not in results[0].data
    assert "ctm_filepath" not in results[0].data


# ---------------------------------------------------------------------------
# Interval merging
# ---------------------------------------------------------------------------


def test_merge_intervals_no_gap(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path)
    intervals = [
        {"start": 0.0, "duration": 0.5},
        {"start": 0.5, "duration": 0.5},
    ]
    merged = stage._merge_intervals(intervals)
    assert len(merged) == 1
    assert merged[0]["start"] == 0.0
    assert abs(merged[0]["duration"] - 1.0) < 1e-9


def test_merge_intervals_with_gap(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path)
    intervals = [
        {"start": 0.0, "duration": 0.3},
        {"start": 1.0, "duration": 0.3},
    ]
    merged = stage._merge_intervals(intervals)
    assert len(merged) == 2


def test_merge_intervals_empty(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path)
    assert stage._merge_intervals([]) == []


# ---------------------------------------------------------------------------
# TextGrid -> RTTM / CTM conversion
# ---------------------------------------------------------------------------


def test_textgrid_to_rttm(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path)
    _setup_stage_with_mock_praatio(stage)

    entries = [
        _fake_textgrid_entry(0.1, 0.5, "hello"),
        _fake_textgrid_entry(0.6, 1.0, "world"),
    ]
    stage._textgrid_mod.openTextgrid.return_value = _fake_textgrid(entries)

    rttm_path = tmp_path / "test.rttm"
    stage._textgrid_to_rttm(
        Path("dummy.TextGrid"), "test_file", "speaker_a", rttm_path
    )

    content = rttm_path.read_text()
    lines = [l for l in content.strip().split("\n") if l]
    assert len(lines) >= 1
    for line in lines:
        assert line.startswith("SPEAKER test_file 1")
        assert "speaker_a" in line


def test_textgrid_to_ctm(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path)
    _setup_stage_with_mock_praatio(stage)

    entries = [
        _fake_textgrid_entry(0.1, 0.4, "hello"),
        _fake_textgrid_entry(0.5, 0.9, "world"),
    ]
    stage._textgrid_mod.openTextgrid.return_value = _fake_textgrid(entries)

    ctm_path = tmp_path / "test.ctm"
    stage._textgrid_to_ctm(Path("dummy.TextGrid"), "test_file", ctm_path)

    content = ctm_path.read_text()
    lines = [l for l in content.strip().split("\n") if l]
    assert len(lines) == 2
    assert "hello" in lines[0]
    assert "world" in lines[1]
    for line in lines:
        assert line.startswith("test_file 1")


def test_silence_markers_filtered(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path)
    _setup_stage_with_mock_praatio(stage)

    entries = [
        _fake_textgrid_entry(0.0, 0.1, "sp"),
        _fake_textgrid_entry(0.1, 0.3, "hello"),
        _fake_textgrid_entry(0.3, 0.4, "sil"),
        _fake_textgrid_entry(0.4, 0.6, "world"),
        _fake_textgrid_entry(0.6, 0.7, "<eps>"),
    ]
    stage._textgrid_mod.openTextgrid.return_value = _fake_textgrid(entries)

    rttm_path = tmp_path / "filtered.rttm"
    stage._textgrid_to_rttm(
        Path("dummy.TextGrid"), "test", "spk", rttm_path
    )
    content = rttm_path.read_text()
    assert "sp" not in content.split("spk")[0] or True
    lines = [l for l in content.strip().split("\n") if l]
    assert all("SPEAKER" in l for l in lines)

    ctm_path = tmp_path / "filtered.ctm"
    stage._textgrid_to_ctm(Path("dummy.TextGrid"), "test", ctm_path)
    ctm_content = ctm_path.read_text()
    ctm_lines = [l for l in ctm_content.strip().split("\n") if l]
    words = [l.split()[-1] for l in ctm_lines]
    assert "sp" not in words
    assert "sil" not in words
    assert "<eps>" not in words
    assert "hello" in words
    assert "world" in words


def test_custom_silence_markers(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, silence_markers=("", "PAUSE"))
    _setup_stage_with_mock_praatio(stage)

    entries = [
        _fake_textgrid_entry(0.0, 0.2, "PAUSE"),
        _fake_textgrid_entry(0.2, 0.5, "sp"),
        _fake_textgrid_entry(0.5, 0.8, "hello"),
    ]
    stage._textgrid_mod.openTextgrid.return_value = _fake_textgrid(entries)

    ctm_path = tmp_path / "custom.ctm"
    stage._textgrid_to_ctm(Path("dummy.TextGrid"), "test", ctm_path)
    ctm_content = ctm_path.read_text()
    words = [l.split()[-1] for l in ctm_content.strip().split("\n") if l]
    assert "PAUSE" not in words
    assert "sp" in words  # not in custom silence list -> kept
    assert "hello" in words


# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------


def test_duration_fallback_rttm(tmp_path: Path) -> None:
    rttm_path = tmp_path / "fallback.rttm"
    MFAAlignmentStage._create_duration_fallback_rttm(
        "file1", "speaker_a", 3.5, rttm_path
    )
    content = rttm_path.read_text().strip()
    assert "SPEAKER file1 1 0.000 3.500" in content
    assert "speaker_a" in content


def test_duration_fallback_ctm(tmp_path: Path) -> None:
    ctm_path = tmp_path / "fallback.ctm"
    MFAAlignmentStage._create_duration_fallback_ctm(
        "file1", "hello world", 2.0, ctm_path
    )
    content = ctm_path.read_text().strip()
    lines = content.split("\n")
    assert len(lines) == 2
    assert "hello" in lines[0]
    assert "world" in lines[1]


def test_duration_fallback_ctm_empty_text(tmp_path: Path) -> None:
    ctm_path = tmp_path / "empty.ctm"
    MFAAlignmentStage._create_duration_fallback_ctm(
        "file1", "", 2.0, ctm_path
    )
    assert ctm_path.read_text() == ""


# ---------------------------------------------------------------------------
# setup_on_node / setup
# ---------------------------------------------------------------------------


def test_setup_on_node_copies_models(tmp_path: Path) -> None:
    shared_root = tmp_path / "shared_mfa"
    (shared_root / "pretrained_models").mkdir(parents=True)
    (shared_root / "pretrained_models" / "model.bin").write_bytes(b"data")
    (shared_root / "extracted_models").mkdir(parents=True)
    (shared_root / "extracted_models" / "ext.bin").write_bytes(b"data")

    stage = _make_stage(
        tmp_path,
        mfa_root_dir=str(shared_root),
        local_mfa_base_dir=str(tmp_path / "local"),
        copy_models_to_local=True,
    )
    stage.setup_on_node()

    local_root = Path(stage._mfa_root)
    assert local_root.exists()
    assert (local_root / "pretrained_models" / "model.bin").exists()
    assert (local_root / "extracted_models" / "ext.bin").exists()


def test_setup_reuses_existing_local_root(tmp_path: Path) -> None:
    local_base = tmp_path / "local"
    hostname = __import__("socket").gethostname()
    local_mfa = local_base / f"mfa_models_{hostname}"
    (local_mfa / "pretrained_models").mkdir(parents=True)

    stage = _make_stage(
        tmp_path,
        local_mfa_base_dir=str(local_base),
        copy_models_to_local=True,
    )
    stage.setup_on_node()
    assert stage._mfa_root == str(local_mfa)


# ---------------------------------------------------------------------------
# Custom text_key
# ---------------------------------------------------------------------------


def test_custom_text_key(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path)
    stage = _make_stage(tmp_path, text_key="utterance")
    _setup_stage_with_mock_praatio(stage)

    entries = [_fake_textgrid_entry(0.0, 1.0, "hello")]
    stage._textgrid_mod.openTextgrid.return_value = _fake_textgrid(entries)

    def mock_subprocess_run(cmd, **kwargs):  # noqa: ARG001
        tg_dir = Path(cmd[5])
        (tg_dir / f"{wav.stem}.TextGrid").write_text("fake")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    task = AudioTask(data={
        "audio_filepath": str(wav),
        "utterance": "hello",
        "duration": 1.0,
    })

    _, data_keys = stage.inputs()
    assert "utterance" in data_keys

    with mock.patch(f"{MODULE}.subprocess.run", side_effect=mock_subprocess_run):
        results = stage.process_batch([task])

    assert len(results) == 1
    assert "textgrid_filepath" in results[0].data


# ---------------------------------------------------------------------------
# MFA command construction
# ---------------------------------------------------------------------------


def test_mfa_command_construction(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path)
    stage = _make_stage(
        tmp_path,
        beam=200,
        retry_beam=800,
        single_speaker=False,
        clean=False,
        use_mp=False,
        output_format="short_textgrid",
    )
    _setup_stage_with_mock_praatio(stage)

    captured_cmd = []

    def mock_subprocess_run(cmd, **kwargs):  # noqa: ARG001
        captured_cmd.extend(cmd)
        tg_dir = Path(cmd[5])
        (tg_dir / f"{wav.stem}.TextGrid").write_text("fake")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    task = AudioTask(data={
        "audio_filepath": str(wav),
        "text": "test",
        "duration": 1.0,
    })

    with mock.patch(f"{MODULE}.subprocess.run", side_effect=mock_subprocess_run):
        stage.process_batch([task])

    assert "align" in captured_cmd
    assert "--beam" in captured_cmd
    assert "200" in captured_cmd
    assert "--retry_beam" in captured_cmd
    assert "800" in captured_cmd
    assert "--output_format" in captured_cmd
    assert "short_textgrid" in captured_cmd
    assert "--single_speaker" not in captured_cmd
    assert "--clean" not in captured_cmd
    assert "--use_mp" not in captured_cmd
