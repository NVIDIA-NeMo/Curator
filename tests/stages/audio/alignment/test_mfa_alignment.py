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

"""Tests for MFAAlignmentStage."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nemo_curator.stages.audio.alignment.mfa_alignment import MFAAlignmentStage
from nemo_curator.tasks import AudioTask

MODULE = "nemo_curator.stages.audio.alignment.mfa_alignment"


def _make_stage(tmp_path: Path, **overrides: object) -> MFAAlignmentStage:
    defaults: dict[str, object] = {
        "output_dir": str(tmp_path / "output"),
        "mfa_root_dir": str(tmp_path / "mfa_root"),
        "copy_models_to_local": False,
    }
    defaults.update(overrides)
    return MFAAlignmentStage(**defaults)  # type: ignore[arg-type]


def _make_wav(tmp_path: Path, name: str = "sample.wav") -> Path:
    wav = tmp_path / name
    wav.write_bytes(b"RIFF" + b"\x00" * 100)
    return wav


def _make_task(
    wav: Path,
    text: str = "hello world",
    *,
    text_key: str = "text",
    **extra: object,
) -> AudioTask:
    data: dict[str, object] = {
        "audio_filepath": str(wav),
        text_key: text,
        "speaker": "spk1",
        "duration": 1.0,
        **extra,
    }
    return AudioTask(data=data)


def _fake_textgrid_entry(start: float, end: float, label: str) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end, label=label)


def _fake_tier(entries: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(entries=entries)


def _fake_textgrid(
    entries: list[SimpleNamespace], tier_name: str = "words"
) -> SimpleNamespace:
    tier = _fake_tier(entries)
    return SimpleNamespace(
        tierNames=[tier_name],
        getTier=lambda _name: tier,  # noqa: ARG005
    )


def _fake_textgrid_multi(tier_entries: dict[str, list[SimpleNamespace]]) -> SimpleNamespace:
    tiers = {name: _fake_tier(entries) for name, entries in tier_entries.items()}
    return SimpleNamespace(
        tierNames=list(tier_entries.keys()),
        getTier=lambda name: tiers[name],
    )


def _align_textgrid_output_dir(cmd: list[str]) -> Path:
    align_idx = cmd.index("align")
    return Path(cmd[align_idx + 4])


def _setup_stage(
    stage: MFAAlignmentStage,
    *,
    textgrid: SimpleNamespace | None = None,
) -> MagicMock:
    """Run setup() with praatio mocked at the import boundary."""
    fake_tg_mod = MagicMock()
    if textgrid is not None:
        fake_tg_mod.openTextgrid.return_value = textgrid
    fake_praatio = MagicMock()
    fake_praatio.textgrid = fake_tg_mod
    with patch.dict(
        sys.modules,
        {"praatio": fake_praatio, "praatio.textgrid": fake_tg_mod},
    ):
        stage.setup()
    return fake_tg_mod


def _mock_mfa_writes_textgrid(wav: Path):
    def _run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:  # noqa: ARG001
        tg_dir = _align_textgrid_output_dir(cmd)
        (tg_dir / f"{wav.stem}.TextGrid").write_text("fake textgrid")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    return _run


class TestMFAAlignmentStage:
    """Test suite for MFAAlignmentStage."""

    def test_stage_properties(self, tmp_path: Path) -> None:
        stage = _make_stage(tmp_path)
        assert stage.name == "MFAAlignmentStage"

        _, data_in = stage.inputs()
        assert "audio_filepath" in data_in
        assert "text" in data_in

        _, data_out = stage.outputs()
        assert "textgrid_filepath" in data_out
        assert "rttm_filepath" in data_out
        assert "ctm_filepath" in data_out

        valid = AudioTask(data={"audio_filepath": "/a.wav", "text": "hello"})
        assert stage.validate_input(valid) is True
        assert stage.validate_input(AudioTask(data={"audio_filepath": "/a.wav"})) is False
        assert stage.validate_input(AudioTask(data={"text": "hello"})) is False

    def test_outputs_reflect_create_flags(self, tmp_path: Path) -> None:
        _, data_no_rttm = _make_stage(tmp_path, create_rttm=False).outputs()
        assert "rttm_filepath" not in data_no_rttm
        assert "ctm_filepath" in data_no_rttm

        _, data_no_ctm = _make_stage(tmp_path, create_ctm=False).outputs()
        assert "rttm_filepath" in data_no_ctm
        assert "ctm_filepath" not in data_no_ctm

        _, data_tg_only = _make_stage(
            tmp_path, create_rttm=False, create_ctm=False
        ).outputs()
        assert data_tg_only == ["textgrid_filepath"]

    def test_output_dir_required(self) -> None:
        with pytest.raises(TypeError, match="output_dir"):
            MFAAlignmentStage()  # type: ignore[call-arg]

    def test_process_raises_not_implemented(self, tmp_path: Path) -> None:
        stage = _make_stage(tmp_path)
        with pytest.raises(NotImplementedError, match="only supports process_batch"):
            stage.process(AudioTask(data={"audio_filepath": "/a.wav", "text": "hi"}))

    def test_setup_raises_without_praatio(self, tmp_path: Path) -> None:
        import builtins

        stage = _make_stage(tmp_path)
        real_import = builtins.__import__

        def _block_praatio(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "praatio":
                raise ImportError("No module named 'praatio'")
            return real_import(name, globals, locals, fromlist, level)

        with (
            patch("builtins.__import__", side_effect=_block_praatio),
            pytest.raises(ImportError, match="praatio is required"),
        ):
            stage.setup()

    def test_process_batch_empty(self, tmp_path: Path) -> None:
        stage = _make_stage(tmp_path)
        _setup_stage(stage)
        assert stage.process_batch([]) == []

    def test_process_batch_success(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        stage = _make_stage(tmp_path)
        entries = [
            _fake_textgrid_entry(0.0, 0.5, "hello"),
            _fake_textgrid_entry(0.5, 1.0, "world"),
        ]
        _setup_stage(stage, textgrid=_fake_textgrid(entries))
        task = _make_task(wav)

        with patch(f"{MODULE}.subprocess.run", side_effect=_mock_mfa_writes_textgrid(wav)):
            results = stage.process_batch([task])

        assert len(results) == 1
        assert "textgrid_filepath" in results[0].data
        assert "rttm_filepath" in results[0].data
        assert "ctm_filepath" in results[0].data
        assert Path(results[0].data["rttm_filepath"]).exists()
        assert Path(results[0].data["ctm_filepath"]).exists()

    def test_process_batch_mfa_failure_raises(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        stage = _make_stage(tmp_path)
        _setup_stage(stage)
        task = _make_task(wav)

        failed = subprocess.CompletedProcess(
            ["mfa"], returncode=1, stdout="error out", stderr="error err"
        )
        with (
            patch(f"{MODULE}.subprocess.run", return_value=failed),
            pytest.raises(RuntimeError, match="mfa align failed"),
        ):
            stage.process_batch([task])

    def test_process_batch_missing_textgrid_fallback(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        stage = _make_stage(tmp_path)
        _setup_stage(stage)
        task = _make_task(wav, duration=2.0)

        ok = subprocess.CompletedProcess(["mfa"], returncode=0, stdout="", stderr="")
        with patch(f"{MODULE}.subprocess.run", return_value=ok):
            results = stage.process_batch([task])

        assert len(results) == 1
        assert results[0].data.get("mfa_skipped") is True
        assert results[0].data["textgrid_filepath"] == ""
        assert Path(results[0].data["rttm_filepath"]).exists()
        assert Path(results[0].data["ctm_filepath"]).exists()
        ctm_lines = Path(results[0].data["ctm_filepath"]).read_text().strip().split("\n")
        assert len(ctm_lines) == 2
        assert "hello" in ctm_lines[0]
        assert "world" in ctm_lines[1]

    def test_process_batch_create_rttm_false(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        stage = _make_stage(tmp_path, create_rttm=False)
        _setup_stage(stage, textgrid=_fake_textgrid([_fake_textgrid_entry(0.0, 1.0, "hello")]))
        task = _make_task(wav, text="hello")

        with patch(f"{MODULE}.subprocess.run", side_effect=_mock_mfa_writes_textgrid(wav)):
            results = stage.process_batch([task])

        assert "rttm_filepath" not in results[0].data
        assert "ctm_filepath" in results[0].data
        assert "textgrid_filepath" in results[0].data

    def test_process_batch_create_ctm_false(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        stage = _make_stage(tmp_path, create_ctm=False)
        _setup_stage(stage, textgrid=_fake_textgrid([_fake_textgrid_entry(0.0, 1.0, "hello")]))
        task = _make_task(wav, text="hello")

        with patch(f"{MODULE}.subprocess.run", side_effect=_mock_mfa_writes_textgrid(wav)):
            results = stage.process_batch([task])

        assert "rttm_filepath" in results[0].data
        assert "ctm_filepath" not in results[0].data

    def test_process_batch_textgrid_only(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        stage = _make_stage(tmp_path, create_rttm=False, create_ctm=False)
        _setup_stage(stage)
        task = _make_task(wav, text="hello")

        with patch(f"{MODULE}.subprocess.run", side_effect=_mock_mfa_writes_textgrid(wav)):
            results = stage.process_batch([task])

        assert "textgrid_filepath" in results[0].data
        assert "rttm_filepath" not in results[0].data
        assert "ctm_filepath" not in results[0].data

    def test_process_batch_prefers_words_tier_over_phones(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        stage = _make_stage(tmp_path)
        textgrid = _fake_textgrid_multi({
            "phones": [_fake_textgrid_entry(0.0, 0.1, "AH")],
            "words": [_fake_textgrid_entry(0.1, 0.5, "hello")],
        })
        _setup_stage(stage, textgrid=textgrid)
        task = _make_task(wav, text="hello")

        with patch(f"{MODULE}.subprocess.run", side_effect=_mock_mfa_writes_textgrid(wav)):
            results = stage.process_batch([task])

        ctm_words = [
            line.split()[-1]
            for line in Path(results[0].data["ctm_filepath"]).read_text().strip().split("\n")
            if line
        ]
        assert ctm_words == ["hello"]

    def test_process_batch_raises_when_only_phone_tiers(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        stage = _make_stage(tmp_path)
        textgrid = _fake_textgrid_multi({
            "phones": [_fake_textgrid_entry(0.0, 0.1, "AH")],
        })
        _setup_stage(stage, textgrid=textgrid)
        task = _make_task(wav, text="hello")

        with (
            patch(f"{MODULE}.subprocess.run", side_effect=_mock_mfa_writes_textgrid(wav)),
            pytest.raises(ValueError, match="Refusing to parse phone-level tiers"),
        ):
            stage.process_batch([task])

    def test_process_batch_filters_silence_markers(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        stage = _make_stage(tmp_path)
        entries = [
            _fake_textgrid_entry(0.0, 0.1, "sp"),
            _fake_textgrid_entry(0.1, 0.3, "hello"),
            _fake_textgrid_entry(0.3, 0.4, "sil"),
            _fake_textgrid_entry(0.4, 0.6, "world"),
            _fake_textgrid_entry(0.6, 0.7, "<eps>"),
        ]
        _setup_stage(stage, textgrid=_fake_textgrid(entries))
        task = _make_task(wav)

        with patch(f"{MODULE}.subprocess.run", side_effect=_mock_mfa_writes_textgrid(wav)):
            results = stage.process_batch([task])

        ctm_words = [
            line.split()[-1]
            for line in Path(results[0].data["ctm_filepath"]).read_text().strip().split("\n")
            if line
        ]
        assert ctm_words == ["hello", "world"]

    def test_process_batch_custom_silence_markers(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        stage = _make_stage(tmp_path, silence_markers=("", "PAUSE"))
        entries = [
            _fake_textgrid_entry(0.0, 0.2, "PAUSE"),
            _fake_textgrid_entry(0.2, 0.5, "sp"),
            _fake_textgrid_entry(0.5, 0.8, "hello"),
        ]
        _setup_stage(stage, textgrid=_fake_textgrid(entries))
        task = _make_task(wav, text="hello")

        with patch(f"{MODULE}.subprocess.run", side_effect=_mock_mfa_writes_textgrid(wav)):
            results = stage.process_batch([task])

        ctm_words = [
            line.split()[-1]
            for line in Path(results[0].data["ctm_filepath"]).read_text().strip().split("\n")
            if line
        ]
        assert "PAUSE" not in ctm_words
        assert "sp" in ctm_words
        assert "hello" in ctm_words

    def test_custom_text_key(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        stage = _make_stage(tmp_path, text_key="utterance")
        _setup_stage(stage, textgrid=_fake_textgrid([_fake_textgrid_entry(0.0, 1.0, "hello")]))
        task = _make_task(wav, text="hello", text_key="utterance")

        _, data_keys = stage.inputs()
        assert "utterance" in data_keys

        with patch(f"{MODULE}.subprocess.run", side_effect=_mock_mfa_writes_textgrid(wav)):
            results = stage.process_batch([task])

        assert len(results) == 1
        assert "textgrid_filepath" in results[0].data

    def test_mfa_command_construction(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        stage = _make_stage(
            tmp_path,
            mfa_command="conda run -n mfa mfa",
            beam=200,
            retry_beam=800,
            single_speaker=False,
            clean=False,
            use_mp=False,
            output_format="short_textgrid",
        )
        _setup_stage(stage, textgrid=_fake_textgrid([_fake_textgrid_entry(0.0, 1.0, "test")]))
        captured_cmd: list[str] = []

        def capture_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:  # noqa: ARG001
            captured_cmd.extend(cmd)
            tg_dir = _align_textgrid_output_dir(cmd)
            (tg_dir / f"{wav.stem}.TextGrid").write_text("fake")
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        task = _make_task(wav, text="test")
        with patch(f"{MODULE}.subprocess.run", side_effect=capture_run):
            stage.process_batch([task])

        assert "align" in captured_cmd
        assert "--beam" in captured_cmd and "200" in captured_cmd
        assert "--retry_beam" in captured_cmd and "800" in captured_cmd
        assert "--output_format" in captured_cmd and "short_textgrid" in captured_cmd
        assert "--single_speaker" not in captured_cmd
        assert "--clean" not in captured_cmd
        assert "--use_mp" not in captured_cmd

    def test_setup_on_node_copies_models(self, tmp_path: Path) -> None:
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

    def test_setup_on_node_reuses_existing_local_root(self, tmp_path: Path) -> None:
        import socket

        local_base = tmp_path / "local"
        local_mfa = local_base / f"mfa_models_{socket.gethostname()}"
        (local_mfa / "pretrained_models").mkdir(parents=True)

        stage = _make_stage(
            tmp_path,
            local_mfa_base_dir=str(local_base),
            copy_models_to_local=True,
        )
        stage.setup_on_node()
        assert stage._mfa_root == str(local_mfa)

    def test_shared_mfa_root_does_not_delete_command_history(self, tmp_path: Path) -> None:
        shared_root = tmp_path / "shared_mfa"
        shared_root.mkdir()
        history = shared_root / "command_history.yaml"
        history.write_text("history: []\n")

        stage = _make_stage(
            tmp_path,
            mfa_root_dir=str(shared_root),
            copy_models_to_local=False,
        )
        stage._mfa_root = str(shared_root)
        assert stage._is_node_local_mfa_root() is False

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        tg_out = tmp_path / "tg_out"
        tg_out.mkdir()

        with patch(
            f"{MODULE}.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, "", ""),
        ):
            stage._run_mfa_align(corpus, tg_out)

        assert history.exists()
        assert history.read_text() == "history: []\n"
