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

import json
import os
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from nemo_curator.stages.audio.alignment.mfa_alignment import (
    _MFA_CACHE_MARKER_NAME,
    MFAAlignmentStage,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from collections.abc import Callable

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
        getTier=lambda _name: tier,
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
    """Run setup() and inject a mock TextGrid parser."""
    stage.setup()
    fake_tg_mod = MagicMock()
    if textgrid is not None:
        fake_tg_mod.openTextgrid.return_value = textgrid
    stage._textgrid_mod = fake_tg_mod
    return fake_tg_mod


def _mock_mfa_writes_textgrid(wav: Path) -> Callable[..., subprocess.CompletedProcess]:
    def _run(cmd: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
        tg_dir = _align_textgrid_output_dir(cmd)
        (tg_dir / f"{wav.stem}.TextGrid").write_text("fake textgrid")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    return _run


def _patch_mfa_subprocess(stage: MFAAlignmentStage, **kwargs: object):  # noqa: ANN202
    """Patch the stage's MFA subprocess seam (replaces ``subprocess.run`` mocking).

    ``_run_mfa_subprocess`` -- not the module-level ``subprocess.run`` -- is
    the stage's process-invocation boundary, since MFA is now run via
    ``Popen`` (with its own process group) to support timeout + killpg.
    """
    return patch.object(stage, "_run_mfa_subprocess", **kwargs)


class TestMFAAlignmentStage:
    """Test suite for MFAAlignmentStage."""

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
        assert data_tg_only == ["textgrid_filepath", "mfa_skipped", "duration"]

    def test_outputs_declares_custom_duration_key(self, tmp_path: Path) -> None:
        """The configurable duration output must be declared under its custom name too."""
        stage = _make_stage(tmp_path, duration_key="seconds")
        _, data_keys = stage.outputs()
        assert "seconds" in data_keys
        assert "duration" not in data_keys

    def test_process_batch_empty(self, tmp_path: Path) -> None:
        stage = _make_stage(tmp_path)
        _setup_stage(stage)
        assert stage.process_batch([]) == []

    def test_process_batch_accepts_numpy_array_of_tasks(self, tmp_path: Path) -> None:
        """Ray Data's actual batch format is a numpy object array, not a list.

        ``if not tasks:`` raises ``ValueError`` ("truth value of an array
        with more than one element is ambiguous") for a multi-element numpy
        array, so ``process_batch`` must use ``len(tasks) == 0`` instead.
        This mirrors ``batch["item"]`` in
        ``RayDataStageAdapter._process_batch_internal``.
        """
        np = pytest.importorskip("numpy")
        wav = _make_wav(tmp_path)
        stage = _make_stage(tmp_path)
        entries = [
            _fake_textgrid_entry(0.0, 0.5, "hello"),
            _fake_textgrid_entry(0.5, 1.0, "world"),
        ]
        _setup_stage(stage, textgrid=_fake_textgrid(entries))
        tasks = np.array([_make_task(wav), _make_task(wav)], dtype=object)

        with _patch_mfa_subprocess(stage, side_effect=_mock_mfa_writes_textgrid(wav)):
            results = stage.process_batch(tasks)

        assert isinstance(results, list)
        assert len(results) == len(tasks)

    def test_process_batch_success(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        stage = _make_stage(tmp_path)
        entries = [
            _fake_textgrid_entry(0.0, 0.5, "hello"),
            _fake_textgrid_entry(0.5, 1.0, "world"),
        ]
        _setup_stage(stage, textgrid=_fake_textgrid(entries))
        task = _make_task(wav)

        with _patch_mfa_subprocess(stage, side_effect=_mock_mfa_writes_textgrid(wav)):
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
            _patch_mfa_subprocess(stage, return_value=failed),
            pytest.raises(RuntimeError, match="mfa align failed"),
        ):
            stage.process_batch([task])

    def test_process_batch_missing_textgrid_fallback(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        stage = _make_stage(tmp_path)
        _setup_stage(stage)
        task = _make_task(wav, duration=2.0)

        ok = subprocess.CompletedProcess(["mfa"], returncode=0, stdout="", stderr="")
        with _patch_mfa_subprocess(stage, return_value=ok):
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

    def test_process_batch_skips_bad_task_and_continues(self, tmp_path: Path) -> None:
        """A single bad row is marked skipped without aborting the batch."""
        good_wav = _make_wav(tmp_path, name="good.wav")
        stage = _make_stage(tmp_path)
        _setup_stage(
            stage, textgrid=_fake_textgrid([_fake_textgrid_entry(0.0, 1.0, "hello")])
        )

        bad = _make_task(tmp_path / "does_not_exist.wav", text="hello")
        good = _make_task(good_wav, text="hello")

        with _patch_mfa_subprocess(stage, side_effect=_mock_mfa_writes_textgrid(good_wav)):
            results = stage.process_batch([bad, good])

        assert len(results) == 2
        assert results[0] is bad
        assert results[1] is good
        assert results[0].data["mfa_skipped"] is True
        assert results[0].data["textgrid_filepath"] == ""
        assert results[1].data["mfa_skipped"] is False
        assert results[1].data["textgrid_filepath"] != ""

    def test_process_batch_all_invalid_skips_without_running_mfa(
        self, tmp_path: Path
    ) -> None:
        """When every task fails pre-flight, MFA is never invoked."""
        stage = _make_stage(tmp_path)
        _setup_stage(stage)

        missing_file = _make_task(tmp_path / "missing.wav", text="hello")
        empty_text = _make_task(_make_wav(tmp_path, name="ok.wav"), text="   ")

        with _patch_mfa_subprocess(stage) as run_mock:
            results = stage.process_batch([missing_file, empty_text])

        run_mock.assert_not_called()
        assert len(results) == 2
        for task in results:
            assert task.data["mfa_skipped"] is True
            assert task.data["textgrid_filepath"] == ""
            assert task.data["rttm_filepath"] == ""
            assert task.data["ctm_filepath"] == ""

    def test_process_batch_create_rttm_false(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        stage = _make_stage(tmp_path, create_rttm=False)
        _setup_stage(stage, textgrid=_fake_textgrid([_fake_textgrid_entry(0.0, 1.0, "hello")]))
        task = _make_task(wav, text="hello")

        with _patch_mfa_subprocess(stage, side_effect=_mock_mfa_writes_textgrid(wav)):
            results = stage.process_batch([task])

        assert "rttm_filepath" not in results[0].data
        assert "ctm_filepath" in results[0].data
        assert "textgrid_filepath" in results[0].data

    def test_process_batch_create_ctm_false(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        stage = _make_stage(tmp_path, create_ctm=False)
        _setup_stage(stage, textgrid=_fake_textgrid([_fake_textgrid_entry(0.0, 1.0, "hello")]))
        task = _make_task(wav, text="hello")

        with _patch_mfa_subprocess(stage, side_effect=_mock_mfa_writes_textgrid(wav)):
            results = stage.process_batch([task])

        assert "rttm_filepath" in results[0].data
        assert "ctm_filepath" not in results[0].data

    def test_process_batch_textgrid_only(self, tmp_path: Path) -> None:
        wav = _make_wav(tmp_path)
        stage = _make_stage(tmp_path, create_rttm=False, create_ctm=False)
        _setup_stage(stage)
        task = _make_task(wav, text="hello")

        with _patch_mfa_subprocess(stage, side_effect=_mock_mfa_writes_textgrid(wav)):
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

        with _patch_mfa_subprocess(stage, side_effect=_mock_mfa_writes_textgrid(wav)):
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
            _patch_mfa_subprocess(stage, side_effect=_mock_mfa_writes_textgrid(wav)),
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

        with _patch_mfa_subprocess(stage, side_effect=_mock_mfa_writes_textgrid(wav)):
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

        with _patch_mfa_subprocess(stage, side_effect=_mock_mfa_writes_textgrid(wav)):
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

        with _patch_mfa_subprocess(stage, side_effect=_mock_mfa_writes_textgrid(wav)):
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

        def capture_run(cmd: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
            captured_cmd.extend(cmd)
            tg_dir = _align_textgrid_output_dir(cmd)
            (tg_dir / f"{wav.stem}.TextGrid").write_text("fake")
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        task = _make_task(wav, text="test")
        with _patch_mfa_subprocess(stage, side_effect=capture_run):
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
        local_base = tmp_path / "local"
        stage = _make_stage(
            tmp_path,
            local_mfa_base_dir=str(local_base),
            copy_models_to_local=True,
        )
        cache_dir = stage._local_cache_dir()
        (cache_dir / "pretrained_models").mkdir(parents=True)
        marker = {"identity": stage._cache_identity(), "complete": True}
        (cache_dir / _MFA_CACHE_MARKER_NAME).write_text(json.dumps(marker), encoding="utf-8")

        with patch(f"{MODULE}.shutil.copytree") as copytree_mock:
            stage.setup_on_node()

        copytree_mock.assert_not_called()
        assert stage._mfa_root == str(cache_dir)

    def test_setup_on_node_rebuilds_interrupted_copy(self, tmp_path: Path) -> None:
        """A cache dir left behind by an interrupted copy (no marker) must be rebuilt, not reused."""
        shared_root = tmp_path / "shared_mfa"
        (shared_root / "pretrained_models").mkdir(parents=True)
        (shared_root / "pretrained_models" / "model.bin").write_bytes(b"real-data")

        stage = _make_stage(
            tmp_path,
            mfa_root_dir=str(shared_root),
            local_mfa_base_dir=str(tmp_path / "local"),
            copy_models_to_local=True,
        )
        cache_dir = stage._local_cache_dir()
        # Simulate a crash mid-copy: the directory exists (from a previous run)
        # but is empty -- no completeness marker was ever written.
        cache_dir.mkdir(parents=True)

        stage.setup_on_node()

        local_root = Path(stage._mfa_root)
        assert local_root == cache_dir
        assert (local_root / "pretrained_models" / "model.bin").read_bytes() == b"real-data"
        marker = json.loads((local_root / _MFA_CACHE_MARKER_NAME).read_text())
        assert marker["complete"] is True

    def test_setup_on_node_sequential_different_roots_do_not_collide(self, tmp_path: Path) -> None:
        """Two sources on the same host/local base must never share a cache namespace."""
        local_base = tmp_path / "local"

        source_a = tmp_path / "mfa_root_a"
        (source_a / "pretrained_models").mkdir(parents=True)
        (source_a / "pretrained_models" / "a.model").write_bytes(b"model-a")

        source_b = tmp_path / "mfa_root_b"
        (source_b / "pretrained_models").mkdir(parents=True)
        (source_b / "pretrained_models" / "b.model").write_bytes(b"model-b")

        stage_a = _make_stage(
            tmp_path,
            mfa_root_dir=str(source_a),
            local_mfa_base_dir=str(local_base),
            copy_models_to_local=True,
        )
        stage_a.setup_on_node()
        cache_a = Path(stage_a._mfa_root)
        assert (cache_a / "pretrained_models" / "a.model").exists()

        stage_b = _make_stage(
            tmp_path,
            mfa_root_dir=str(source_b),
            local_mfa_base_dir=str(local_base),
            copy_models_to_local=True,
        )
        stage_b.setup_on_node()
        cache_b = Path(stage_b._mfa_root)

        # Distinct caches: source B never lands in / corrupts source A's cache.
        assert cache_a != cache_b
        assert (cache_b / "pretrained_models" / "b.model").exists()
        assert not (cache_b / "pretrained_models" / "a.model").exists()
        assert (cache_a / "pretrained_models" / "a.model").read_bytes() == b"model-a"
        assert not (cache_a / "pretrained_models" / "b.model").exists()

    def test_cache_digest_differs_by_model_identity_not_just_source(self, tmp_path: Path) -> None:
        """Same source, different requested model -- must not share a cache namespace."""
        stage_arpa = _make_stage(tmp_path, acoustic_model="english_us_arpa", dictionary="english_us_arpa")
        stage_mfa = _make_stage(tmp_path, acoustic_model="english_mfa", dictionary="english_mfa")
        assert stage_arpa._local_cache_dir() != stage_mfa._local_cache_dir()

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

        with _patch_mfa_subprocess(
            stage,
            return_value=subprocess.CompletedProcess([], 0, "", ""),
        ):
            stage._run_mfa_align(corpus, tg_out)

        assert history.exists()
        assert history.read_text() == "history: []\n"

    def test_num_jobs_must_be_positive(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="num_jobs must be positive"):
            _make_stage(tmp_path, num_jobs=0)

    def test_align_timeout_seconds_must_be_positive(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="align_timeout_seconds must be positive"):
            _make_stage(tmp_path, align_timeout_seconds=0)

    def test_resources_default_to_num_jobs_cpus(self, tmp_path: Path) -> None:
        """Ray/Xenna must reserve CPUs proportional to the -j jobs MFA forks."""
        stage = _make_stage(tmp_path, num_jobs=16)
        assert stage.resources.cpus == 16.0

        default_stage = _make_stage(tmp_path)
        assert default_stage.resources.cpus == 1.0

    def test_resources_explicit_override_is_preserved(self, tmp_path: Path) -> None:
        stage = _make_stage(tmp_path, num_jobs=16, resources=Resources(cpus=2.0))
        assert stage.resources.cpus == 2.0

    def test_run_mfa_subprocess_success(self, tmp_path: Path) -> None:
        stage = _make_stage(tmp_path)
        result = stage._run_mfa_subprocess(["echo", "hello"], {**os.environ})
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_run_mfa_subprocess_timeout_kills_process_group(self, tmp_path: Path) -> None:
        """On timeout, the whole process group is killed and a bounded TimeoutError raised."""
        stage = _make_stage(tmp_path, align_timeout_seconds=5)

        fake_process = MagicMock()
        fake_process.pid = 4321
        fake_process.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd=["mfa"], timeout=5),
            ("stuck stdout", "stuck stderr"),
        ]

        with (
            patch(f"{MODULE}.subprocess.Popen", return_value=fake_process) as popen_mock,
            patch(f"{MODULE}.os.getpgid", return_value=4321) as getpgid_mock,
            patch(f"{MODULE}.os.killpg") as killpg_mock,
            pytest.raises(TimeoutError, match="timed out after 5s"),
        ):
            stage._run_mfa_subprocess(["mfa", "align"], {})

        assert popen_mock.call_args.kwargs["start_new_session"] is True
        getpgid_mock.assert_called_once_with(4321)
        killpg_mock.assert_called_once_with(4321, signal.SIGKILL)

    def test_run_mfa_subprocess_timeout_diagnostic_is_bounded(self, tmp_path: Path) -> None:
        """The raised diagnostic must not balloon with unbounded subprocess output."""
        stage = _make_stage(tmp_path, align_timeout_seconds=5)

        fake_process = MagicMock()
        fake_process.pid = 1234
        fake_process.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd=["mfa"], timeout=5),
            ("x" * 10_000, "y" * 10_000),
        ]

        with (
            patch(f"{MODULE}.subprocess.Popen", return_value=fake_process),
            patch(f"{MODULE}.os.getpgid", return_value=1234),
            patch(f"{MODULE}.os.killpg"),
            pytest.raises(TimeoutError) as exc_info,
        ):
            stage._run_mfa_subprocess(["mfa", "align"], {})

        assert len(str(exc_info.value)) < 5_000
