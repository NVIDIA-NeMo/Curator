# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.tasks import AudioTask
from tutorials.audio.vad_refinement.mfa_alignment import MFAAlignmentStage


def _stage(tmp_path: Path) -> MFAAlignmentStage:
    stage = MFAAlignmentStage(
        output_dir=str(tmp_path / "out"),
        copy_models_to_local=False,
        mfa_root_dir=str(tmp_path / "models"),
        num_jobs=1,
        g2p_model="",
        create_rttm=False,
        create_ctm=False,
    )
    stage.setup_on_node()
    stage.setup()
    return stage


def test_batch_rejects_duplicate_filename_stems_before_mfa(tmp_path: Path) -> None:
    tasks = []
    for name in ("first", "second"):
        path = tmp_path / name / "same.wav"
        path.parent.mkdir()
        sf.write(path, np.zeros(1600), 16000)
        tasks.append(AudioTask(data={"audio_filepath": str(path), "text": "hello"}))
    with pytest.raises(ValueError, match="Duplicate audio filename stem"):
        _stage(tmp_path).process_batch(tasks)


def test_mfa_nonzero_exit_is_not_treated_as_alignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage = _stage(tmp_path)

    def failed(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        assert "align" in command
        assert "--output_format" in command
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="alignment failed")

    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="mfa align failed"):
        stage._run_mfa_align(tmp_path / "corpus", tmp_path / "textgrids")
