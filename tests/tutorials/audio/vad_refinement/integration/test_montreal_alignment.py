# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

from nemo_curator.tasks import AudioTask
from tutorials.audio.vad_refinement.montreal_alignment import MontrealAligner


def test_cli_failure_is_reported_with_log_path(tmp_path: Path) -> None:
    executable = tmp_path / "failed_mfa"
    executable.write_text(f"#!{sys.executable}\nraise SystemExit(7)\n")
    executable.chmod(0o755)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture-audio")
    aligner = MontrealAligner(
        tmp_path / "output", executable=str(executable), dictionary="dictionary", acoustic_model="model"
    )
    try:
        task = AudioTask(
            data={
                "segment_id": "clip",
                "speaker_id": "speaker",
                "audio_filepath": str(audio),
                "text": "supplied transcript",
            }
        )
        with pytest.raises(RuntimeError, match="MFA failed with exit code 7; see"):
            aligner.process_batch([task])
        assert "mfa_textgrid" not in task.data
        assert len(list((tmp_path / "output").rglob("mfa.log"))) == 1
    finally:
        aligner.close()


def test_transcript_is_required_before_external_command(tmp_path: Path) -> None:
    aligner = MontrealAligner(
        tmp_path / "output", executable="must-not-be-executed", dictionary="dictionary", acoustic_model="model"
    )
    try:
        task = AudioTask(data={"segment_id": "clip", "speaker_id": "speaker", "text": ""})
        with pytest.raises(ValueError, match="nonempty supplied transcript"):
            aligner.process_batch([task])
        assert not list((tmp_path / "output").rglob("mfa.log"))
    finally:
        aligner.close()
