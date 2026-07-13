# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import david_ai_ram_session as ram_session
import pytest
from david_ai_common import (
    fastmss_textgrid_path,
    masked_speaker_audio_path,
    masked_speaker_rttm_path,
    recording_id,
    recording_textgrid_path,
    session_mixed_audio_path,
    session_rttm_path,
    session_textgrid_path,
)
from david_ai_manifest import resolve_speaker_audio_path
from david_ai_mfa_align import SessionAlignResult
from david_ai_ram_lhotse import write_all_textgrids
from stage_ram_session_pipeline import filter_sessions_from_file, sessions_without_done_flags


def _manifest_row(tmp_path: Path, session_id: str, speaker_id: str, start: float, end: float) -> dict:
    rec_id = recording_id(speaker_id, session_id)
    audio_path = tmp_path / f"{rec_id}.wav"
    audio_path.write_bytes(b"source")
    return {
        "session_id": session_id,
        "speaker_id": speaker_id,
        "recording_id": rec_id,
        "segment_index": 0,
        "start": start,
        "end": end,
        "audio_filepath": str(audio_path),
        "audio_filepath_16k": str(audio_path),
    }


def test_speaker_audio_resolution_priority_and_fallback(tmp_path: Path) -> None:
    speaker_id = "speaker"
    preprocessed = tmp_path / f"{speaker_id}_preprocessed.wav"
    ordinary = tmp_path / f"{speaker_id}.wav"
    postprocessed = tmp_path / f"{speaker_id}_postprocessed.wav"
    postprocess = tmp_path / f"{speaker_id}_postprocess.wav"

    with pytest.raises(FileNotFoundError):
        resolve_speaker_audio_path(tmp_path, speaker_id)
    preprocessed.touch()
    assert resolve_speaker_audio_path(tmp_path, speaker_id) == preprocessed
    ordinary.touch()
    assert resolve_speaker_audio_path(tmp_path, speaker_id) == ordinary
    postprocessed.touch()
    assert resolve_speaker_audio_path(tmp_path, speaker_id) == postprocessed
    postprocess.touch()
    assert resolve_speaker_audio_path(tmp_path, speaker_id) == postprocess


def test_mix_persists_masked_speaker_and_mixed_wavs(tmp_path: Path, monkeypatch) -> None:
    session_id = "session"
    rows = [
        _manifest_row(tmp_path, session_id, "speaker-a", 1.0, 2.0),
        _manifest_row(tmp_path, session_id, "speaker-b", 3.0, 4.0),
    ]
    captured: dict[str, tuple[list[tuple[float, float]], float]] = {}

    def fake_prepare(src: Path, dst: Path, *, speech_intervals, boundary_indent, **kwargs) -> bool:
        assert dst.suffix == ".wav"
        captured[src.name] = (speech_intervals, boundary_indent)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(f"prepared:{src.name}".encode())
        return True

    def fake_mix(audio_paths: list[Path], output_path: Path) -> bool:
        assert output_path.suffix == ".wav"
        assert all(path.suffix == ".wav" for path in audio_paths)
        output_path.write_bytes(b"|".join(path.read_bytes() for path in audio_paths))
        return True

    monkeypatch.setattr(ram_session, "prepare_speaker_audio_for_session_mix", fake_prepare)
    monkeypatch.setattr(ram_session, "mix_audio_files", fake_mix)

    audio_masked_dir = tmp_path / "audio_16k_masked"
    audio_mixed_dir = tmp_path / "audio_mixed"
    ram_session._mix_session_from_manifest(
        session_id,
        rows,
        audio_masked_dir=audio_masked_dir,
        audio_mixed_dir=audio_mixed_dir,
        session_ram=tmp_path / "scratch",
        noise_level=0.0002,
        stitch_ms=5.0,
        boundary_offset=0.5,
        rec_durations={row["recording_id"]: 10.0 for row in rows},
    )

    assert captured[f"{rows[0]['recording_id']}.wav"] == ([(1.0, 2.0)], 0.5)
    assert captured[f"{rows[1]['recording_id']}.wav"] == ([(3.0, 4.0)], 0.5)
    for row in rows:
        assert masked_speaker_audio_path(audio_masked_dir, row["speaker_id"], session_id).is_file()
    assert session_mixed_audio_path(audio_mixed_dir, session_id).suffix == ".wav"
    assert session_mixed_audio_path(audio_mixed_dir, session_id).is_file()


def test_masked_speaker_rttms_are_filtered_from_session_rttm(tmp_path: Path) -> None:
    session_id = "session"
    rows = [
        _manifest_row(tmp_path, session_id, "speaker-a", 1.0, 2.0),
        _manifest_row(tmp_path, session_id, "speaker-b", 3.0, 4.0),
    ]
    lines = [
        f"SPEAKER {session_id} 1 1.000000 1.000000 <NA> <NA> speaker-a <NA> <NA>",
        f"SPEAKER {session_id} 1 3.000000 1.000000 <NA> <NA> speaker-b <NA> <NA>",
    ]

    ram_session._write_masked_speaker_rttms(
        session_id,
        rows,
        lines,
        audio_masked_dir=tmp_path,
    )

    for speaker_id in ("speaker-a", "speaker-b"):
        text = masked_speaker_rttm_path(tmp_path, speaker_id, session_id).read_text()
        assert recording_id(speaker_id, session_id) in text
        assert f" {speaker_id} " in text
        other = "speaker-b" if speaker_id == "speaker-a" else "speaker-a"
        assert f" {other} " not in text


def test_write_all_textgrids_writes_both_variants_when_fastmss_is_empty(tmp_path: Path) -> None:
    session_id = "session"
    rec_id = recording_id("speaker", session_id)
    result = SessionAlignResult(
        ok=True,
        fb_words=[(1.0, 2.0, "speech", "speaker")],
        audio_duration=3.0,
        recordings=[
            {
                "session_id": session_id,
                "recording_id": rec_id,
                "merged_words": [],
                "fb_words": [[1.0, 2.0, "speech"]],
                "audio_duration": 3.0,
            }
        ],
    )

    write_all_textgrids(result, tmp_path)

    assert session_textgrid_path(tmp_path, session_id, variant="fastmss").is_file()
    assert session_textgrid_path(tmp_path, session_id, variant="ordinary").is_file()
    assert fastmss_textgrid_path(tmp_path, rec_id).is_file()
    assert recording_textgrid_path(tmp_path, rec_id, variant="ordinary").is_file()


def test_done_flag_requires_mixed_wav_rttm_and_textgrids(tmp_path: Path) -> None:
    session_id = "session"
    speaker_id = "speaker"
    row = _manifest_row(tmp_path, session_id, speaker_id, 1.0, 2.0)
    work_dir = tmp_path / "work"
    audio_masked_dir = work_dir / "audio_16k_masked"
    audio_mixed_dir = work_dir / "audio_mixed"
    textgrid_dir = work_dir / "textgrids"

    with pytest.raises(ram_session.PipelineError):
        ram_session._finalize_session_success(
            session_id,
            [row],
            work_dir=work_dir,
            audio_masked_dir=audio_masked_dir,
            audio_mixed_dir=audio_mixed_dir,
            textgrid_dir=textgrid_dir,
        )
    assert not ram_session.session_done_path(work_dir, session_id).exists()

    rec_id = row["recording_id"]
    outputs = [
        session_mixed_audio_path(audio_mixed_dir, session_id),
        session_rttm_path(audio_mixed_dir, session_id),
        session_textgrid_path(textgrid_dir, session_id, variant="ordinary"),
        session_textgrid_path(textgrid_dir, session_id, variant="fastmss"),
        masked_speaker_audio_path(audio_masked_dir, speaker_id, session_id),
        masked_speaker_rttm_path(audio_masked_dir, speaker_id, session_id),
        recording_textgrid_path(textgrid_dir, rec_id, variant="ordinary"),
        fastmss_textgrid_path(textgrid_dir, rec_id),
    ]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"output")

    ram_session._finalize_session_success(
        session_id,
        [row],
        work_dir=work_dir,
        audio_masked_dir=audio_masked_dir,
        audio_mixed_dir=audio_mixed_dir,
        textgrid_dir=textgrid_dir,
    )
    assert ram_session.session_done_path(work_dir, session_id).read_text() == "ok\n"


def test_parallel_resume_selects_only_sessions_without_done_flags(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    sessions = [tmp_path / "session-a", tmp_path / "session-b"]
    ram_session._mark_session_done(work_dir, "session-a")

    pending = sessions_without_done_flags(sessions, work_dir)

    assert [session.name for session in pending] == ["session-b"]


def test_session_list_restricts_discovered_sessions(tmp_path: Path) -> None:
    sessions = [tmp_path / name for name in ("session-a", "session-b", "session-c")]
    sessions_file = tmp_path / "sessions.txt"
    sessions_file.write_text("# subset\nsession-c\n\nsession-a\nmissing-session\n", encoding="utf-8")

    selected = filter_sessions_from_file(sessions, sessions_file)

    assert [session.name for session in selected] == ["session-a", "session-c"]
