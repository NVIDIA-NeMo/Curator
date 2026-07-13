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

"""Per-session worker for the strict on-the-fly RAM E2E pipeline."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from david_ai_common import (
    PipelineError,
    build_session_rttm_lines_from_words,
    fastmss_textgrid_path,
    ffprobe_duration,
    group_recordings_by_session,
    group_segments_by_recording,
    log_exception,
    masked_speaker_audio_path,
    masked_speaker_rttm_path,
    mix_audio_files,
    prepare_speaker_audio_for_session_mix,
    recording_id,
    recording_textgrid_path,
    run_thread_pool,
    session_mixed_audio_path,
    session_rttm_path,
    session_textgrid_path,
    setup_mfa_worker_root,
    write_rttm,
)
from david_ai_manifest import build_session_rows
from david_ai_mfa_align import align_session
from david_ai_ram_lhotse import write_all_textgrids

if TYPE_CHECKING:
    from pathlib import Path

_PROCESS_MFA: dict | None = None


def _lazy_mfa_worker(
    *,
    ram_dir: Path,
    mfa_dict: Path,
    mfa_acoustic: str,
    mfa_g2p: str,
) -> tuple[Path, Path, str, str | None, Path]:
    """Initialize one ephemeral MFA model/database root per process."""
    global _PROCESS_MFA
    if _PROCESS_MFA is None:
        worker_dir = ram_dir / "mfa_workers" / f"worker_{os.getpid()}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        mfa_root, local_dict, acoustic_arg, g2p_arg = setup_mfa_worker_root(
            worker_dir,
            mfa_dict=mfa_dict,
            mfa_acoustic=mfa_acoustic,
            mfa_g2p=mfa_g2p,
        )
        _PROCESS_MFA = {
            "mfa_root": mfa_root,
            "local_dict": local_dict,
            "acoustic_arg": acoustic_arg,
            "g2p_arg": g2p_arg,
            "temp_parent": worker_dir / "align_temp",
        }
    cfg = _PROCESS_MFA
    return cfg["mfa_root"], cfg["local_dict"], cfg["acoustic_arg"], cfg["g2p_arg"], cfg["temp_parent"]


@dataclass
class SessionRamResult:
    session_id: str
    ok: bool
    error: str = ""


def session_done_path(work_dir: Path, session_id: str) -> Path:
    return work_dir / ".done" / "sessions" / f"{session_id}.done"


def is_session_done(work_dir: Path, session_id: str) -> bool:
    """Return whether a previous run validated every required session output."""
    return session_done_path(work_dir, session_id).is_file()


def _clear_session_done(work_dir: Path, session_id: str) -> None:
    session_done_path(work_dir, session_id).unlink(missing_ok=True)


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _validate_session_outputs(
    session_id: str,
    norm_rows: list[dict],
    *,
    audio_masked_dir: Path,
    audio_mixed_dir: Path,
    textgrid_dir: Path,
) -> None:
    """Require every declared deliverable before writing the session success flag."""
    required = [
        session_mixed_audio_path(audio_mixed_dir, session_id),
        session_rttm_path(audio_mixed_dir, session_id),
        session_textgrid_path(textgrid_dir, session_id, variant="ordinary"),
        session_textgrid_path(textgrid_dir, session_id, variant="fastmss"),
    ]
    speaker_ids = sorted({row["speaker_id"] for row in norm_rows})
    for speaker_id in speaker_ids:
        rec_id = recording_id(speaker_id, session_id)
        required.extend(
            [
                masked_speaker_audio_path(audio_masked_dir, speaker_id, session_id),
                masked_speaker_rttm_path(audio_masked_dir, speaker_id, session_id),
                recording_textgrid_path(textgrid_dir, rec_id, variant="ordinary"),
                fastmss_textgrid_path(textgrid_dir, rec_id),
            ]
        )
    missing = [str(path) for path in required if not _nonempty(path)]
    if missing:
        msg = f"missing or empty session outputs: {missing}"
        raise PipelineError(msg)


def _mark_session_done(work_dir: Path, session_id: str) -> None:
    path = session_done_path(work_dir, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n", encoding="utf-8")


def _finalize_session_success(
    session_id: str,
    norm_rows: list[dict],
    *,
    work_dir: Path,
    audio_masked_dir: Path,
    audio_mixed_dir: Path,
    textgrid_dir: Path,
) -> None:
    _validate_session_outputs(
        session_id,
        norm_rows,
        audio_masked_dir=audio_masked_dir,
        audio_mixed_dir=audio_mixed_dir,
        textgrid_dir=textgrid_dir,
    )
    _mark_session_done(work_dir, session_id)


def _mix_prep_workers(num_speakers: int) -> int:
    raw = os.environ.get("MIX_PREP_WORKERS", "").strip()
    if raw:
        try:
            return max(1, min(int(raw), num_speakers))
        except ValueError:
            pass
    return max(1, num_speakers)


def _manifest_speech_intervals_by_recording(norm_rows: list[dict]) -> dict[str, list[tuple[float, float]]]:
    """Return exact original manifest boundaries, before the 0.5-second protection offset."""
    return {
        rec_id: [(float(row["start"]), float(row["end"])) for row in rows]
        for rec_id, rows in group_segments_by_recording(norm_rows).items()
    }


def _prepare_speaker_tracks_for_mix(
    entries: list[dict],
    *,
    session_id: str,
    session_scratch: Path,
    audio_masked_dir: Path,
    manifest_speech: dict[str, list[tuple[float, float]]],
    rec_durations: dict[str, float],
    noise_level: float,
    stitch_ms: float,
    boundary_offset: float,
) -> list[tuple[Path, Path]]:
    """Create 16 kHz masked speaker WAVs and their persistent destinations."""
    specs: list[tuple[Path, Path, Path, list[tuple[float, float]], float, int, str]] = []
    for entry in entries:
        rec_id = entry["recording_id"]
        speaker_id = entry["speaker_id"]
        src = entry["audio_path"]
        if not src.is_file():
            msg = f"missing source audio {src}"
            raise FileNotFoundError(msg)

        speech = manifest_speech.get(rec_id)
        if not speech:
            msg = f"no original manifest boundaries for {rec_id}"
            raise PipelineError(msg)
        duration = rec_durations.get(rec_id, 0.0)
        if duration <= 0:
            try:
                duration = ffprobe_duration(src)
            except RuntimeError:
                duration = max(end for _, end in speech) + 0.01

        local_dst = session_scratch / f"{rec_id}.wav"
        persistent_dst = masked_speaker_audio_path(audio_masked_dir, speaker_id, session_id)
        seed = hash((session_id, rec_id)) & 0xFFFFFFFF
        specs.append((src, local_dst, persistent_dst, speech, duration, seed, rec_id))

    def _prepare_one(
        spec: tuple[Path, Path, Path, list[tuple[float, float]], float, int, str],
    ) -> tuple[Path, Path]:
        src, local_dst, persistent_dst, speech, duration, seed, rec_id = spec
        if not prepare_speaker_audio_for_session_mix(
            src,
            local_dst,
            speech_intervals=speech,
            audio_duration=duration,
            noise_level=noise_level,
            seed=seed,
            preserve_speech=True,
            stitch_ms=stitch_ms,
            boundary_indent=boundary_offset,
        ):
            msg = f"pause noise prep failed for {rec_id}"
            raise PipelineError(msg)
        return local_dst, persistent_dst

    return run_thread_pool(specs, _prepare_one, workers=_mix_prep_workers(len(specs)))


def _publish_audio(local_path: Path, output_path: Path) -> None:
    """Publish completed local audio without exposing a partial final path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(local_path, temp_path)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.is_file():
            temp_path.unlink()


def _write_masked_speaker_rttms(
    session_id: str,
    norm_rows: list[dict],
    session_rttm_lines: list[str],
    *,
    audio_masked_dir: Path,
) -> None:
    """Filter the session RTTM into one recording-scoped RTTM per speaker."""
    audio_masked_dir.mkdir(parents=True, exist_ok=True)
    for speaker_id in sorted({row["speaker_id"] for row in norm_rows}):
        rec_id = recording_id(speaker_id, session_id)
        output_lines: list[str] = []
        for line in session_rttm_lines:
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER" or parts[7] != speaker_id:
                continue
            parts[1] = rec_id
            output_lines.append(" ".join(parts))
        if not output_lines:
            msg = f"no RTTM intervals for masked speaker {rec_id}"
            raise PipelineError(msg)
        masked_speaker_rttm_path(audio_masked_dir, speaker_id, session_id).write_text(
            "\n".join(output_lines) + "\n",
            encoding="utf-8",
        )


def _mix_session_from_manifest(
    session_id: str,
    norm_rows: list[dict],
    *,
    audio_masked_dir: Path,
    audio_mixed_dir: Path,
    session_ram: Path,
    noise_level: float,
    stitch_ms: float,
    boundary_offset: float,
    rec_durations: dict[str, float],
) -> None:
    entries = group_recordings_by_session(norm_rows).get(session_id, [])
    if not entries:
        msg = "no speaker recordings to mix"
        raise PipelineError(msg)

    session_scratch = session_ram / "mix"
    if session_scratch.exists():
        shutil.rmtree(session_scratch, ignore_errors=True)
    session_scratch.mkdir(parents=True, exist_ok=True)

    prepared_tracks = _prepare_speaker_tracks_for_mix(
        entries,
        session_id=session_id,
        session_scratch=session_scratch,
        audio_masked_dir=audio_masked_dir,
        manifest_speech=_manifest_speech_intervals_by_recording(norm_rows),
        rec_durations=rec_durations,
        noise_level=noise_level,
        stitch_ms=stitch_ms,
        boundary_offset=boundary_offset,
    )
    local_mixed = session_scratch / f"{session_id}.wav"
    if not mix_audio_files([local for local, _ in prepared_tracks], local_mixed):
        msg = "session mix failed"
        raise PipelineError(msg)

    for local_path, persistent_path in prepared_tracks:
        _publish_audio(local_path, persistent_path)
    _publish_audio(local_mixed, session_mixed_audio_path(audio_mixed_dir, session_id))


def process_session_ram(
    session_dir: Path,
    *,
    work_dir: Path,
    audio_masked_dir: Path,
    audio_mixed_dir: Path,
    textgrid_dir: Path,
    mfa_dict: Path,
    mfa_acoustic: str,
    mfa_g2p: str,
    ram_dir: Path,
    num2words_lang: str = "en",
    mfa_num_jobs: int = 2,
    segment_padding: float = 0.5,
    rttm_merge_gap: float = 0.2,
    noise_level: float = 0.0002,
    stitch_ms: float = 5.0,
    boundary_offset: float = 0.5,
) -> SessionRamResult:
    """Run every E2E step from raw transcript/WAV, without reading persisted pipeline state."""
    session_id = session_dir.name
    session_ram = ram_dir / "sessions" / session_id
    _clear_session_done(work_dir, session_id)

    try:
        norm_rows = build_session_rows(
            session_dir,
            num2words_lang=num2words_lang,
        )
        if not norm_rows:
            return SessionRamResult(session_id=session_id, ok=False, error="no manifest rows")

        if session_ram.exists():
            shutil.rmtree(session_ram, ignore_errors=True)
        session_ram.mkdir(parents=True, exist_ok=True)
        audio_masked_dir.mkdir(parents=True, exist_ok=True)
        textgrid_dir.mkdir(parents=True, exist_ok=True)
        audio_mixed_dir.mkdir(parents=True, exist_ok=True)

        worker_mfa_root, local_dict, acoustic_arg, g2p_arg, temp_parent = _lazy_mfa_worker(
            ram_dir=ram_dir,
            mfa_dict=mfa_dict,
            mfa_acoustic=mfa_acoustic,
            mfa_g2p=mfa_g2p,
        )
        temp_parent.mkdir(parents=True, exist_ok=True)
        align_result = align_session(
            session_id,
            norm_rows,
            mfa_dict=local_dict,
            mfa_acoustic=mfa_acoustic,
            textgrid_dir=textgrid_dir,
            temp_parent=temp_parent / session_id,
            num_jobs=mfa_num_jobs,
            fallback_log=session_ram / "fallback.jsonl",
            segment_padding=segment_padding,
            worker_mfa_root=worker_mfa_root,
            worker_acoustic=acoustic_arg,
            worker_g2p=g2p_arg,
            mfa_g2p=mfa_g2p,
            keep_temp=False,
            use_fallback=True,
            write_textgrids=False,
        )
        if not align_result.ok:
            return SessionRamResult(session_id=session_id, ok=False, error="MFA alignment failed")

        rttm_lines = build_session_rttm_lines_from_words(
            session_id,
            align_result.merged_words,
            align_result.fb_words,
            merge_gap=rttm_merge_gap,
        )
        if not rttm_lines:
            return SessionRamResult(session_id=session_id, ok=False, error="empty session RTTM")
        write_rttm(session_rttm_path(audio_mixed_dir, session_id), rttm_lines)
        _write_masked_speaker_rttms(
            session_id,
            norm_rows,
            rttm_lines,
            audio_masked_dir=audio_masked_dir,
        )

        write_all_textgrids(align_result, textgrid_dir)

        rec_durations = {
            rec["recording_id"]: float(rec.get("audio_duration", 0.0))
            for rec in align_result.recordings
        }
        _mix_session_from_manifest(
            session_id,
            norm_rows,
            audio_masked_dir=audio_masked_dir,
            audio_mixed_dir=audio_mixed_dir,
            session_ram=session_ram,
            noise_level=noise_level,
            stitch_ms=stitch_ms,
            boundary_offset=boundary_offset,
            rec_durations=rec_durations,
        )
        _finalize_session_success(
            session_id,
            norm_rows,
            work_dir=work_dir,
            audio_masked_dir=audio_masked_dir,
            audio_mixed_dir=audio_mixed_dir,
            textgrid_dir=textgrid_dir,
        )
        return SessionRamResult(session_id=session_id, ok=True)
    except Exception as exc:
        log_exception(f"RAM session pipeline failed for {session_id}", exc)
        return SessionRamResult(session_id=session_id, ok=False, error=str(exc))
    finally:
        shutil.rmtree(session_ram, ignore_errors=True)
