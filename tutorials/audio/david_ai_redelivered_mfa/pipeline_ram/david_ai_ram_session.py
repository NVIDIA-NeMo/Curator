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

"""Per-session RAM pipeline: manifest + norm + 16k + MFA + Lhotse + mix in one pass."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from david_ai_common import (
    PipelineError,
    audio_16k_path,
    build_session_rttm_lines_from_words,
    ffprobe_duration,
    group_recordings_by_session,
    group_segments_by_recording,
    is_mixed_audio_done,
    is_session_alignment_done,
    load_session_rttm_by_speaker,
    log_exception,
    mark_alignment_done,
    mark_mixed_audio_done,
    mix_audio_files,
    prepare_speaker_audio_for_session_mix,
    resample_opus,
    session_alignment_textgrids_exist,
    session_mixed_audio_path,
    session_rttm_path,
    setup_mfa_worker_root,
    write_rttm,
)
from david_ai_ram_lhotse import (
    build_session_lhotse_cuts,
    session_has_recording_alignments,
    session_lhotse_path,
    write_recording_textgrids,
)
from stage0_build_manifests import build_session_manifests
from stage2_mfa_align_textgrids import align_session

_PROCESS_MFA: dict | None = None


def _mfa_scratch_base(ram_dir: Path) -> Path:
    """Base dir for MFA worker roots (extracted models + temp DB).

    MFA extracts the acoustic and G2P models and builds its temp database under
    the worker root. Placing that on tmpfs (/dev/shm, same as ``ram_dir``) can
    exhaust /dev/shm across many workers, truncating the ~30 MB G2P ``model.fst``
    and causing ``FstIOError: Read failed``. Default to node-local disk (/tmp)
    which has ample space; override with ``DAVIDAI_MFA_SCRATCH``.
    """
    override = os.environ.get("DAVIDAI_MFA_SCRATCH", "").strip()
    if override:
        return Path(override)
    return Path("/tmp/david_ai_mfa_scratch")


def _lazy_mfa_worker(
    *,
    ram_dir: Path,
    mfa_dict: Path,
    mfa_acoustic: str,
    mfa_g2p: str | None = None,
) -> tuple[Path, Path, str, str | None, Path]:
    global _PROCESS_MFA
    if _PROCESS_MFA is None:
        worker_dir = _mfa_scratch_base(ram_dir) / f"worker_{os.getpid()}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        mfa_root, local_dict, acoustic_arg, g2p_arg = setup_mfa_worker_root(
            worker_dir,
            mfa_dict=mfa_dict,
            mfa_acoustic=mfa_acoustic,
            mfa_g2p=mfa_g2p,
        )
        _PROCESS_MFA = {
            "worker_dir": worker_dir,
            "mfa_root": mfa_root,
            "local_dict": local_dict,
            "acoustic_arg": acoustic_arg,
            "g2p_arg": g2p_arg,
            "temp_parent": worker_dir / "align_temp",
        }
    cfg = _PROCESS_MFA
    return cfg["mfa_root"], cfg["local_dict"], cfg["acoustic_arg"], cfg["g2p_arg"], cfg["temp_parent"]


def ram_session_done_path(work_dir: Path, session_id: str) -> Path:
    return work_dir / ".done" / "sessions" / f"{session_id}.done"


def is_ram_session_done_marker(work_dir: Path, session_id: str) -> bool:
    return ram_session_done_path(work_dir, session_id).is_file()


def mark_ram_session_done(work_dir: Path, session_id: str) -> None:
    path = ram_session_done_path(work_dir, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n", encoding="utf-8")


def clear_ram_session_done(work_dir: Path, session_id: str) -> None:
    path = ram_session_done_path(work_dir, session_id)
    if path.is_file():
        path.unlink()


def clear_all_ram_session_done(work_dir: Path) -> None:
    done_dir = work_dir / ".done" / "sessions"
    if done_dir.is_dir():
        shutil.rmtree(done_dir, ignore_errors=True)


def speaker_ids_from_session_dir(session_dir: Path) -> list[str]:
    from david_ai_common import POSTPROCESSED_RE

    speaker_ids: set[str] = set()
    for wav in session_dir.glob("*_postprocessed.wav"):
        match = POSTPROCESSED_RE.match(wav.name)
        if match:
            speaker_ids.add(match.group(1))
    return sorted(speaker_ids)


def session_outputs_done(
    session_id: str,
    *,
    work_dir: Path,
    audio_16k_dir: Path,
    audio_mixed_dir: Path,
    lhotse_dir: Path,
    textgrid_dir: Path,
    speaker_ids: list[str],
) -> bool:
    if is_ram_session_done_marker(work_dir, session_id):
        return True
    if not is_mixed_audio_done(audio_mixed_dir, session_id):
        return False
    if not session_rttm_path(audio_mixed_dir, session_id).is_file():
        return False
    if not session_mixed_audio_path(audio_mixed_dir, session_id).is_file():
        return False
    if not session_alignment_textgrids_exist(textgrid_dir, session_id):
        return False
    for speaker_id in speaker_ids:
        if not audio_16k_path(audio_16k_dir, speaker_id, session_id).is_file():
            return False
    if session_has_recording_alignments(textgrid_dir, speaker_ids, session_id):
        if not session_lhotse_path(lhotse_dir, session_id).is_file():
            return False
    return True


def session_needs_ram_processing(
    session_dir: Path,
    *,
    work_dir: Path,
    audio_16k_dir: Path,
    audio_mixed_dir: Path,
    lhotse_dir: Path,
    textgrid_dir: Path,
    force: bool = False,
) -> bool:
    if force:
        return True
    speaker_ids = speaker_ids_from_session_dir(session_dir)
    if not speaker_ids:
        return True
    return not session_outputs_done(
        session_dir.name,
        work_dir=work_dir,
        audio_16k_dir=audio_16k_dir,
        audio_mixed_dir=audio_mixed_dir,
        lhotse_dir=lhotse_dir,
        textgrid_dir=textgrid_dir,
        speaker_ids=speaker_ids,
    )


@dataclass
class SessionRamResult:
    session_id: str
    ok: bool
    skipped: bool = False
    error: str = ""
    cuts: int = 0


def _unique_speaker_ids(norm_rows: list[dict]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in norm_rows:
        speaker_id = row["speaker_id"]
        if speaker_id not in seen:
            seen.add(speaker_id)
            out.append(speaker_id)
    return sorted(out)


def _encode_session_audio(
    norm_rows: list[dict],
    *,
    audio_16k_dir: Path,
    target_sr: int,
    opus_bitrate: str,
    force: bool,
) -> bool:
    by_recording = group_segments_by_recording(norm_rows)
    for rec_id, segments in by_recording.items():
        src = Path(segments[0]["audio_filepath"])
        dst = Path(segments[0]["audio_filepath_16k"])
        if dst.is_file() and not force:
            continue
        if force and dst.is_file():
            dst.unlink()
        if not src.is_file():
            raise FileNotFoundError(f"missing source audio for {rec_id}: {src}")
        if not resample_opus(src, dst, target_sr=target_sr, bitrate=opus_bitrate):
            raise PipelineError(f"16k encode failed for {rec_id}")
    return True


def _mix_session_from_existing_rttm(
    session_id: str,
    norm_rows: list[dict],
    *,
    audio_mixed_dir: Path,
    ram_scratch: Path,
    rttm_merge_gap: float,
    opus_bitrate: str,
    noise_level: float,
    preserve_speech: bool,
    stitch_ms: float,
    boundary_indent: float,
) -> bool:
    entries = group_recordings_by_session(norm_rows).get(session_id, [])
    if not entries:
        return False

    mixed_audio = session_mixed_audio_path(audio_mixed_dir, session_id)
    rttm_path = session_rttm_path(audio_mixed_dir, session_id)
    session_rttm = load_session_rttm_by_speaker(rttm_path, merge_gap=rttm_merge_gap)
    if not session_rttm:
        return False

    session_scratch = ram_scratch / session_id / "mix"
    if session_scratch.exists():
        shutil.rmtree(session_scratch, ignore_errors=True)
    session_scratch.mkdir(parents=True, exist_ok=True)

    prepared_paths: list[Path] = []
    try:
        for entry in entries:
            rec_id = entry["recording_id"]
            speaker_id = entry["speaker_id"]
            src = entry["audio_path"]
            if not src.is_file():
                raise FileNotFoundError(f"missing 16k audio {src}")

            speech = session_rttm.get(speaker_id)
            if not speech:
                prepared_paths.append(src)
                continue

            try:
                duration = ffprobe_duration(src)
            except RuntimeError:
                duration = max((end for _, end in speech), default=0.0) + 0.01

            dst = session_scratch / f"{rec_id}.opus"
            seed = hash((session_id, rec_id)) & 0xFFFFFFFF
            if not prepare_speaker_audio_for_session_mix(
                src,
                dst,
                speech_intervals=speech,
                audio_duration=duration,
                opus_bitrate=opus_bitrate,
                noise_level=noise_level,
                seed=seed,
                preserve_speech=preserve_speech,
                stitch_ms=stitch_ms,
                boundary_indent=boundary_indent,
            ):
                raise PipelineError(f"pause noise prep failed for {rec_id}")
            prepared_paths.append(dst)

        if not mix_audio_files(prepared_paths, mixed_audio, opus_bitrate=opus_bitrate):
            raise PipelineError("session mix failed")
        mark_mixed_audio_done(audio_mixed_dir, session_id)
        return True
    finally:
        shutil.rmtree(session_scratch, ignore_errors=True)


def _mix_session_from_alignment(
    session_id: str,
    norm_rows: list[dict],
    align_result,
    *,
    audio_mixed_dir: Path,
    ram_scratch: Path,
    rttm_merge_gap: float,
    opus_bitrate: str,
    noise_level: float,
    preserve_speech: bool,
    stitch_ms: float,
    boundary_indent: float,
) -> bool:
    entries = group_recordings_by_session(norm_rows).get(session_id, [])
    if not entries:
        return False

    mixed_audio = session_mixed_audio_path(audio_mixed_dir, session_id)
    rttm_path = session_rttm_path(audio_mixed_dir, session_id)
    session_rttm = load_session_rttm_by_speaker(rttm_path, merge_gap=rttm_merge_gap)

    session_scratch = ram_scratch / session_id / "mix"
    if session_scratch.exists():
        shutil.rmtree(session_scratch, ignore_errors=True)
    session_scratch.mkdir(parents=True, exist_ok=True)

    rec_durations = {rec["recording_id"]: float(rec.get("audio_duration", 0.0)) for rec in align_result.recordings}
    prepared_paths: list[Path] = []

    try:
        for entry in entries:
            rec_id = entry["recording_id"]
            speaker_id = entry["speaker_id"]
            src = entry["audio_path"]
            if not src.is_file():
                raise FileNotFoundError(f"missing 16k audio {src}")

            speech = session_rttm.get(speaker_id)
            if not speech:
                prepared_paths.append(src)
                continue

            duration = rec_durations.get(rec_id) or None
            dst = session_scratch / f"{rec_id}.opus"
            seed = hash((session_id, rec_id)) & 0xFFFFFFFF
            if not prepare_speaker_audio_for_session_mix(
                src,
                dst,
                speech_intervals=speech,
                audio_duration=duration,
                opus_bitrate=opus_bitrate,
                noise_level=noise_level,
                seed=seed,
                preserve_speech=preserve_speech,
                stitch_ms=stitch_ms,
                boundary_indent=boundary_indent,
            ):
                raise PipelineError(f"pause noise prep failed for {rec_id}")
            prepared_paths.append(dst)

        if not mix_audio_files(prepared_paths, mixed_audio, opus_bitrate=opus_bitrate):
            raise PipelineError("session mix failed")
        mark_mixed_audio_done(audio_mixed_dir, session_id)
        return True
    finally:
        shutil.rmtree(session_scratch, ignore_errors=True)


def process_session_ram(
    session_dir: Path,
    *,
    work_dir: Path,
    audio_16k_dir: Path,
    audio_mixed_dir: Path,
    lhotse_dir: Path,
    textgrid_dir: Path,
    mfa_dict: Path,
    mfa_acoustic: str,
    mfa_g2p: str | None = None,
    ram_dir: Path,
    num2words_lang: str = "en",
    mfa_num_jobs: int = 4,
    segment_padding: float = 0.5,
    rttm_merge_gap: float = 0.2,
    target_sr: int = 16000,
    opus_bitrate: str = "32k",
    noise_level: float = 0.0002,
    preserve_speech: bool = True,
    stitch_ms: float = 5.0,
    boundary_indent: float = 0.2,
    lexicon_dir: Path | None = None,
    force: bool = False,
) -> SessionRamResult:
    session_id = session_dir.name
    session_ram = ram_dir / "sessions" / session_id

    try:
        _raw_rows, norm_rows, _norm_logged, _log_entries = build_session_manifests(
            session_dir,
            audio_16k_dir=audio_16k_dir,
            num2words_lang=num2words_lang,
            lexicon_dir=lexicon_dir,
        )
        if not norm_rows:
            return SessionRamResult(session_id=session_id, ok=False, error="no manifest rows")

        speaker_ids = _unique_speaker_ids(norm_rows)
        if not force and session_outputs_done(
            session_id,
            work_dir=work_dir,
            audio_16k_dir=audio_16k_dir,
            audio_mixed_dir=audio_mixed_dir,
            lhotse_dir=lhotse_dir,
            textgrid_dir=textgrid_dir,
            speaker_ids=speaker_ids,
        ):
            return SessionRamResult(session_id=session_id, ok=True, skipped=True)

        if force:
            clear_ram_session_done(work_dir, session_id)

        _encode_session_audio(
            norm_rows,
            audio_16k_dir=audio_16k_dir,
            target_sr=target_sr,
            opus_bitrate=opus_bitrate,
            force=force,
        )

        if session_ram.exists():
            shutil.rmtree(session_ram, ignore_errors=True)
        session_ram.mkdir(parents=True, exist_ok=True)

        textgrid_dir.mkdir(parents=True, exist_ok=True)
        audio_mixed_dir.mkdir(parents=True, exist_ok=True)
        lhotse_dir.mkdir(parents=True, exist_ok=True)

        align_result = None
        need_mfa = force or not is_session_alignment_done(
            session_id,
            textgrid_dir=textgrid_dir,
            alignments_jsonl=None,
        )
        if need_mfa:
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
            )
            if not align_result.ok:
                return SessionRamResult(session_id=session_id, ok=False, error="MFA alignment failed")
            mark_alignment_done(textgrid_dir, session_id)
            write_recording_textgrids(align_result, textgrid_dir)

        cuts_count = build_session_lhotse_cuts(
            session_id,
            norm_rows,
            textgrid_dir=textgrid_dir,
            lhotse_dir=lhotse_dir,
            align_result=align_result,
            force=force,
        )

        rttm_path = session_rttm_path(audio_mixed_dir, session_id)
        if align_result is not None and (force or not rttm_path.is_file()):
            rttm_lines = build_session_rttm_lines_from_words(
                session_id,
                align_result.merged_words,
                align_result.fb_words,
                merge_gap=rttm_merge_gap,
            )
            if not rttm_lines:
                return SessionRamResult(session_id=session_id, ok=False, error="empty session RTTM")
            write_rttm(rttm_path, rttm_lines)
        elif not rttm_path.is_file():
            return SessionRamResult(session_id=session_id, ok=False, error="missing session RTTM for mix")

        if force or not is_mixed_audio_done(audio_mixed_dir, session_id):
            if align_result is not None:
                mixed_ok = _mix_session_from_alignment(
                    session_id,
                    norm_rows,
                    align_result,
                    audio_mixed_dir=audio_mixed_dir,
                    ram_scratch=session_ram,
                    rttm_merge_gap=rttm_merge_gap,
                    opus_bitrate=opus_bitrate,
                    noise_level=noise_level,
                    preserve_speech=preserve_speech,
                    stitch_ms=stitch_ms,
                    boundary_indent=boundary_indent,
                )
            else:
                mixed_ok = _mix_session_from_existing_rttm(
                    session_id,
                    norm_rows,
                    audio_mixed_dir=audio_mixed_dir,
                    ram_scratch=session_ram,
                    rttm_merge_gap=rttm_merge_gap,
                    opus_bitrate=opus_bitrate,
                    noise_level=noise_level,
                    preserve_speech=preserve_speech,
                    stitch_ms=stitch_ms,
                    boundary_indent=boundary_indent,
                )
            if not mixed_ok:
                return SessionRamResult(session_id=session_id, ok=False, error="session mix failed")

        if not session_alignment_textgrids_exist(textgrid_dir, session_id):
            return SessionRamResult(session_id=session_id, ok=False, error="missing session TextGrids")

        mark_ram_session_done(work_dir, session_id)
        return SessionRamResult(session_id=session_id, ok=True, cuts=cuts_count)
    except Exception as exc:
        log_exception(f"RAM session pipeline failed for {session_id}", exc)
        return SessionRamResult(session_id=session_id, ok=False, error=str(exc))
    finally:
        shutil.rmtree(session_ram, ignore_errors=True)
