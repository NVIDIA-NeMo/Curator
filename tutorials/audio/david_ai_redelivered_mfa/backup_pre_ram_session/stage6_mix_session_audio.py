#!/usr/bin/env python3
"""Stage 6: white-noise pauses per speaker (from session RTTM), then mix to session Opus."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from david_ai_common import (
    PipelineError,
    all_mixed_audio_done,
    clear_mixed_audio_done,
    ffprobe_duration,
    finish_stage,
    group_recordings_by_session,
    is_mixed_audio_done,
    load_alignments_by_recording,
    load_norm_manifest_rows,
    load_session_rttm_by_speaker,
    log_exception,
    mark_mixed_audio_done,
    mix_audio_files,
    parse_rttm_speech_intervals,
    prepare_speaker_audio_for_session_mix,
    run_main,
    run_thread_pool,
    session_mixed_audio_path,
    session_rttm_path,
    speech_intervals_from_recording_alignment,
)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _speech_intervals_for_speaker(
    speaker_id: str,
    recording_id: str,
    *,
    session_rttm: dict[str, list[tuple[float, float]]],
    alignments: dict[str, dict],
    recording_rttm_path: Path | None,
    rttm_merge_gap: float,
) -> list[tuple[float, float]] | None:
    if speaker_id in session_rttm and session_rttm[speaker_id]:
        return session_rttm[speaker_id]

    rec_row = alignments.get(recording_id)
    if rec_row is not None:
        logger.warning(
            "%s: speaker %s missing from session RTTM; using alignments for %s",
            recording_id,
            speaker_id,
            recording_id,
        )
        return speech_intervals_from_recording_alignment(rec_row, merge_gap=rttm_merge_gap)

    if recording_rttm_path is not None and recording_rttm_path.is_file():
        lines = recording_rttm_path.read_text(encoding="utf-8").splitlines()
        intervals = parse_rttm_speech_intervals(lines, merge_gap=rttm_merge_gap)
        if intervals:
            logger.warning(
                "%s: using per-recording RTTM fallback %s",
                speaker_id,
                recording_rttm_path.name,
            )
            return intervals

    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests-dir", type=Path, required=True)
    ap.add_argument(
        "--audio-mixed-dir",
        type=Path,
        required=True,
        help="Output dir; session RTTM read from {session_id}.rttm here (stage 4)",
    )
    ap.add_argument(
        "--alignments-jsonl",
        type=Path,
        default=None,
        help="Fallback per-recording speech intervals (default: <workdir>/alignments.jsonl)",
    )
    ap.add_argument(
        "--rttm-dir",
        type=Path,
        default=None,
        help="Optional per-recording RTTM fallback when session RTTM lacks a speaker",
    )
    ap.add_argument(
        "--rttm-merge-gap",
        type=float,
        default=0.2,
        help="Merge gap for speech intervals (same as stage 4 RTTM)",
    )
    ap.add_argument(
        "--noise-level",
        type=float,
        default=0.0002,
        help="White-noise amplitude for pause regions (0–1 scale on PCM samples)",
    )
    ap.add_argument(
        "--preserve-speech",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not modify RTTM speech regions (default: only pause gaps get noise)",
    )
    ap.add_argument(
        "--stitch-ms",
        type=float,
        default=5.0,
        help="Crossfade inside pause regions (original pause audio <-> noise) when --preserve-speech",
    )
    ap.add_argument(
        "--boundary-indent",
        type=float,
        default=0.2,
        help="Seconds of original audio kept untouched on each side of every speech interval",
    )
    ap.add_argument("--opus-bitrate", default="32k", help="Opus encoder bitrate for mixed output")
    ap.add_argument("--session", action="append", default=[], help="Optional session_id filter")
    ap.add_argument("--work-dir", type=Path, default=None, help="Work dir for .done marker")
    ap.add_argument("--stage-done-name", default=None, help="Stage name for .done marker")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="Parallel session mix workers")
    args = ap.parse_args()

    manifests_dir = args.manifests_dir.resolve()
    audio_mixed_dir = args.audio_mixed_dir.resolve()
    alignments_jsonl = (
        args.alignments_jsonl.resolve()
        if args.alignments_jsonl
        else (manifests_dir.parent / "alignments.jsonl")
    )
    rttm_dir = args.rttm_dir.resolve() if args.rttm_dir else None

    alignments: dict[str, dict] = {}
    if alignments_jsonl.is_file():
        alignments = load_alignments_by_recording(alignments_jsonl)

    rows, manifest_errors = load_norm_manifest_rows(manifests_dir, sessions=args.session or None)
    by_session = group_recordings_by_session(rows)
    if not by_session:
        raise PipelineError("No sessions found in normalized manifests")

    session_ids = sorted(by_session.keys())
    if not args.force and all_mixed_audio_done(audio_mixed_dir, session_ids):
        logger.info("All %d sessions already mixed (%s/.done), skipping stage 6", len(session_ids), audio_mixed_dir)
        return finish_stage(args.work_dir, args.stage_done_name, 0)

    missing_session_rttm = [
        sid for sid in session_ids if not session_rttm_path(audio_mixed_dir, sid).is_file()
    ]
    if missing_session_rttm and not alignments and rttm_dir is None:
        raise PipelineError(
            f"Missing session RTTM for {len(missing_session_rttm)} sessions in {audio_mixed_dir} "
            "(run stage 4 first) and no alignments/rttm-dir fallback"
        )
    if missing_session_rttm:
        logger.warning(
            "%d sessions missing %s/*.rttm; will fall back to alignments/per-recording RTTM",
            len(missing_session_rttm),
            audio_mixed_dir,
        )

    audio_mixed_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = audio_mixed_dir / ".mix_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, args.workers)

    def _mix_one(item: tuple[str, list[dict]]) -> str:
        session_id, entries = item
        mixed_audio = session_mixed_audio_path(audio_mixed_dir, session_id)
        if args.force:
            clear_mixed_audio_done(audio_mixed_dir, session_id)
            legacy_wav = audio_mixed_dir / f"{session_id}.wav"
            if legacy_wav.is_file():
                legacy_wav.unlink()
        elif is_mixed_audio_done(audio_mixed_dir, session_id):
            return "skip"
        elif mixed_audio.is_file():
            mark_mixed_audio_done(audio_mixed_dir, session_id)
            logger.info("%s: existing mixed audio, marked done", session_id)
            return "skip"

        session_scratch = scratch_dir / session_id
        if session_scratch.exists():
            shutil.rmtree(session_scratch, ignore_errors=True)
        session_scratch.mkdir(parents=True, exist_ok=True)

        session_rttm = load_session_rttm_by_speaker(
            session_rttm_path(audio_mixed_dir, session_id),
            merge_gap=args.rttm_merge_gap,
        )
        if session_rttm:
            logger.info(
                "%s: using session RTTM (%d speakers, %d lines)",
                session_id,
                len(session_rttm),
                sum(len(v) for v in session_rttm.values()),
            )

        try:
            prepared_paths: list[Path] = []
            for entry in entries:
                rec_id = entry["recording_id"]
                speaker_id = entry["speaker_id"]
                src = entry["audio_path"]
                if not src.is_file():
                    logger.warning("%s: missing speaker audio %s", session_id, src)
                    return "fail"

                speech = _speech_intervals_for_speaker(
                    speaker_id,
                    rec_id,
                    session_rttm=session_rttm,
                    alignments=alignments,
                    recording_rttm_path=(rttm_dir / f"{rec_id}.rttm") if rttm_dir else None,
                    rttm_merge_gap=args.rttm_merge_gap,
                )
                if speech is None:
                    logger.warning(
                        "%s: no speech intervals for speaker %s; using raw audio",
                        session_id,
                        speaker_id,
                    )
                    prepared_paths.append(src)
                    continue

                rec_row = alignments.get(rec_id, {})
                duration = float(rec_row.get("audio_duration", 0.0)) or None
                if duration is None:
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
                    opus_bitrate=args.opus_bitrate,
                    noise_level=args.noise_level,
                    seed=seed,
                    preserve_speech=args.preserve_speech,
                    stitch_ms=args.stitch_ms,
                    boundary_indent=args.boundary_indent,
                ):
                    logger.warning("%s: pause noise prep failed for %s", session_id, rec_id)
                    return "fail"
                prepared_paths.append(dst)

            if mix_audio_files(prepared_paths, mixed_audio, opus_bitrate=args.opus_bitrate):
                mark_mixed_audio_done(audio_mixed_dir, session_id)
                logger.info(
                    "%s: mixed %d speaker tracks (session RTTM pause mask) -> %s",
                    session_id,
                    len(entries),
                    mixed_audio.name,
                )
                return "ok"
            logger.warning("%s: failed to mix session audio", session_id)
        except Exception as exc:
            log_exception(f"session audio mix failed for {session_id}", exc)
        finally:
            shutil.rmtree(session_scratch, ignore_errors=True)
        return "fail"

    outcomes = run_thread_pool(sorted(by_session.items()), _mix_one, workers=workers)
    ok = outcomes.count("ok")
    skip = outcomes.count("skip")
    fail = outcomes.count("fail")

    logger.info(
        "Stage 6 done: ok=%d skip=%d fail=%d sessions=%d manifest_errors=%d workers=%d",
        ok,
        skip,
        fail,
        len(by_session),
        manifest_errors,
        workers,
    )
    exit_code = 1 if (fail or manifest_errors) else 0
    return finish_stage(args.work_dir, args.stage_done_name, exit_code)


if __name__ == "__main__":
    run_main(main)
