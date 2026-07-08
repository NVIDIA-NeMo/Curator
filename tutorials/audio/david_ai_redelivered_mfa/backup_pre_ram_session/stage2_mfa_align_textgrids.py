#!/usr/bin/env python3
"""Stage 2: MFA align per segment, concatenate to per-session TextGrids + alignments cache."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from david_ai_common import (
    PipelineError,
    append_jsonl,
    extract_segment_wav,
    ffprobe_duration,
    finish_stage,
    group_segments_by_recording,
    group_segments_by_session,
    load_alignment_ids,
    load_norm_manifest_rows,
    log_exception,
    map_segment_words_to_recording,
    maybe_skip_done_stage,
    mfa_models_root,
    partition_list,
    resolve_mfa_acoustic_model,
    run_main,
    safe_parse_textgrid_words,
    segment_fallback_log_entry,
    session_textgrid_path,
    setup_mfa_worker_root,
    words_to_json,
    write_textgrid,
)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RecordingAlignResult:
    ok: bool
    mfa_segments: int = 0
    fallback_segments: int = 0
    fallback_entries: list[dict] = field(default_factory=list)
    merged_words: list[tuple[float, float, str]] = field(default_factory=list)
    fb_words: list[tuple[float, float, str]] = field(default_factory=list)
    audio_duration: float = 0.0


@dataclass
class SessionAlignResult:
    ok: bool
    mfa_segments: int = 0
    fallback_segments: int = 0
    merged_words: list[tuple[float, float, str, str]] = field(default_factory=list)
    fb_words: list[tuple[float, float, str, str]] = field(default_factory=list)
    audio_duration: float = 0.0
    recordings: list[dict] = field(default_factory=list)


def _apply_segment_fallback(
    seg: dict,
    recording_id: str,
    *,
    reason: str,
    detail: str,
    fb_words: list[tuple[float, float, str]],
    fallback_entries: list[dict],
) -> None:
    start = float(seg["start"])
    end = float(seg["end"])
    fb_words.append((start, end, "speech"))
    fallback_entries.append(
        segment_fallback_log_entry(
            seg,
            recording_id,
            reason=reason,
            detail=detail,
        )
    )
    logger.warning(
        "%s segment %s: MFA failed (%s); using manifest speech [%.3f, %.3f]",
        recording_id,
        seg.get("segment_index"),
        reason,
        start,
        end,
    )


def _segment_miss_or_fallback(
    seg: dict,
    recording_id: str,
    *,
    reason: str,
    detail: str,
    fb_words: list[tuple[float, float, str]],
    fallback_entries: list[dict],
    use_fallback: bool,
) -> None:
    if use_fallback:
        _apply_segment_fallback(
            seg,
            recording_id,
            reason=reason,
            detail=detail,
            fb_words=fb_words,
            fallback_entries=fallback_entries,
        )


def align_recording(
    recording_id: str,
    segments: list[dict],
    *,
    mfa_dict: Path,
    mfa_acoustic: str,
    temp_parent: Path,
    num_jobs: int,
    fallback_log: Path,
    segment_padding: float,
    fallback_log_lock: threading.Lock | None = None,
    worker_mfa_root: Path | None = None,
    worker_acoustic: str | None = None,
    keep_temp: bool = False,
    use_fallback: bool = True,
) -> RecordingAlignResult:
    temp_parent.mkdir(parents=True, exist_ok=True)
    try:
        if keep_temp:
            temp_root = temp_parent / f"align_{recording_id}"
            if temp_root.exists():
                shutil.rmtree(temp_root)
            temp_root.mkdir(parents=True, exist_ok=True)
            return _align_recording_impl(
                recording_id,
                segments,
                mfa_dict=mfa_dict,
                mfa_acoustic=mfa_acoustic,
                temp_root=temp_root,
                num_jobs=num_jobs,
                fallback_log=fallback_log,
                segment_padding=segment_padding,
                fallback_log_lock=fallback_log_lock,
                worker_mfa_root=worker_mfa_root,
                worker_acoustic=worker_acoustic,
                cleanup_temp=False,
                use_fallback=use_fallback,
            )
        with tempfile.TemporaryDirectory(
            prefix=f"mfa_{recording_id}_",
            dir=temp_parent,
        ) as td:
            return _align_recording_impl(
                recording_id,
                segments,
                mfa_dict=mfa_dict,
                mfa_acoustic=mfa_acoustic,
                temp_root=Path(td),
                num_jobs=num_jobs,
                fallback_log=fallback_log,
                segment_padding=segment_padding,
                fallback_log_lock=fallback_log_lock,
                worker_mfa_root=worker_mfa_root,
                worker_acoustic=worker_acoustic,
                cleanup_temp=False,
                use_fallback=use_fallback,
            )
    except Exception as exc:
        log_exception(f"MFA alignment failed for {recording_id}", exc)
        return RecordingAlignResult(ok=False)


def _align_recording_impl(
    recording_id: str,
    segments: list[dict],
    *,
    mfa_dict: Path,
    mfa_acoustic: str,
    temp_root: Path,
    num_jobs: int,
    fallback_log: Path,
    segment_padding: float,
    fallback_log_lock: threading.Lock | None = None,
    worker_mfa_root: Path | None = None,
    worker_acoustic: str | None = None,
    cleanup_temp: bool = True,
    use_fallback: bool = True,
) -> RecordingAlignResult:
    audio_path = Path(segments[0]["audio_filepath_16k"])
    speaker_id = segments[0]["speaker_id"]
    if not audio_path.is_file():
        logger.warning("%s: missing 16k audio %s", recording_id, audio_path)
        return RecordingAlignResult(ok=False)

    usable = [s for s in segments if (s.get("text_norm") or "").strip()]
    if not usable:
        logger.warning("%s: no segments with normalized text", recording_id)
        return RecordingAlignResult(ok=False)

    try:
        audio_duration = ffprobe_duration(audio_path)
    except RuntimeError:
        audio_duration = max(float(s["end"]) for s in usable) + 0.05

    corpus_name = f"corpus_{recording_id}"
    corpus_dir = temp_root / corpus_name / speaker_id
    aligned_dir = temp_root / "aligned"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)

    merged_words: list[tuple[float, float, str]] = []
    fb_words: list[tuple[float, float, str]] = []
    fallback_entries: list[dict] = []
    mfa_segments = 0

    seg_meta: list[tuple[dict, Path, float]] = []
    for seg in usable:
        seg_idx = int(seg["segment_index"])
        seg_wav = corpus_dir / f"seg_{seg_idx:05d}.wav"
        seg_txt = corpus_dir / f"seg_{seg_idx:05d}.txt"
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        try:
            extract_start = extract_segment_wav(
                audio_path,
                seg_wav,
                seg_start,
                seg_end,
                padding=segment_padding,
                max_duration=audio_duration,
            )
            if extract_start is None:
                _segment_miss_or_fallback(
                    seg,
                    recording_id,
                    reason="segment_export_failed",
                    detail="ffmpeg extract failed",
                    fb_words=fb_words,
                    fallback_entries=fallback_entries,
                    use_fallback=use_fallback,
                )
                continue
            seg_txt.write_text(seg["text_norm"].strip(), encoding="utf-8")
        except OSError as exc:
            log_exception(f"{recording_id} segment {seg_idx} export", exc)
            _segment_miss_or_fallback(
                seg,
                recording_id,
                reason="segment_export_failed",
                detail=str(exc),
                fb_words=fb_words,
                fallback_entries=fallback_entries,
                use_fallback=use_fallback,
            )
            continue
        seg_meta.append((seg, seg_wav, extract_start))

    mfa_failed_globally = False
    if seg_meta:
        mfa_root = worker_mfa_root or (temp_root / "mfa_root")
        if worker_mfa_root is None:
            mfa_root.mkdir(parents=True, exist_ok=True)
        acoustic_arg = worker_acoustic or resolve_mfa_acoustic_model(mfa_acoustic)
        align_cmd = [
            "mfa",
            "align",
            str(corpus_dir.parent),
            str(mfa_dict),
            acoustic_arg,
            str(aligned_dir),
        ]
        align_cmd.append("--clean" if worker_mfa_root is None else "--no_clean")
        align_cmd.extend(
            [
                "--use_mp",
                "-j",
                str(num_jobs),
                "--beam",
                "100",
                "--retry_beam",
                "400",
                "--output_format",
                "long_textgrid",
                "--uses_speaker_adaptation",
                "false",
                "-t",
                str(mfa_root),
            ]
        )
        logger.info("%s: running MFA on %d segments", recording_id, len(seg_meta))
        mfa_env = os.environ.copy()
        mfa_env["TMPDIR"] = str(temp_root.parent)
        mfa_env["MFA_ROOT_DIR"] = str(mfa_root)
        try:
            result = subprocess.run(align_cmd, capture_output=True, text=True, env=mfa_env)
        except OSError as exc:
            logger.error("%s: mfa align failed to start: %s", recording_id, exc)
            mfa_failed_globally = True
            detail = str(exc)
        else:
            detail = result.stderr[-1200:] if result.returncode != 0 else ""
            if result.returncode != 0:
                logger.error(
                    "%s: mfa align failed (exit %d): %s",
                    recording_id,
                    result.returncode,
                    detail,
                )
                mfa_failed_globally = True

        for seg, seg_wav, extract_start in seg_meta:
            if mfa_failed_globally:
                _segment_miss_or_fallback(
                    seg,
                    recording_id,
                    reason="mfa_align_failed",
                    detail=detail,
                    fb_words=fb_words,
                    fallback_entries=fallback_entries,
                    use_fallback=use_fallback,
                )
                continue

            tg_path = aligned_dir / speaker_id / f"{seg_wav.stem}.TextGrid"
            seg_end = float(seg["end"])
            extract_end = min(audio_duration, seg_end + segment_padding)
            words = safe_parse_textgrid_words(tg_path) if tg_path.is_file() else []
            mapped_words = map_segment_words_to_recording(
                words,
                extract_start=extract_start,
                extract_end=extract_end,
            )
            if not mapped_words:
                reason = "missing_textgrid" if not tg_path.is_file() else "empty_alignment"
                _segment_miss_or_fallback(
                    seg,
                    recording_id,
                    reason=reason,
                    detail=tg_path.name,
                    fb_words=fb_words,
                    fallback_entries=fallback_entries,
                    use_fallback=use_fallback,
                )
                continue

            merged_words.extend(mapped_words)
            mfa_segments += 1

    if not merged_words and not fb_words:
        logger.warning("%s: no segment output produced", recording_id)
        return RecordingAlignResult(ok=False)

    merged_words.sort(key=lambda x: x[0])
    fb_words.sort(key=lambda x: x[0])

    if use_fallback:
        for entry in fallback_entries:
            try:
                append_jsonl(fallback_log, entry, lock=fallback_log_lock)
            except PipelineError as exc:
                log_exception(f"cannot write MFA fallback log for {recording_id}", exc)

    if cleanup_temp and temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    if worker_mfa_root is not None:
        stale_db = worker_mfa_root / corpus_name
        if stale_db.exists():
            shutil.rmtree(stale_db, ignore_errors=True)

    logger.info(
        "%s: aligned %d MFA words, %d fallback segments",
        recording_id,
        len(merged_words),
        len(fallback_entries),
    )
    return RecordingAlignResult(
        ok=True,
        mfa_segments=mfa_segments,
        fallback_segments=len(fallback_entries),
        fallback_entries=fallback_entries,
        merged_words=merged_words,
        fb_words=fb_words,
        audio_duration=audio_duration,
    )


def align_session(
    session_id: str,
    segments: list[dict],
    *,
    mfa_dict: Path,
    mfa_acoustic: str,
    textgrid_dir: Path,
    temp_parent: Path,
    num_jobs: int,
    fallback_log: Path,
    segment_padding: float,
    fallback_log_lock: threading.Lock | None = None,
    worker_mfa_root: Path | None = None,
    worker_acoustic: str | None = None,
    keep_temp: bool = False,
    use_fallback: bool = True,
) -> SessionAlignResult:
    by_recording = group_segments_by_recording(segments)
    session_merged: list[tuple[float, float, str, str]] = []
    session_fb: list[tuple[float, float, str, str]] = []
    recording_rows: list[dict] = []
    session_duration = 0.0
    mfa_segments = 0
    fallback_segments = 0

    for rec_id, rec_segments in sorted(by_recording.items()):
        result = align_recording(
            rec_id,
            rec_segments,
            mfa_dict=mfa_dict,
            mfa_acoustic=mfa_acoustic,
            temp_parent=temp_parent,
            num_jobs=num_jobs,
            fallback_log=fallback_log,
            segment_padding=segment_padding,
            fallback_log_lock=fallback_log_lock,
            worker_mfa_root=worker_mfa_root,
            worker_acoustic=worker_acoustic,
            keep_temp=keep_temp,
            use_fallback=use_fallback,
        )
        if not result.ok:
            logger.warning("%s: speaker recording %s failed", session_id, rec_id)
            continue

        speaker_id = rec_segments[0]["speaker_id"]
        for start, end, word in result.merged_words:
            session_merged.append((start, end, word, speaker_id))
        for start, end, word in result.fb_words:
            session_fb.append((start, end, word, speaker_id))
        session_duration = max(session_duration, result.audio_duration)
        mfa_segments += result.mfa_segments
        fallback_segments += result.fallback_segments
        recording_rows.append(
            {
                "recording_id": rec_id,
                "speaker_id": speaker_id,
                "session_id": session_id,
                "audio_filepath_16k": rec_segments[0]["audio_filepath_16k"],
                "audio_duration": result.audio_duration,
                "merged_words": words_to_json(result.merged_words),
                "fb_words": words_to_json(result.fb_words),
            }
        )

    if not session_merged and not session_fb:
        logger.warning("%s: no session alignment output", session_id)
        return SessionAlignResult(ok=False)

    session_merged.sort(key=lambda x: x[0])
    session_fb.sort(key=lambda x: x[0])
    max_seg_end = max((end for _, end, _, _ in session_merged + session_fb), default=0.0)
    xmax = max(session_duration, max_seg_end) + 0.01

    fastmss_words = [(s, e, w) for s, e, w, _ in session_merged]
    ordinary_words = sorted(
        [(s, e, w) for s, e, w, _ in session_merged] + [(s, e, w) for s, e, w, _ in session_fb],
        key=lambda x: x[0],
    )
    write_textgrid(
        fastmss_words,
        session_textgrid_path(textgrid_dir, session_id, variant="fastmss"),
        xmin=0.0,
        xmax=xmax,
    )
    write_textgrid(
        ordinary_words,
        session_textgrid_path(textgrid_dir, session_id, variant="ordinary"),
        xmin=0.0,
        xmax=xmax,
    )

    logger.info(
        "%s: session TextGrids (%d MFA words, %d fallback, %d speakers)",
        session_id,
        len(session_merged),
        len(session_fb),
        len(recording_rows),
    )
    return SessionAlignResult(
        ok=True,
        mfa_segments=mfa_segments,
        fallback_segments=fallback_segments,
        merged_words=session_merged,
        fb_words=session_fb,
        audio_duration=xmax,
        recordings=recording_rows,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests-dir", type=Path, required=True)
    ap.add_argument("--mfa-dict", type=Path, required=True)
    ap.add_argument("--mfa-acoustic", default="english_us_arpa")
    ap.add_argument(
        "--textgrid-dir",
        type=Path,
        required=True,
        help="Per-session TextGrid output ({session_id}.TextGrid and {session_id}_fastmss.TextGrid)",
    )
    ap.add_argument(
        "--alignments-jsonl",
        type=Path,
        default=None,
        help="Compact per-session alignment cache (default: <workdir>/alignments.jsonl)",
    )
    ap.add_argument("--mfa-temp-dir", type=Path, required=True)
    ap.add_argument(
        "--mfa-workers-dir",
        type=Path,
        default=None,
        help="Per-worker MFA roots with copied models (default: <mfa-temp-dir>/workers)",
    )
    ap.add_argument(
        "--run-rttm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write per-recording RTTM files (legacy)",
    )
    ap.add_argument("--rttm-dir", type=Path, default=None, help="Shared RTTM output dir for worker stage 3")
    ap.add_argument(
        "--rttm-merge-gap",
        type=float,
        default=0.2,
        help="RTTM merge gap when --run-rttm is enabled",
    )
    ap.add_argument(
        "--mfa-fallback-log",
        type=Path,
        default=None,
        help="JSONL log for segments where MFA failed and manifest boundaries were used",
    )
    ap.add_argument("--num-jobs", type=int, default=4, help="MFA parallel jobs per speaker recording")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel sessions (each worker uses its own MFA root)",
    )
    ap.add_argument(
        "--segment-padding",
        type=float,
        default=0.5,
        help="Seconds of audio context to include before/after each manifest segment for MFA",
    )
    ap.add_argument("--recording", action="append", default=[], help="Optional recording_id filter")
    ap.add_argument("--session", action="append", default=[], help="Optional session_id filter")
    ap.add_argument("--work-dir", type=Path, default=None, help="Work dir for .done marker")
    ap.add_argument("--stage-done-name", default=None, help="Stage name for .done marker")
    ap.add_argument("--force", action="store_true", help="Re-run even if alignment already cached")
    ap.add_argument("--keep-temp", action="store_true", help="Keep per-speaker MFA temp dirs")
    args = ap.parse_args()

    if maybe_skip_done_stage(args.work_dir, args.stage_done_name, force=args.force):
        return 0

    manifests_dir = args.manifests_dir.resolve()
    mfa_dict = args.mfa_dict.resolve()
    textgrid_dir = args.textgrid_dir.resolve()
    alignments_jsonl = (
        args.alignments_jsonl.resolve()
        if args.alignments_jsonl
        else (manifests_dir.parent / "alignments.jsonl")
    )
    mfa_temp_dir = args.mfa_temp_dir.resolve()
    fallback_log = (
        args.mfa_fallback_log.resolve()
        if args.mfa_fallback_log
        else (manifests_dir.parent / "logs" / "mfa_segment_fallback.jsonl")
    )

    if not mfa_dict.is_file():
        raise PipelineError(f"MFA dictionary not found: {mfa_dict}")

    rows, manifest_errors = load_norm_manifest_rows(manifests_dir, sessions=args.session or None)
    grouped = group_segments_by_session(rows)

    if args.session:
        wanted_sessions = set(args.session)
        grouped = {k: v for k, v in grouped.items() if k in wanted_sessions}

    if args.recording:
        wanted_recs = set(args.recording)
        grouped = {
            session_id: [s for s in segs if s["recording_id"] in wanted_recs]
            for session_id, segs in grouped.items()
        }
        grouped = {k: v for k, v in grouped.items() if v}

    if not grouped:
        raise PipelineError("No sessions to align")

    textgrid_dir.mkdir(parents=True, exist_ok=True)
    mfa_temp_dir.mkdir(parents=True, exist_ok=True)
    alignments_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fallback_log.parent.mkdir(parents=True, exist_ok=True)
    if args.force:
        if fallback_log.is_file():
            fallback_log.unlink()
        if alignments_jsonl.is_file():
            alignments_jsonl.unlink()

    workers = max(1, args.workers)
    mfa_workers_dir = (
        args.mfa_workers_dir.resolve()
        if args.mfa_workers_dir
        else (mfa_temp_dir / "workers")
    )
    rttm_dir = (
        args.rttm_dir.resolve()
        if args.rttm_dir
        else (manifests_dir.parent / "rttm")
    )

    done_ids = load_alignment_ids(alignments_jsonl) if not args.force else set()
    items = sorted(grouped.items())
    to_process = [(sid, segs) for sid, segs in items if sid not in done_ids]
    skip = len(items) - len(to_process)

    if not to_process and skip == len(items):
        logger.info("Stage 2 done: all %d session alignments already cached", len(items))
        return finish_stage(args.work_dir, args.stage_done_name, 0)

    ok, fail = _run_worker_subprocesses(
        to_process,
        workers=workers,
        mfa_workers_dir=mfa_workers_dir,
        manifests_dir=manifests_dir,
        mfa_dict=mfa_dict,
        mfa_acoustic=args.mfa_acoustic,
        textgrid_dir=textgrid_dir,
        alignments_jsonl=alignments_jsonl,
        mfa_temp_dir=mfa_temp_dir,
        fallback_log=fallback_log,
        num_jobs=args.num_jobs,
        segment_padding=args.segment_padding,
        keep_temp=args.keep_temp,
        force=args.force,
        run_rttm=args.run_rttm,
        rttm_dir=rttm_dir,
        rttm_merge_gap=args.rttm_merge_gap,
    )

    logger.info(
        "Stage 2 done: ok=%d fail=%d skip=%d total=%d manifest_errors=%d "
        "alignments=%s textgrids=%s fallback_log=%s workers=%d",
        ok,
        fail,
        skip,
        len(grouped),
        manifest_errors,
        alignments_jsonl,
        textgrid_dir,
        fallback_log,
        workers,
    )
    exit_code = 1 if (fail or manifest_errors) else 0
    return finish_stage(args.work_dir, args.stage_done_name, exit_code)


def _run_worker_subprocesses(
    items: list[tuple[str, list[dict]]],
    *,
    workers: int,
    mfa_workers_dir: Path,
    manifests_dir: Path,
    mfa_dict: Path,
    mfa_acoustic: str,
    textgrid_dir: Path,
    alignments_jsonl: Path,
    mfa_temp_dir: Path,
    fallback_log: Path,
    num_jobs: int,
    segment_padding: float,
    keep_temp: bool,
    force: bool,
    run_rttm: bool,
    rttm_dir: Path,
    rttm_merge_gap: float,
) -> tuple[int, int]:
    mfa_workers_dir.mkdir(parents=True, exist_ok=True)
    source_root = mfa_models_root()
    shards = partition_list(items, workers)
    worker_script = Path(__file__).with_name("stage2_mfa_worker.py")
    procs: list[subprocess.Popen] = []

    for worker_id, shard in enumerate(shards):
        worker_dir = mfa_workers_dir / f"worker_{worker_id:02d}"
        setup_mfa_worker_root(
            worker_dir,
            mfa_dict=mfa_dict,
            mfa_acoustic=mfa_acoustic,
            source_mfa_root=source_root,
            force=force,
        )
        shard_path = worker_dir / "sessions.json"
        shard_path.write_text(
            json.dumps([session_id for session_id, _ in shard]),
            encoding="utf-8",
        )
        cmd = [
            sys.executable,
            str(worker_script),
            "--worker-id",
            str(worker_id),
            "--worker-dir",
            str(worker_dir),
            "--sessions-file",
            str(shard_path),
            "--manifests-dir",
            str(manifests_dir),
            "--mfa-dict",
            str(mfa_dict),
            "--mfa-acoustic",
            mfa_acoustic,
            "--textgrid-dir",
            str(textgrid_dir),
            "--alignments-jsonl",
            str(alignments_jsonl),
            "--mfa-temp-dir",
            str(mfa_temp_dir),
            "--mfa-fallback-log",
            str(fallback_log),
            "--num-jobs",
            str(num_jobs),
            "--segment-padding",
            str(segment_padding),
            "--rttm-merge-gap",
            str(rttm_merge_gap),
        ]
        if keep_temp:
            cmd.append("--keep-temp")
        if run_rttm:
            cmd.extend(["--rttm-dir", str(rttm_dir)])
        logger.info("worker %02d: %d sessions", worker_id, len(shard))
        procs.append(subprocess.Popen(cmd))

    for proc in procs:
        rc = proc.wait()
        if rc != 0:
            logger.error("worker subprocess failed (exit %d)", rc)

    ok = fail = 0
    for worker_id in range(len(shards)):
        worker_dir = mfa_workers_dir / f"worker_{worker_id:02d}"
        result_path = worker_dir / "result.json"
        if not result_path.is_file():
            logger.error("worker %02d: missing result.json", worker_id)
            fail += len(shards[worker_id])
            continue
        data = json.loads(result_path.read_text(encoding="utf-8"))
        ok += int(data.get("ok", 0))
        fail += int(data.get("fail", 0))

    return ok, fail


if __name__ == "__main__":
    run_main(main)
