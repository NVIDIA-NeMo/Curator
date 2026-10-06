"""Ephemeral MFA alignment used only by the on-the-fly RAM E2E pipeline."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from david_ai_common import (
    PipelineError,
    append_jsonl,
    append_mfa_g2p_args,
    extract_segment_wav,
    ffprobe_duration,
    group_segments_by_recording,
    log_exception,
    map_segment_words_to_recording,
    mfa_subprocess_env,
    resolve_mfa_acoustic_model,
    resolve_mfa_g2p_model,
    run_thread_pool,
    safe_parse_textgrid_words,
    segment_fallback_log_entry,
    session_textgrid_path,
    words_to_json,
    write_textgrid,
)

if TYPE_CHECKING:
    import threading

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Guard against a wedged `mfa align` (e.g. SQLite/pynini lock contention seen when
# many run in a worker pool) blocking its caller forever and hanging the whole shard,
# mirroring the FFMPEG_TIMEOUT_S protection already used for ffmpeg subprocesses.
MFA_ALIGN_TIMEOUT_S = 3600


def _segment_extract_workers(num_segments: int) -> int:
    raw = os.environ.get("SEG_EXTRACT_WORKERS", "").strip()
    if raw:
        try:
            return max(1, min(int(raw), num_segments))
        except ValueError:
            pass
    return max(1, min(num_segments, 8))


def _export_segment_for_mfa(
    seg: dict,
    *,
    audio_path: Path,
    corpus_dir: Path,
    segment_padding: float,
    audio_duration: float,
) -> tuple[dict, Path, float] | None:
    seg_idx = int(seg["segment_index"])
    seg_wav = corpus_dir / f"seg_{seg_idx:05d}.wav"
    seg_txt = corpus_dir / f"seg_{seg_idx:05d}.txt"
    seg_start = float(seg["start"])
    seg_end = float(seg["end"])
    extract_start = extract_segment_wav(
        audio_path,
        seg_wav,
        seg_start,
        seg_end,
        padding=segment_padding,
        max_duration=audio_duration,
    )
    if extract_start is None:
        return None
    seg_txt.write_text(seg["text_norm"].strip(), encoding="utf-8")
    return seg, seg_wav, extract_start


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
    worker_g2p: str | None = None,
    mfa_g2p: str | None = None,
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
                worker_g2p=worker_g2p,
                mfa_g2p=mfa_g2p,
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
                worker_g2p=worker_g2p,
                mfa_g2p=mfa_g2p,
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
    worker_g2p: str | None = None,
    mfa_g2p: str | None = None,
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

    def _export_one(seg: dict) -> tuple[dict, Path, float] | None:
        try:
            return _export_segment_for_mfa(
                seg,
                audio_path=audio_path,
                corpus_dir=corpus_dir,
                segment_padding=segment_padding,
                audio_duration=audio_duration,
            )
        except OSError as exc:
            log_exception(f"{recording_id} segment {seg.get('segment_index')} export", exc)
            return None

    extract_results = run_thread_pool(
        usable,
        _export_one,
        workers=_segment_extract_workers(len(usable)),
    )
    for seg, exported in zip(usable, extract_results, strict=True):
        if exported is None:
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
        seg_meta.append(exported)

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
        g2p_arg = worker_g2p
        if g2p_arg is None and mfa_g2p:
            try:
                g2p_arg = str(resolve_mfa_g2p_model(mfa_g2p))
            except FileNotFoundError:
                logger.warning("%s: MFA G2P model not found for %r", recording_id, mfa_g2p)
        append_mfa_g2p_args(align_cmd, g2p_path=g2p_arg)
        logger.info("%s: running MFA on %d segments", recording_id, len(seg_meta))
        mfa_env = mfa_subprocess_env(temp_root=temp_root, mfa_root=mfa_root)
        try:
            result = subprocess.run(
                align_cmd,
                capture_output=True,
                text=True,
                env=mfa_env,
                timeout=MFA_ALIGN_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            logger.exception("%s: mfa align timed out after %ds", recording_id, MFA_ALIGN_TIMEOUT_S)
            mfa_failed_globally = True
            detail = f"mfa align timed out after {MFA_ALIGN_TIMEOUT_S}s: {exc}"
        except OSError as exc:
            logger.exception("%s: mfa align failed to start", recording_id)
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
    worker_g2p: str | None = None,
    mfa_g2p: str | None = None,
    keep_temp: bool = False,
    use_fallback: bool = True,
    write_textgrids: bool = True,
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
            worker_g2p=worker_g2p,
            mfa_g2p=mfa_g2p,
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
    if write_textgrids:
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
    else:
        logger.info(
            "%s: MFA alignment (%d words, %d fallback, %d speakers; TextGrids skipped)",
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
