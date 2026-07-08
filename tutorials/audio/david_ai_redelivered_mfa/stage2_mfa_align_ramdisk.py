#!/usr/bin/env python3
"""Stage 2 (RAM disk): MFA align per segment with all scratch on tmpfs.

Same result as stage2_mfa_align_textgrids.py (per-session TextGrids + alignments
cache), but every intermediate the aligner needs — segment .wav/.txt, the MFA
corpus SQLite DB, per-worker MFA roots, and MFA's TextGrid output — is placed on
a RAM-backed tmpfs mount (default /dev/shm). Nothing is written to the working
disk, and the whole scratch tree is removed on exit even if a worker crashes.

MFA itself has no in-memory alignment API (it reads a corpus directory + DB from
disk), so "in memory" here means those files exist only in RAM and are never
persisted to the working disk.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path

from david_ai_common import (
    PipelineError,
    clear_all_alignment_done,
    finish_stage,
    group_segments_by_session,
    load_norm_manifest_rows,
    maybe_skip_done_stage,
    plan_sessions_needing_alignment,
    run_main,
)
from stage2_mfa_align_textgrids import _run_worker_subprocesses

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_tmpfs(path: Path) -> bool:
    """Best-effort check that *path* lives on a RAM-backed filesystem."""
    try:
        target = path.resolve()
        best_match = ""
        best_type = ""
        with open("/proc/mounts", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point, fs_type = parts[1], parts[2]
                if str(target).startswith(mount_point) and len(mount_point) >= len(best_match):
                    best_match = mount_point
                    best_type = fs_type
        return best_type in {"tmpfs", "ramfs"}
    except OSError:
        return False


def _pick_ram_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in (
        os.environ.get("XDG_RUNTIME_DIR"),
        "/dev/shm",
        "/run/shm",
    ):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    raise PipelineError("No RAM disk found; pass --ram-dir pointing to a tmpfs mount")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests-dir", type=Path, required=True)
    ap.add_argument("--mfa-dict", type=Path, required=True)
    ap.add_argument("--mfa-acoustic", default="english_us_arpa")
    ap.add_argument(
        "--textgrid-dir",
        type=Path,
        required=True,
        help="Per-session TextGrid output (persisted to disk)",
    )
    ap.add_argument(
        "--alignments-jsonl",
        type=Path,
        default=None,
        help="Per-session alignment cache (persisted; default: <workdir>/alignments.jsonl)",
    )
    ap.add_argument(
        "--ram-dir",
        type=Path,
        default=None,
        help="tmpfs mount for all MFA scratch (default: $XDG_RUNTIME_DIR or /dev/shm)",
    )
    ap.add_argument(
        "--allow-non-tmpfs",
        action="store_true",
        help="Proceed even if --ram-dir is not a tmpfs/ramfs mount (not recommended)",
    )
    ap.add_argument("--mfa-fallback-log", type=Path, default=None)
    ap.add_argument("--num-jobs", type=int, default=4, help="MFA parallel jobs per speaker recording")
    ap.add_argument("--workers", type=int, default=1, help="Parallel sessions")
    ap.add_argument(
        "--segment-padding",
        type=float,
        default=0.5,
        help="Seconds of audio context before/after each manifest segment for MFA",
    )
    ap.add_argument("--recording", action="append", default=[], help="Optional recording_id filter")
    ap.add_argument("--session", action="append", default=[], help="Optional session_id filter")
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--stage-done-name", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--run-rttm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write per-recording RTTM files (persisted)",
    )
    ap.add_argument("--rttm-dir", type=Path, default=None)
    ap.add_argument("--rttm-merge-gap", type=float, default=0.2)
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
    fallback_log = (
        args.mfa_fallback_log.resolve()
        if args.mfa_fallback_log
        else (manifests_dir.parent / "logs" / "mfa_segment_fallback.jsonl")
    )
    rttm_dir = args.rttm_dir.resolve() if args.rttm_dir else (manifests_dir.parent / "rttm")

    if not mfa_dict.is_file():
        raise PipelineError(f"MFA dictionary not found: {mfa_dict}")

    ram_root = _pick_ram_root(args.ram_dir).resolve()
    ram_root.mkdir(parents=True, exist_ok=True)
    if not _is_tmpfs(ram_root):
        msg = f"--ram-dir {ram_root} is not a tmpfs/ramfs mount"
        if not args.allow_non_tmpfs:
            raise PipelineError(f"{msg}; pass --allow-non-tmpfs to override")
        logger.warning("%s; proceeding because --allow-non-tmpfs was set", msg)

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
    alignments_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fallback_log.parent.mkdir(parents=True, exist_ok=True)
    if args.force:
        if fallback_log.is_file():
            fallback_log.unlink()
        if alignments_jsonl.is_file():
            alignments_jsonl.unlink()
        clear_all_alignment_done(textgrid_dir)

    to_process, skip = plan_sessions_needing_alignment(
        grouped,
        textgrid_dir=textgrid_dir,
        alignments_jsonl=alignments_jsonl,
        force=args.force,
    )

    if not to_process and skip == len(grouped):
        logger.info("Stage 2 (RAM) done: all %d session alignments already cached", len(grouped))
        return finish_stage(args.work_dir, args.stage_done_name, 0)

    workers = max(1, args.workers)

    # Everything below lives on the RAM disk and is destroyed on exit.
    scratch = Path(tempfile.mkdtemp(prefix="mfa_ramdisk_", dir=ram_root))
    mfa_temp_dir = scratch / "temp"
    mfa_workers_dir = scratch / "workers"
    mfa_temp_dir.mkdir(parents=True, exist_ok=True)
    mfa_workers_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup() -> None:
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
            logger.info("removed RAM scratch %s", scratch)

    atexit.register(_cleanup)

    def _on_signal(signum: int, _frame: object) -> None:
        _cleanup()
        # Restore default handling and re-raise so exit codes stay meaningful.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_signal)

    logger.info(
        "Stage 2 (RAM) START: %d sessions on tmpfs %s (scratch=%s workers=%d)",
        len(to_process),
        ram_root,
        scratch,
        workers,
    )

    try:
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
            keep_temp=False,
            force=args.force,
            run_rttm=args.run_rttm,
            rttm_dir=rttm_dir,
            rttm_merge_gap=args.rttm_merge_gap,
        )
    finally:
        _cleanup()

    logger.info(
        "Stage 2 (RAM) done: ok=%d fail=%d skip=%d total=%d manifest_errors=%d "
        "alignments=%s textgrids=%s workers=%d",
        ok,
        fail,
        skip,
        len(grouped),
        manifest_errors,
        alignments_jsonl,
        textgrid_dir,
        workers,
    )
    exit_code = 1 if (fail or manifest_errors) else 0
    if exit_code == 0:
        remaining, _ = plan_sessions_needing_alignment(
            grouped,
            textgrid_dir=textgrid_dir,
            alignments_jsonl=alignments_jsonl,
            force=False,
        )
        if remaining:
            logger.info(
                "Stage 2 (RAM) partial resume point: %d/%d sessions still need alignment "
                "(resubmit with STAGE=2 STAGE_END=7)",
                len(remaining),
                len(grouped),
            )
            return 1
    return finish_stage(args.work_dir, args.stage_done_name, exit_code)


if __name__ == "__main__":
    run_main(main)
