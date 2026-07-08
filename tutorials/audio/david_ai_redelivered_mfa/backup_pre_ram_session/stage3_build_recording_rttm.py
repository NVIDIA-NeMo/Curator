#!/usr/bin/env python3
"""Stage 3: per-recording RTTM from merged TextGrid + MFA fallback intervals."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from david_ai_common import (
    PipelineError,
    build_recording_rttm_lines,
    finish_stage,
    group_segments_by_recording,
    load_norm_manifest_rows,
    log_exception,
    maybe_skip_done_stage,
    run_main,
    run_thread_pool,
    write_rttm,
)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests-dir", type=Path, required=True)
    ap.add_argument("--textgrid-dir", type=Path, required=True)
    ap.add_argument("--rttm-dir", type=Path, required=True)
    ap.add_argument(
        "--mfa-fallback-log",
        type=Path,
        default=None,
        help="JSONL log with manifest-boundary fallbacks from stage 2",
    )
    ap.add_argument(
        "--rttm-merge-gap",
        type=float,
        default=0.2,
        help="Merge neighboring RTTM intervals when pause between them is <= this many seconds",
    )
    ap.add_argument("--session", action="append", default=[], help="Optional session_id filter")
    ap.add_argument("--work-dir", type=Path, default=None, help="Work dir for .done marker")
    ap.add_argument("--stage-done-name", default=None, help="Stage name for .done marker")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="Parallel RTTM workers")
    args = ap.parse_args()

    if maybe_skip_done_stage(args.work_dir, args.stage_done_name, force=args.force):
        return 0

    manifests_dir = args.manifests_dir.resolve()
    textgrid_dir = args.textgrid_dir.resolve()
    rttm_dir = args.rttm_dir.resolve()
    fallback_log = (
        args.mfa_fallback_log.resolve()
        if args.mfa_fallback_log
        else (manifests_dir.parent / "logs" / "mfa_segment_fallback.jsonl")
    )

    rows, manifest_errors = load_norm_manifest_rows(manifests_dir, sessions=args.session or None)
    grouped = group_segments_by_recording(rows)
    if not grouped:
        raise PipelineError("No recordings found in normalized manifests")

    rttm_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, args.workers)

    def _build_one(item: tuple[str, list[dict]]) -> str:
        rec_id, segments = item
        speaker_id = segments[0]["speaker_id"]
        tg_path = textgrid_dir / f"{rec_id}.TextGrid"
        rttm_path = rttm_dir / f"{rec_id}.rttm"

        if not tg_path.is_file():
            logger.warning("%s: missing TextGrid %s", rec_id, tg_path)
            return "fail"
        if rttm_path.exists() and not args.force:
            return "skip"
        try:
            lines = build_recording_rttm_lines(
                rec_id,
                speaker_id,
                tg_path,
                fallback_log=fallback_log,
                merge_gap=args.rttm_merge_gap,
            )
            write_rttm(rttm_path, lines)
            logger.info("%s: wrote %s (%d intervals)", rec_id, rttm_path.name, len(lines))
        except Exception as exc:
            log_exception(f"RTTM write failed for {rec_id}", exc)
            return "fail"
        return "ok"

    outcomes = run_thread_pool(sorted(grouped.items()), _build_one, workers=workers)
    ok = outcomes.count("ok")
    skip = outcomes.count("skip")
    fail = outcomes.count("fail")

    logger.info(
        "Stage 3 done: ok=%d skip=%d fail=%d total=%d manifest_errors=%d workers=%d",
        ok,
        skip,
        fail,
        len(grouped),
        manifest_errors,
        workers,
    )
    exit_code = 1 if (fail or manifest_errors) else 0
    return finish_stage(args.work_dir, args.stage_done_name, exit_code)


if __name__ == "__main__":
    run_main(main)
