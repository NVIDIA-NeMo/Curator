#!/usr/bin/env python3
"""Stage 5: merge per-recording RTTMs into per-session RTTM files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from david_ai_common import (
    PipelineError,
    finish_stage,
    group_recordings_by_session,
    load_norm_manifest_rows,
    log_exception,
    maybe_skip_done_stage,
    merge_session_rttm,
    run_main,
)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests-dir", type=Path, required=True)
    ap.add_argument("--rttm-dir", type=Path, required=True)
    ap.add_argument("--rttm-session-dir", type=Path, required=True)
    ap.add_argument("--session", action="append", default=[], help="Optional session_id filter")
    ap.add_argument("--work-dir", type=Path, default=None, help="Work dir for .done marker")
    ap.add_argument("--stage-done-name", default=None, help="Stage name for .done marker")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if maybe_skip_done_stage(args.work_dir, args.stage_done_name, force=args.force):
        return 0

    manifests_dir = args.manifests_dir.resolve()
    rttm_dir = args.rttm_dir.resolve()
    rttm_session_dir = args.rttm_session_dir.resolve()

    rows, manifest_errors = load_norm_manifest_rows(manifests_dir, sessions=args.session or None)
    by_session = group_recordings_by_session(rows)
    if not by_session:
        raise PipelineError("No sessions found in normalized manifests")

    rttm_session_dir.mkdir(parents=True, exist_ok=True)
    ok = skip = fail = 0
    for session_id, entries in sorted(by_session.items()):
        session_rttm = rttm_session_dir / f"{session_id}.rttm"
        if session_rttm.exists() and not args.force:
            skip += 1
            continue
        try:
            rttm_paths = [rttm_dir / f"{e['recording_id']}.rttm" for e in entries]
            n_lines = merge_session_rttm(rttm_paths, session_id, session_rttm)
            logger.info("%s: merged session RTTM (%d lines)", session_id, n_lines)
            ok += 1
        except Exception as exc:
            fail += 1
            log_exception(f"session RTTM merge failed for {session_id}", exc)

    logger.info(
        "Stage 5 done: ok=%d skip=%d fail=%d sessions=%d manifest_errors=%d",
        ok,
        skip,
        fail,
        len(by_session),
        manifest_errors,
    )
    exit_code = 1 if (fail or manifest_errors) else 0
    return finish_stage(args.work_dir, args.stage_done_name, exit_code)


if __name__ == "__main__":
    run_main(main)
