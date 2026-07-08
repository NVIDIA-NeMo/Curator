#!/usr/bin/env python3
"""Stage 2 worker: MFA align session shard; write session TextGrids + alignments JSONL."""

from __future__ import annotations

import argparse
import json
import logging
import threading
from pathlib import Path

from david_ai_common import (
    append_alignment_record,
    build_rttm_lines_from_words,
    group_segments_by_session,
    load_norm_manifest_rows,
    log_exception,
    run_main,
    session_alignment_record,
    setup_mfa_worker_root,
    words_from_json,
    write_rttm,
)
from stage2_mfa_align_textgrids import align_session

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--worker-dir", type=Path, required=True)
    ap.add_argument("--sessions-file", type=Path, required=True)
    ap.add_argument("--manifests-dir", type=Path, required=True)
    ap.add_argument("--mfa-dict", type=Path, required=True)
    ap.add_argument("--mfa-acoustic", default="english_us_arpa")
    ap.add_argument("--textgrid-dir", type=Path, required=True)
    ap.add_argument("--alignments-jsonl", type=Path, required=True)
    ap.add_argument("--mfa-temp-dir", type=Path, required=True)
    ap.add_argument("--mfa-fallback-log", type=Path, required=True)
    ap.add_argument("--num-jobs", type=int, default=4)
    ap.add_argument("--segment-padding", type=float, default=0.5)
    ap.add_argument("--keep-temp", action="store_true")
    ap.add_argument("--rttm-dir", type=Path, default=None)
    ap.add_argument("--rttm-merge-gap", type=float, default=0.2)
    args = ap.parse_args()

    worker_dir = args.worker_dir.resolve()
    session_ids = json.loads(args.sessions_file.read_text(encoding="utf-8"))
    if not session_ids:
        logger.info("worker %02d: empty shard", args.worker_id)
        return 0

    mfa_dict = args.mfa_dict.resolve()
    worker_mfa_root, local_dict, acoustic_arg = setup_mfa_worker_root(
        worker_dir,
        mfa_dict=mfa_dict,
        mfa_acoustic=args.mfa_acoustic,
    )

    rows, _ = load_norm_manifest_rows(args.manifests_dir.resolve())
    grouped = group_segments_by_session(rows)

    textgrid_dir = args.textgrid_dir.resolve()
    alignments_jsonl = args.alignments_jsonl.resolve()
    mfa_temp_dir = args.mfa_temp_dir.resolve()
    fallback_log = args.mfa_fallback_log.resolve()
    temp_parent = mfa_temp_dir / f"worker_{args.worker_id:02d}"
    io_lock = threading.Lock()

    ok = fail = 0
    for session_id in session_ids:
        segments = grouped.get(session_id)
        if not segments:
            logger.warning("worker %02d: unknown session %s", args.worker_id, session_id)
            fail += 1
            continue

        result = align_session(
            session_id,
            segments,
            mfa_dict=local_dict,
            mfa_acoustic=args.mfa_acoustic,
            textgrid_dir=textgrid_dir,
            temp_parent=temp_parent,
            num_jobs=args.num_jobs,
            fallback_log=fallback_log,
            segment_padding=args.segment_padding,
            fallback_log_lock=io_lock,
            worker_mfa_root=worker_mfa_root,
            worker_acoustic=acoustic_arg,
            keep_temp=args.keep_temp,
        )
        if not result.ok:
            fail += 1
            continue

        append_alignment_record(
            alignments_jsonl,
            session_alignment_record(
                session_id,
                merged_words=result.merged_words,
                fb_words=result.fb_words,
                audio_duration=result.audio_duration,
                recordings=result.recordings,
            ),
            lock=io_lock,
        )
        ok += 1

        if args.rttm_dir is not None:
            for rec_row in result.recordings:
                rec_id = rec_row["recording_id"]
                speaker_id = rec_row["speaker_id"]
                rttm_path = args.rttm_dir.resolve() / f"{rec_id}.rttm"
                try:
                    lines = build_rttm_lines_from_words(
                        rec_id,
                        speaker_id,
                        words_from_json(rec_row["merged_words"]),
                        words_from_json(rec_row["fb_words"]),
                        merge_gap=args.rttm_merge_gap,
                    )
                    write_rttm(rttm_path, lines)
                except Exception as exc:
                    log_exception(f"worker {args.worker_id:02d} RTTM failed for {rec_id}", exc)

    result_path = worker_dir / "result.json"
    result_path.write_text(
        json.dumps({"worker_id": args.worker_id, "ok": ok, "fail": fail}),
        encoding="utf-8",
    )
    logger.info("worker %02d done: ok=%d fail=%d", args.worker_id, ok, fail)
    return 1 if fail else 0


if __name__ == "__main__":
    run_main(main)
