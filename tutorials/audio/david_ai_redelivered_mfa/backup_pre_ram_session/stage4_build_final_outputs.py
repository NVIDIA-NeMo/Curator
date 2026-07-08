#!/usr/bin/env python3
"""Stage 4: build Lhotse cutset + session RTTM from session alignments cache."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from david_ai_common import (
    PipelineError,
    alignment_items_from_words,
    build_session_rttm_lines_from_words,
    finish_stage,
    group_segments_by_recording,
    load_alignments_by_recording,
    load_alignments_by_session,
    load_norm_manifest_rows,
    log_exception,
    maybe_skip_done_stage,
    partition_list,
    run_main,
    run_thread_pool,
    tagged_words_from_json,
    words_from_json,
    write_rttm,
)
from lhotse import CutSet, MonoCut, Recording, RecordingSet, SupervisionSegment, SupervisionSet, fix_manifests

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def build_cut_from_alignment(recording_id: str, row: dict, segments: list[dict]) -> MonoCut | None:
    audio_path = Path(row["audio_filepath_16k"])
    speaker_id = row["speaker_id"]
    session_id = row.get("session_id") or segments[0]["session_id"]
    merged_words = words_from_json(row["merged_words"])

    if not audio_path.is_file():
        logger.warning("%s: missing audio %s", recording_id, audio_path)
        return None
    if not merged_words:
        logger.warning("%s: empty fastmss alignment", recording_id)
        return None

    words = alignment_items_from_words(merged_words)
    text_parts = [(s.get("text_norm") or "").strip() for s in segments]
    text = " ".join(t for t in text_parts if t)

    rec = Recording.from_file(audio_path, recording_id=recording_id)
    sup = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=rec.duration,
        channel=0,
        text=text,
        speaker=speaker_id,
        custom={"session_id": session_id},
    )
    sup = sup.with_alignment("word", words)
    return MonoCut(
        id=recording_id,
        start=0.0,
        duration=rec.duration,
        channel=0,
        recording=rec,
        supervisions=[sup],
    )


def _process_shard(
    session_ids: list[str],
    *,
    session_alignments: dict[str, dict],
    rec_alignments: dict[str, dict],
    grouped: dict[str, list[dict]],
    rttm_merge_gap: float,
) -> tuple[list[MonoCut], dict[str, list[str]]]:
    cuts: list[MonoCut] = []
    session_rttm_lines: dict[str, list[str]] = {}

    for session_id in session_ids:
        session_row = session_alignments.get(session_id)
        if session_row is not None:
            merged_words = tagged_words_from_json(session_row["merged_words"])
            fb_words = tagged_words_from_json(session_row["fb_words"])
            lines = build_session_rttm_lines_from_words(
                session_id,
                merged_words,
                fb_words,
                merge_gap=rttm_merge_gap,
            )
            if lines:
                session_rttm_lines[session_id] = lines

        if session_row and session_row.get("recordings"):
            rec_ids = [rec["recording_id"] for rec in session_row["recordings"]]
        else:
            rec_ids = sorted(
                rec_id
                for rec_id, segs in grouped.items()
                if segs and segs[0]["session_id"] == session_id
            )
        for rec_id in rec_ids:
            row = rec_alignments.get(rec_id)
            segments = grouped.get(rec_id)
            if row is None or not segments:
                continue
            cut = build_cut_from_alignment(rec_id, row, segments)
            if cut is not None:
                cuts.append(cut)

    return cuts, session_rttm_lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests-dir", type=Path, required=True)
    ap.add_argument("--alignments-jsonl", type=Path, required=True)
    ap.add_argument("--lhotse-dir", type=Path, required=True)
    ap.add_argument(
        "--rttm-mixed-dir",
        type=Path,
        required=True,
        help="Session RTTM for mixed audio ({session_id}.rttm next to mixed WAV)",
    )
    ap.add_argument("--prefix", default="david_ai")
    ap.add_argument(
        "--rttm-merge-gap",
        type=float,
        default=0.2,
        help="Merge neighboring RTTM intervals when pause between them is <= this many seconds",
    )
    ap.add_argument("--workers", type=int, default=2, help="Parallel workers for final output build")
    ap.add_argument("--session", action="append", default=[], help="Optional session_id filter")
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--stage-done-name", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if maybe_skip_done_stage(args.work_dir, args.stage_done_name, force=args.force):
        return 0

    manifests_dir = args.manifests_dir.resolve()
    alignments_jsonl = args.alignments_jsonl.resolve()
    lhotse_dir = args.lhotse_dir.resolve()
    rttm_mixed_dir = args.rttm_mixed_dir.resolve()
    prefix = args.prefix
    aligned_path = lhotse_dir / f"{prefix}_aligned_cuts.jsonl.gz"

    if aligned_path.exists() and not args.force:
        logger.info("Final outputs already exist: %s", aligned_path)
        return finish_stage(args.work_dir, args.stage_done_name, 0)

    if not alignments_jsonl.is_file():
        raise PipelineError(f"Missing alignments cache: {alignments_jsonl} (run stage 2 first)")

    rows, manifest_errors = load_norm_manifest_rows(manifests_dir, sessions=args.session or None)
    grouped = group_segments_by_recording(rows)
    session_alignments = load_alignments_by_session(alignments_jsonl)
    rec_alignments = load_alignments_by_recording(alignments_jsonl)
    if not grouped or not rec_alignments:
        raise PipelineError("No recordings or alignments found")

    session_ids = sorted(session_alignments.keys()) if session_alignments else sorted(
        {s["session_id"] for segs in grouped.values() for s in segs}
    )
    if args.session:
        wanted = set(args.session)
        session_ids = [sid for sid in session_ids if sid in wanted]

    workers = max(1, args.workers)
    shards = partition_list(session_ids, workers)

    def _run_shard(shard: list[str]) -> tuple[list[MonoCut], dict[str, list[str]]]:
        return _process_shard(
            shard,
            session_alignments=session_alignments,
            rec_alignments=rec_alignments,
            grouped=grouped,
            rttm_merge_gap=args.rttm_merge_gap,
        )

    shard_results = run_thread_pool(shards, _run_shard, workers=workers)

    cuts: list[MonoCut] = []
    session_rttm_lines: dict[str, list[str]] = {}
    for shard_cuts, shard_rttm in shard_results:
        cuts.extend(shard_cuts)
        session_rttm_lines.update(shard_rttm)

    if not cuts:
        raise PipelineError("No Lhotse cuts built from alignments cache")

    lhotse_dir.mkdir(parents=True, exist_ok=True)
    rttm_mixed_dir.mkdir(parents=True, exist_ok=True)

    try:
        cutset = CutSet.from_cuts(cuts)
        rec_set = RecordingSet.from_recordings([c.recording for c in cutset])
        sup_set = SupervisionSet.from_segments([s for c in cutset for s in c.supervisions])
        rec_set, sup_set = fix_manifests(rec_set, sup_set)

        rec_path = lhotse_dir / f"{prefix}_recordings.jsonl.gz"
        sup_path = lhotse_dir / f"{prefix}_supervisions.jsonl.gz"
        cuts_path = lhotse_dir / f"{prefix}_cuts.jsonl.gz"

        rec_set.to_file(rec_path)
        sup_set.to_file(sup_path)
        cutset.to_file(cuts_path)
        cutset.to_file(aligned_path)
    except Exception as exc:
        raise PipelineError(f"failed to write Lhotse manifests: {exc}") from exc

    rttm_ok = rttm_fail = 0
    for session_id in sorted(session_rttm_lines.keys()):
        session_path = rttm_mixed_dir / f"{session_id}.rttm"
        if session_path.exists() and not args.force:
            continue
        lines = session_rttm_lines[session_id]
        if not lines:
            logger.warning("%s: no RTTM lines from alignments", session_id)
            rttm_fail += 1
            continue
        try:
            write_rttm(session_path, lines)
            logger.info("%s: wrote mixed-audio RTTM (%d lines)", session_id, len(lines))
            rttm_ok += 1
        except Exception as exc:
            rttm_fail += 1
            log_exception(f"session RTTM write failed for {session_id}", exc)

    logger.info(
        "Stage 4 done: cuts=%d sessions_rttm=%d rttm_fail=%d manifest_errors=%d workers=%d",
        len(cutset),
        rttm_ok,
        rttm_fail,
        manifest_errors,
        workers,
    )
    exit_code = 1 if (rttm_fail or manifest_errors) else 0
    return finish_stage(args.work_dir, args.stage_done_name, exit_code)


if __name__ == "__main__":
    run_main(main)
