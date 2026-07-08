#!/usr/bin/env python3
"""Stage 4: Lhotse aligned cutset from fastmss TextGrids and 16 kHz audio."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from david_ai_common import (
    PipelineError,
    alignment_items_from_textgrid,
    fastmss_textgrid_path,
    finish_stage,
    group_segments_by_recording,
    load_norm_manifest_rows,
    log_exception,
    maybe_skip_done_stage,
    recording_textgrid_path,
    run_main,
)
from lhotse import CutSet, MonoCut, Recording, RecordingSet, SupervisionSegment, SupervisionSet, fix_manifests

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def build_cut(
    recording_id: str,
    segments: list[dict],
    tg_path: Path,
    *,
    textgrid_dir: Path,
) -> MonoCut | None:
    try:
        return _build_cut_impl(recording_id, segments, tg_path, textgrid_dir=textgrid_dir)
    except Exception as exc:
        log_exception(f"failed to build cut for {recording_id}", exc)
        return None


def _build_cut_impl(
    recording_id: str,
    segments: list[dict],
    tg_path: Path,
    *,
    textgrid_dir: Path,
) -> MonoCut | None:
    audio_path = Path(segments[0]["audio_filepath_16k"])
    speaker_id = segments[0]["speaker_id"]
    session_id = segments[0]["session_id"]

    if not audio_path.is_file():
        logger.warning("%s: missing audio %s", recording_id, audio_path)
        return None
    if not tg_path.is_file():
        logger.warning("%s: missing fastmss TextGrid %s", recording_id, tg_path)
        return None

    words = alignment_items_from_textgrid(tg_path)
    if not words:
        logger.warning("%s: empty word alignment in %s", recording_id, tg_path)
        return None

    text_parts = [(s.get("text_norm") or "").strip() for s in segments]
    text = " ".join(t for t in text_parts if t)

    rec = Recording.from_file(audio_path, recording_id=recording_id)
    ordinary_path = recording_textgrid_path(textgrid_dir, recording_id, variant="ordinary")
    custom = {
        "session_id": session_id,
        "textgrid_path": str(ordinary_path.resolve()) if ordinary_path.is_file() else "",
        "fastmss_textgrid_path": str(tg_path.resolve()),
    }

    sup = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=rec.duration,
        channel=0,
        text=text,
        speaker=speaker_id,
        custom=custom,
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests-dir", type=Path, required=True)
    ap.add_argument("--textgrid-dir", type=Path, required=True)
    ap.add_argument("--lhotse-dir", type=Path, required=True)
    ap.add_argument("--prefix", default="david_ai")
    ap.add_argument("--session", action="append", default=[], help="Optional session_id filter")
    ap.add_argument("--work-dir", type=Path, default=None, help="Work dir for .done marker")
    ap.add_argument("--stage-done-name", default=None, help="Stage name for .done marker")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if maybe_skip_done_stage(args.work_dir, args.stage_done_name, force=args.force):
        return 0

    manifests_dir = args.manifests_dir.resolve()
    textgrid_dir = args.textgrid_dir.resolve()
    lhotse_dir = args.lhotse_dir.resolve()
    prefix = args.prefix
    aligned_path = lhotse_dir / f"{prefix}_aligned_cuts.jsonl.gz"

    if aligned_path.exists() and not args.force:
        logger.info("Lhotse cutset already exists: %s", aligned_path)
        return finish_stage(args.work_dir, args.stage_done_name, 0)

    rows, manifest_errors = load_norm_manifest_rows(manifests_dir, sessions=args.session or None)
    grouped = group_segments_by_recording(rows)
    if not grouped:
        raise PipelineError("No recordings found in normalized manifests")

    lhotse_dir.mkdir(parents=True, exist_ok=True)
    cuts = []
    cut_fail = 0
    for rec_id, segments in sorted(grouped.items()):
        tg_path = fastmss_textgrid_path(textgrid_dir, rec_id)
        cut = build_cut(rec_id, segments, tg_path, textgrid_dir=textgrid_dir)
        if cut is None:
            cut_fail += 1
            continue
        cuts.append(cut)

    if not cuts:
        raise PipelineError("No aligned cuts built (missing fastmss TextGrids or alignments?)")

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

    logger.info(
        "Stage 4 done: cuts=%d cut_fail=%d manifest_errors=%d output=%s",
        len(cutset),
        cut_fail,
        manifest_errors,
        aligned_path,
    )
    exit_code = 1 if (cut_fail or manifest_errors) else 0
    return finish_stage(args.work_dir, args.stage_done_name, exit_code)


if __name__ == "__main__":
    run_main(main)
