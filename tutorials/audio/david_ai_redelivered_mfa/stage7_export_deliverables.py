#!/usr/bin/env python3
"""Stage 7: index deliverables — per-speaker 16 kHz Opus, mixed Opus+RTTM, Lhotse."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from david_ai_common import (
    PipelineError,
    finish_stage,
    group_recordings_by_session,
    load_norm_manifest_rows,
    maybe_skip_done_stage,
    run_main,
    session_mixed_audio_path,
    write_jsonl,
)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests-dir", type=Path, required=True)
    ap.add_argument("--audio-16k-dir", type=Path, required=True)
    ap.add_argument("--audio-mixed-dir", type=Path, required=True)
    ap.add_argument("--lhotse-dir", type=Path, required=True)
    ap.add_argument("--prefix", default="david_ai")
    ap.add_argument("--deliverables-dir", type=Path, required=True)
    ap.add_argument("--session", action="append", default=[])
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--stage-done-name", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if maybe_skip_done_stage(args.work_dir, args.stage_done_name, force=args.force):
        return 0

    manifests_dir = args.manifests_dir.resolve()
    audio_16k_dir = args.audio_16k_dir.resolve()
    audio_mixed_dir = args.audio_mixed_dir.resolve()
    lhotse_dir = args.lhotse_dir.resolve()
    deliverables_dir = args.deliverables_dir.resolve()
    prefix = args.prefix

    lhotse_cuts = lhotse_dir / f"{prefix}_aligned_cuts.jsonl.gz"
    if not lhotse_cuts.is_file():
        raise PipelineError(f"Missing Lhotse cutset: {lhotse_cuts} (run stage 4 first)")

    rows, manifest_errors = load_norm_manifest_rows(manifests_dir, sessions=args.session or None)
    by_session = group_recordings_by_session(rows)
    if not by_session:
        raise PipelineError("No sessions found in manifests")

    deliverables_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = deliverables_dir / "manifest.jsonl"

    entries: list[dict] = []
    missing = 0

    for session_id, recs in sorted(by_session.items()):
        mixed_audio = session_mixed_audio_path(audio_mixed_dir, session_id)
        mixed_rttm = audio_mixed_dir / f"{session_id}.rttm"
        speakers: list[dict] = []

        for rec in recs:
            audio_16k = Path(rec["audio_path"])
            if not audio_16k.is_file():
                logger.warning("%s: missing 16 kHz audio %s", rec["recording_id"], audio_16k)
                missing += 1
            speakers.append(
                {
                    "speaker_id": rec["speaker_id"],
                    "recording_id": rec["recording_id"],
                    "audio_16k": str(audio_16k.resolve()),
                }
            )

        if not mixed_audio.is_file():
            logger.warning("%s: missing mixed audio %s", session_id, mixed_audio)
            missing += 1
        if not mixed_rttm.is_file():
            logger.warning("%s: missing mixed RTTM %s", session_id, mixed_rttm)
            missing += 1

        entries.append(
            {
                "session_id": session_id,
                "mixed_audio": str(mixed_audio.resolve()) if mixed_audio.is_file() else "",
                "mixed_rttm": str(mixed_rttm.resolve()) if mixed_rttm.is_file() else "",
                "speakers": speakers,
                "lhotse_aligned_cuts": str(lhotse_cuts.resolve()),
                "lhotse_recordings": str((lhotse_dir / f"{prefix}_recordings.jsonl.gz").resolve()),
                "lhotse_supervisions": str((lhotse_dir / f"{prefix}_supervisions.jsonl.gz").resolve()),
            }
        )

    write_jsonl(manifest_path, entries)

    summary = {
        "sessions": len(entries),
        "speakers_total": sum(len(e["speakers"]) for e in entries),
        "audio_16k_dir": str(audio_16k_dir),
        "audio_mixed_dir": str(audio_mixed_dir),
        "lhotse_dir": str(lhotse_dir),
        "manifest": str(manifest_path),
        "lhotse_aligned_cuts": str(lhotse_cuts),
    }
    (deliverables_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Stage 7 done: sessions=%d speakers=%d missing=%d manifest=%s",
        len(entries),
        summary["speakers_total"],
        missing,
        manifest_path,
    )
    exit_code = 1 if (missing or manifest_errors) else 0
    return finish_stage(args.work_dir, args.stage_done_name, exit_code)


if __name__ == "__main__":
    run_main(main)
