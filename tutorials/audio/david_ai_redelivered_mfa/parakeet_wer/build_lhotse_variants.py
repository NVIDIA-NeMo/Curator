#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build exact-0%, 0-10%, and 0-100% WER Lhotse CutSet variants."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.supervision import AlignmentItem


@dataclass(frozen=True)
class WERVariant:
    name: str
    upper_pct: float
    exact_zero: bool = False


VARIANTS = (
    WERVariant(name="wer_000", upper_pct=0.0, exact_zero=True),
    WERVariant(name="wer_000_010", upper_pct=10.0),
    WERVariant(name="wer_000_100", upper_pct=100.0),
)


def segment_matches_variant(row: dict[str, Any], variant: WERVariant) -> bool:
    wer = row.get("wer_pct")
    if wer is None or not row.get("alignment"):
        return False
    value = float(wer)
    return value == 0.0 if variant.exact_zero else 0.0 <= value <= variant.upper_pct


def _alignment_items(row: dict[str, Any], cut_duration: float) -> list[AlignmentItem]:
    items: list[AlignmentItem] = []
    for item in row["alignment"]:
        start = max(0.0, float(item["start"]))
        end = min(cut_duration, start + float(item["duration"]))
        if end <= start:
            continue
        items.append(
            AlignmentItem(
                symbol=str(item["symbol"]),
                start=round(start, 6),
                duration=round(end - start, 6),
            )
        )
    return items


def _build_cut(row: dict[str, Any], recording: Recording) -> MonoCut | None:
    offset = max(0.0, float(row["start"]))
    duration = min(float(row["duration"]), recording.duration - offset)
    if duration <= 0:
        return None
    cut_id = f"{row['recording_id']}_{int(row['segment_index']):05d}"
    supervision = SupervisionSegment(
        id=cut_id,
        recording_id=recording.id,
        start=0.0,
        duration=duration,
        channel=0,
        text=str(row["text"]),
        language="en",
        speaker=str(row["speaker_id"]),
        alignment={"word": _alignment_items(row, duration)},
        custom={
            "session_id": row["session_id"],
            "segment_index": int(row["segment_index"]),
            "text_raw": row["text_raw"],
            "pred_text": row["pred_text"],
            "wer_pct": float(row["wer_pct"]),
            "alignment_source": "fastmss_textgrid",
        },
    )
    return MonoCut(
        id=cut_id,
        start=offset,
        duration=duration,
        channel=0,
        recording=recording,
        supervisions=[supervision],
        custom={"wer_pct": float(row["wer_pct"])},
    )


def build_variant(
    audit_path: Path,
    output_dir: Path,
    variant: WERVariant,
    recording_cache: dict[str, Recording],
) -> dict[str, Any]:
    cuts: list[MonoCut] = []
    matched = invalid_bounds = 0
    with audit_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not segment_matches_variant(row, variant):
                continue
            matched += 1
            recording_id = str(row["recording_id"])
            if recording_id not in recording_cache:
                recording_cache[recording_id] = Recording.from_file(
                    row["audio_filepath"],
                    recording_id=recording_id,
                )
            cut = _build_cut(row, recording_cache[recording_id])
            if cut is None:
                invalid_bounds += 1
                continue
            cuts.append(cut)

    variant_dir = output_dir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    cut_path = variant_dir / "cuts.jsonl.gz"
    CutSet.from_cuts(cuts).to_file(cut_path)
    summary = {
        "variant": variant.name,
        "exact_zero": variant.exact_zero,
        "upper_wer_pct": variant.upper_pct,
        "matched_segments": matched,
        "cuts_written": len(cuts),
        "invalid_audio_bounds": invalid_bounds,
        "recordings_referenced": len({cut.recording_id for cut in cuts}),
        "cut_manifest": str(cut_path),
    }
    (variant_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_all_variants(audit_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Build all configured Lhotse WER variants from one shard audit."""
    recording_cache: dict[str, Recording] = {}
    summaries = [build_variant(audit_path, output_dir, variant, recording_cache) for variant in VARIANTS]
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.audit_jsonl.is_file():
        parser.error(f"audit JSONL does not exist: {args.audit_jsonl}")
    summaries = build_all_variants(args.audit_jsonl, args.output_dir)
    summary_path = args.output_dir / "summary.json"
    print(json.dumps(summaries, indent=2, sort_keys=True))
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
