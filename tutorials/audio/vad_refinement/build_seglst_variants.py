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

# ruff: noqa: PLR0913

"""Build baseline-MFA and VAD-refined SegLST files from one session manifest."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from textgrid import TextGrid

from tutorials.audio.vad_refinement.rectify_vad_boundaries import Interval, load_manifest, read_rttm
from tutorials.audio.vad_refinement.textgrid_rttm import SILENCE_MARKS, speech_intervals_from_rows

MAX_ALIAS_TIMING_ERROR = 0.25
MAX_TIME_PRECISION = 9


def _normalized_text(value: object) -> str:
    return " ".join(str(value or "").split())


def infer_speaker_aliases(
    rows: list[dict[str, Any]],
    reference_seglst: list[dict[str, Any]],
) -> dict[str, str]:
    """Map manifest speaker IDs to aliases used by an optional reference SegLST."""
    reference_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in reference_seglst:
        reference_by_text[_normalized_text(entry.get("words"))].append(entry)

    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        text = _normalized_text(row.get("text") or row.get("text_raw"))
        start = float(row["offset"])
        end = start + float(row["duration"])
        candidates = reference_by_text.get(text, [])
        if not candidates:
            candidates = reference_seglst
        best = min(
            candidates,
            key=lambda entry: abs(float(entry["start_time"]) - start) + abs(float(entry["end_time"]) - end),
            default=None,
        )
        if best is None:
            continue
        timing_error = abs(float(best["start_time"]) - start) + abs(float(best["end_time"]) - end)
        if timing_error <= MAX_ALIAS_TIMING_ERROR:
            votes[str(row["speaker_id"])][str(best["speaker"])] += 1
    return {speaker_id: aliases.most_common(1)[0][0] for speaker_id, aliases in votes.items() if aliases}


def _merge_intervals(intervals: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _segment_intervals(
    row: dict[str, Any],
    intervals: dict[str, list[Interval]],
    *,
    boundary_window: float,
    split_internal_pauses: bool,
    fallback_to_segment: bool,
) -> list[Interval]:
    reference_start = float(row["offset"])
    reference_end = reference_start + float(row["duration"])
    matching = [
        (start, end)
        for start, end in intervals.get(str(row["speaker_id"]), [])
        if min(reference_end, end) > max(reference_start, start)
    ]
    if not matching:
        return [(reference_start, reference_end)] if fallback_to_segment else []
    window_start = max(0.0, reference_start - boundary_window)
    window_end = reference_end + boundary_window
    clipped = _merge_intervals([(max(window_start, start), min(window_end, end)) for start, end in matching])
    if not clipped:
        return [(reference_start, reference_end)] if fallback_to_segment else []
    if split_internal_pauses:
        return clipped
    return [(clipped[0][0], clipped[-1][1])]


def _aligned_words(
    row: dict[str, Any],
    textgrid_key: str,
) -> list[tuple[float, float, str]]:
    path_value = row.get(textgrid_key) or row.get("mfa_textgrid")
    if not path_value:
        return []
    path = Path(str(path_value))
    if not path.is_file():
        return []
    offset = float(row["offset"])
    textgrid = TextGrid.fromFile(str(path))
    return [
        (
            offset + float(interval.minTime),
            offset + float(interval.maxTime),
            interval.mark.strip(),
        )
        for interval in textgrid.getFirst("words").intervals
        if interval.mark.strip().casefold() not in {*SILENCE_MARKS, "<unk>"}
    ]


def _piece_texts(
    row: dict[str, Any],
    pieces: list[Interval],
    textgrid_key: str,
) -> list[str | None]:
    raw_text = str(row.get("text") or row.get("text_raw") or "")
    if len(pieces) == 1:
        return [raw_text]
    groups: list[list[str]] = [[] for _ in pieces]
    for word_start, word_end, word in _aligned_words(row, textgrid_key):
        overlaps = [
            max(0.0, min(piece_end, word_end) - max(piece_start, word_start)) for piece_start, piece_end in pieces
        ]
        best_index = max(range(len(pieces)), key=overlaps.__getitem__)
        if overlaps[best_index] > 0:
            groups[best_index].append(word)
    if any(groups):
        return [" ".join(words) if words else None for words in groups]
    longest_index = max(
        range(len(pieces)),
        key=lambda index: pieces[index][1] - pieces[index][0],
    )
    return [raw_text if index == longest_index else None for index in range(len(pieces))]


def _format_time(value: float, precision: int) -> str:
    rendered = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    return rendered if "." in rendered else f"{rendered}.0"


def build_seglst_from_intervals(
    rows: list[dict[str, Any]],
    intervals: dict[str, list[Interval]],
    *,
    speaker_aliases: dict[str, str] | None = None,
    boundary_window: float = 0.5,
    precision: int = 3,
    split_internal_pauses: bool = True,
    textgrid_key: str = "mfa_textgrid",
    fallback_to_segment: bool = True,
) -> list[dict[str, str]]:
    """Build SegLST rows while retaining internal silence gaps."""
    aliases = speaker_aliases or {}
    entries: list[tuple[float, int, int, dict[str, str]]] = []
    for index, row in enumerate(rows):
        pieces = _segment_intervals(
            row,
            intervals,
            boundary_window=boundary_window,
            split_internal_pauses=split_internal_pauses,
            fallback_to_segment=fallback_to_segment,
        )
        if not pieces:
            continue
        texts = _piece_texts(row, pieces, textgrid_key)
        for piece_index, ((start, end), text) in enumerate(zip(pieces, texts, strict=True)):
            if not text:
                continue
            entry = {
                "session_id": str(row["session_id"]),
                "words": text,
                "start_time": _format_time(start, precision),
                "end_time": _format_time(end, precision),
                "speaker": aliases.get(str(row["speaker_id"]), str(row["speaker_id"])),
            }
            entries.append((start, index, piece_index, entry))
    return [entry for _, _, _, entry in sorted(entries)]


def build_seglst(
    rows: list[dict[str, Any]],
    rttm_path: Path,
    *,
    speaker_aliases: dict[str, str] | None = None,
    boundary_window: float = 0.5,
    precision: int = 3,
    split_internal_pauses: bool = True,
    textgrid_key: str = "mfa_textgrid",
    fallback_to_segment: bool = True,
) -> list[dict[str, str]]:
    """Build SegLST rows from an RTTM file."""
    _, intervals = read_rttm(rttm_path)
    return build_seglst_from_intervals(
        rows,
        intervals,
        speaker_aliases=speaker_aliases,
        boundary_window=boundary_window,
        precision=precision,
        split_internal_pauses=split_internal_pauses,
        textgrid_key=textgrid_key,
        fallback_to_segment=fallback_to_segment,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_variants(
    source_manifest: Path,
    refined_manifest: Path,
    refined_rttm: Path,
    output_dir: Path,
    *,
    reference_seglst: Path | None = None,
    speaker_map: Path | None = None,
    boundary_window: float = 0.5,
    precision: int = 3,
    split_internal_pauses: bool = True,
) -> dict[str, object]:
    source_rows = load_manifest(source_manifest)
    refined_rows = load_manifest(refined_manifest)
    session_id = str(source_rows[0]["session_id"])
    if str(refined_rows[0]["session_id"]) != session_id:
        msg = "source and refined manifests have different session IDs"
        raise ValueError(msg)

    aliases: dict[str, str] = {}
    if reference_seglst is not None:
        reference = json.loads(reference_seglst.read_text(encoding="utf-8"))
        aliases.update(infer_speaker_aliases(source_rows, reference))
    if speaker_map is not None:
        configured = json.loads(speaker_map.read_text(encoding="utf-8"))
        if not isinstance(configured, dict):
            msg = "speaker map must be a JSON object"
            raise TypeError(msg)
        aliases.update({str(key): str(value) for key, value in configured.items()})

    original = build_seglst_from_intervals(
        source_rows,
        {speaker_id: list(intervals) for speaker_id, intervals in speech_intervals_from_rows(source_rows).items()},
        speaker_aliases=aliases,
        boundary_window=boundary_window,
        precision=precision,
        split_internal_pauses=split_internal_pauses,
        textgrid_key="mfa_textgrid",
        fallback_to_segment=True,
    )
    refined = build_seglst(
        refined_rows,
        refined_rttm,
        speaker_aliases=aliases,
        boundary_window=boundary_window,
        precision=precision,
        split_internal_pauses=split_internal_pauses,
        textgrid_key="vad_refined_textgrid",
        fallback_to_segment=False,
    )

    original_output = output_dir / f"{session_id}_mfa.seglst.json"
    refined_output = output_dir / f"{session_id}_vad_refined.seglst.json"
    _write_json(original_output, original)
    _write_json(refined_output, refined)
    summary: dict[str, object] = {
        "session_id": session_id,
        "source_segments": len(source_rows),
        "mfa_segments": len(original),
        "vad_refined_segments": len(refined),
        "split_internal_pauses": split_internal_pauses,
        "speaker_aliases": aliases,
        "source_manifest": str(source_manifest.resolve()),
        "refined_manifest": str(refined_manifest.resolve()),
        "refined_rttm": str(refined_rttm.resolve()),
        "mfa_seglst": str(original_output.resolve()),
        "vad_refined_seglst": str(refined_output.resolve()),
    }
    _write_json(output_dir / f"{session_id}_seglst_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--refined-manifest", type=Path, required=True)
    parser.add_argument("--refined-rttm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-seglst", type=Path, default=None)
    parser.add_argument("--speaker-map", type=Path, default=None)
    parser.add_argument("--boundary-window", type=float, default=0.5)
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument(
        "--split-internal-pauses",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.boundary_window < 0:
        parser.error("--boundary-window must be non-negative")
    if not 0 <= args.precision <= MAX_TIME_PRECISION:
        parser.error(f"--precision must be in [0, {MAX_TIME_PRECISION}]")
    for path in (
        args.source_manifest,
        args.refined_manifest,
        args.refined_rttm,
        args.reference_seglst,
        args.speaker_map,
    ):
        if path is not None and not path.is_file():
            parser.error(f"input file does not exist: {path}")
    return args


def main() -> int:
    args = parse_args()
    summary = build_variants(
        args.source_manifest,
        args.refined_manifest,
        args.refined_rttm,
        args.output_dir,
        reference_seglst=args.reference_seglst,
        speaker_map=args.speaker_map,
        boundary_window=args.boundary_window,
        precision=args.precision,
        split_internal_pauses=args.split_internal_pauses,
    )
    print(summary["vad_refined_seglst"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
