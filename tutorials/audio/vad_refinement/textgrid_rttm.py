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

"""Convert aligned TextGrid words into recording-global speech intervals."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from textgrid import IntervalTier, TextGrid

SILENCE_MARKS = {"", "sil", "sp", "spn", "<eps>"}


def read_textgrid(path: Path) -> TextGrid:
    """Read without the library's default five-decimal timestamp rounding."""
    textgrid = TextGrid()
    # 324 decimal places retain all finite binary64 values, including subnormals.
    textgrid.read(str(path), round_digits=324)
    return textgrid


def _safe_filename_token(value: str) -> str:
    token = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_" for character in value
    ).strip("._")
    return token or "item"


def merge_with_pause_threshold(
    intervals: list[tuple[float, float]],
    *,
    maximum_pause: float,
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start - merged[-1][1] + 1e-9 < maximum_pause:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def speech_intervals_from_rows(
    rows: list[dict[str, Any]],
    *,
    maximum_pause: float = 0.0,
) -> dict[str, tuple[tuple[float, float], ...]]:
    """Map non-empty TextGrid word intervals onto each full speaker recording."""
    by_speaker: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row.get("rttm_textgrid"):
            path = Path(str(row["rttm_textgrid"]))
        elif row.get("alignment_source") == "manifest_fallback" and row.get("fallback_textgrid"):
            path = Path(str(row["fallback_textgrid"]))
        else:
            path = Path(str(row["mfa_textgrid"]))
        textgrid = read_textgrid(path)
        words = textgrid.getFirst("words")
        offset = float(row["offset"])
        duration = max(float(row["duration"]), float(textgrid.maxTime), float(row.get("annotation_duration", 0)))
        segment_end = offset + duration
        for interval in words.intervals:
            mark = interval.mark.strip().casefold()
            if mark in SILENCE_MARKS:
                continue
            start = min(segment_end, max(offset, offset + float(interval.minTime)))
            end = min(segment_end, max(offset, offset + float(interval.maxTime)))
            if end > start:
                by_speaker[str(row["speaker_id"])].append((start, end))
    return {
        speaker_id: tuple(merge_with_pause_threshold(intervals, maximum_pause=maximum_pause))
        for speaker_id, intervals in by_speaker.items()
    }


def _rttm_line(recording_id: str, speaker_id: str, start: float, end: float) -> str:
    return f"SPEAKER {recording_id} 1 {start:.17g} {end - start:.17g} <NA> <NA> {speaker_id} <NA> <NA>"


def write_rttms(
    output_dir: Path,
    session_id: str,
    intervals: dict[str, tuple[tuple[float, float], ...]],
    *,
    filename_suffix: str = "",
) -> None:
    """Write recording-level and session-level RTTMs from aligned word intervals."""
    rttm_dir = output_dir / "rttm"
    rttm_dir.mkdir(parents=True, exist_ok=True)
    session_lines: list[tuple[float, str]] = []
    for speaker_id, speaker_intervals in sorted(intervals.items()):
        recording_id = _safe_filename_token(f"{speaker_id}_{session_id}_aligned")
        lines = [_rttm_line(recording_id, speaker_id, start, end) for start, end in speaker_intervals]
        (rttm_dir / f"{recording_id}{filename_suffix}.rttm").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        session_lines.extend(
            (start, _rttm_line(session_id, speaker_id, start, end)) for start, end in speaker_intervals
        )
    safe_session_id = _safe_filename_token(session_id)
    (rttm_dir / f"{safe_session_id}{filename_suffix}.rttm").write_text(
        "\n".join(line for _, line in sorted(session_lines)) + ("\n" if session_lines else ""),
        encoding="utf-8",
    )


def _words(path: Path, offset: float) -> list[tuple[float, float, str]]:
    textgrid = read_textgrid(path)
    return [
        (
            offset + float(interval.minTime),
            offset + float(interval.maxTime),
            interval.mark.strip(),
        )
        for interval in textgrid.getFirst("words").intervals
        if interval.mark.strip()
    ]


def _write_textgrid(
    words: list[tuple[float, float, str]],
    path: Path,
    *,
    xmax: float,
) -> None:
    xmax = max(xmax, *(end for _, end, _ in words)) if words else xmax
    intervals: list[tuple[float, float, str]] = []
    previous_end = 0.0
    for start, end, word in sorted(words):
        if start > previous_end + 0.001:
            intervals.append((previous_end, start, ""))
        intervals.append((start, end, word))
        previous_end = max(previous_end, end)
    if previous_end < xmax:
        intervals.append((previous_end, xmax, ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write('File type = "ooTextFile"\n')
        stream.write('Object class = "TextGrid"\n\n')
        stream.write(f"xmin = 0.0\nxmax = {xmax}\n")
        stream.write("tiers? <exists>\nsize = 1\nitem []:\n")
        stream.write('    item [1]:\n        class = "IntervalTier"\n')
        stream.write(f'        name = "words"\n        xmin = 0.0\n        xmax = {xmax}\n')
        stream.write(f"        intervals: size = {len(intervals)}\n")
        for index, (start, end, word) in enumerate(intervals, start=1):
            safe_word = word.replace('"', '""')
            stream.write(
                f"        intervals [{index}]:\n"
                f"            xmin = {start}\n"
                f"            xmax = {end}\n"
                f'            text = "{safe_word}"\n'
            )


def _write_multitier_textgrid(
    words_by_tier: dict[str, list[tuple[float, float, str]]],
    path: Path,
    *,
    xmax: float,
) -> None:
    textgrid = TextGrid(minTime=0.0, maxTime=xmax)
    for tier_name, words in sorted(words_by_tier.items()):
        tier = IntervalTier(name=tier_name, minTime=0.0, maxTime=xmax)
        previous_end = 0.0
        for start, end, word in sorted(words):
            if end <= start:
                continue
            if start < previous_end - 1e-9:
                msg = f"overlapping words in TextGrid tier {tier_name!r}: {start} < {previous_end}"
                raise ValueError(msg)
            tier.add(start, end, word)
            previous_end = end
        textgrid.append(tier)
    path.parent.mkdir(parents=True, exist_ok=True)
    textgrid.write(str(path))


def write_combined_textgrids(
    rows: list[dict[str, Any]],
    textgrid_dir: Path,
    session_id: str,
) -> None:
    """Write primary and fallback alignment TextGrids per recording and session."""
    primary_by_speaker: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    fallback_by_speaker: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    xmax = max(
        (float(row["offset"]) + max(float(row["duration"]), float(row.get("annotation_duration", 0))) for row in rows),
        default=0.01,
    )
    for row in rows:
        speaker_id = str(row["speaker_id"])
        offset = float(row["offset"])
        primary_path = Path(row["mfa_textgrid"])
        fallback_path = Path(row.get("fallback_textgrid") or primary_path)
        primary_by_speaker[speaker_id].extend(_words(primary_path, offset))
        fallback_by_speaker[speaker_id].extend(_words(fallback_path, offset))

    for speaker_id in sorted(fallback_by_speaker):
        recording_id = _safe_filename_token(f"{speaker_id}_{session_id}_aligned")
        primary = primary_by_speaker.get(speaker_id, [])
        fallback = fallback_by_speaker[speaker_id]
        _write_textgrid(
            primary,
            textgrid_dir / f"{recording_id}_mfa.TextGrid",
            xmax=xmax,
        )
        _write_textgrid(
            fallback,
            textgrid_dir / f"{recording_id}_fallback.TextGrid",
            xmax=xmax,
        )
    _write_multitier_textgrid(
        dict(primary_by_speaker),
        textgrid_dir / f"{_safe_filename_token(session_id)}_mfa.TextGrid",
        xmax=xmax,
    )
    _write_multitier_textgrid(
        dict(fallback_by_speaker),
        textgrid_dir / f"{_safe_filename_token(session_id)}_fallback.TextGrid",
        xmax=xmax,
    )


def write_refined_textgrids(
    rows: list[dict[str, Any]],
    textgrid_dir: Path,
    session_id: str,
) -> None:
    """Combine segment-level VAD-refined TextGrids per recording and session."""
    refined_by_speaker: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    xmax = max(
        (float(row["offset"]) + max(float(row["duration"]), float(row.get("annotation_duration", 0))) for row in rows),
        default=0.01,
    )
    for row in rows:
        speaker_id = str(row["speaker_id"])
        refined_by_speaker[speaker_id].extend(_words(Path(str(row["rttm_textgrid"])), float(row["offset"])))

    for speaker_id, words in sorted(refined_by_speaker.items()):
        recording_id = _safe_filename_token(f"{speaker_id}_{session_id}_aligned")
        _write_textgrid(
            words,
            textgrid_dir / f"{recording_id}_vad_refined.TextGrid",
            xmax=xmax,
        )
    _write_multitier_textgrid(
        dict(refined_by_speaker),
        textgrid_dir / f"{_safe_filename_token(session_id)}_vad_refined.TextGrid",
        xmax=xmax,
    )
