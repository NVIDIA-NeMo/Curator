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

"""Build segment tasks and write WER-filtered per-speaker manifests."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
from stages import normalize_wer_text
from textgrid import TextGrid

from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path

SILENCE_TOKENS = {"", "sil", "sp", "spn", "<eps>"}
HISTOGRAM_BOUNDS = (0.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, float("inf"))


def recording_id(speaker_id: str, session_id: str) -> str:
    return f"{speaker_id}_{session_id}_postprocessed"


def load_fastmss_words(textgrid_path: Path) -> list[tuple[float, float, str]]:
    """Read recording-global word intervals from the FastMSS words tier."""
    textgrid = TextGrid.fromFile(str(textgrid_path))
    tier = textgrid.getFirst("words")
    return [
        (float(interval.minTime), float(interval.maxTime), interval.mark.strip())
        for interval in tier.intervals
        if interval.mark.strip() not in SILENCE_TOKENS
    ]


def segment_alignments(
    words: list[tuple[float, float, str]],
    start: float,
    end: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Clip recording-global FastMSS words to one segment."""
    absolute: list[dict[str, Any]] = []
    relative: list[dict[str, Any]] = []
    for word_start, word_end, word in words:
        clipped_start = max(start, word_start)
        clipped_end = min(end, word_end)
        if clipped_end <= clipped_start:
            continue
        absolute.append(
            {
                "word": word,
                "start": round(clipped_start, 6),
                "end": round(clipped_end, 6),
            }
        )
        relative.append(
            {
                "symbol": word,
                "start": round(clipped_start - start, 6),
                "duration": round(clipped_end - clipped_start, 6),
            }
        )
    return absolute, relative


def load_session_ids(data_root: Path, sessions_file: Path | None) -> list[str]:
    if sessions_file is None:
        return sorted(path.name for path in data_root.iterdir() if path.is_dir())
    return sorted(
        {
            line.strip()
            for line in sessions_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    )


@dataclass(frozen=True)
class SegmentTaskConfig:
    data_root: Path
    masked_audio_dir: Path
    textgrid_dir: Path
    sessions_file: Path | None
    shard_count: int
    shard_index: int


def _build_session_tasks(config: SegmentTaskConfig, session_id: str) -> list[AudioTask]:
    transcript_path = config.data_root / session_id / "machine_generated_transcript.json"
    if not transcript_path.is_file():
        msg = f"missing transcript: {transcript_path}"
        raise FileNotFoundError(msg)
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = payload.get("transcript") if isinstance(payload, dict) else None
    if not isinstance(segments, list):
        msg = f"invalid transcript list: {transcript_path}"
        raise TypeError(msg)

    by_speaker: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for segment_index, segment in enumerate(segments):
        if isinstance(segment, dict) and segment.get("speaker"):
            by_speaker[str(segment["speaker"])].append((segment_index, segment))

    tasks: list[AudioTask] = []
    for speaker_id, speaker_segments in sorted(by_speaker.items()):
        rec_id = recording_id(speaker_id, session_id)
        audio_path = config.masked_audio_dir / f"{rec_id}.wav"
        textgrid_path = config.textgrid_dir / f"{rec_id}_fastmss.TextGrid"
        if not audio_path.is_file():
            msg = f"missing masked speaker WAV: {audio_path}"
            raise FileNotFoundError(msg)
        if not textgrid_path.is_file():
            msg = f"missing FastMSS TextGrid: {textgrid_path}"
            raise FileNotFoundError(msg)
        words = load_fastmss_words(textgrid_path)

        for segment_index, segment in speaker_segments:
            start = float(segment["start"])
            end = float(segment["end"])
            if end <= start:
                continue
            text_raw = str(segment.get("text") or "").strip()
            absolute_words, alignment = segment_alignments(words, start, end)
            tasks.append(
                AudioTask(
                    dataset_name="david_ai_masked",
                    filepath_key="audio_filepath",
                    data={
                        "session_id": session_id,
                        "speaker_id": speaker_id,
                        "recording_id": rec_id,
                        "segment_index": segment_index,
                        "start": start,
                        "end": end,
                        "duration": round(end - start, 6),
                        "audio_filepath": str(audio_path),
                        "text_raw": text_raw,
                        "text": normalize_wer_text(text_raw),
                        "fastmss_textgrid": str(textgrid_path),
                        "words": absolute_words,
                        "alignment": alignment,
                    },
                )
            )
    return tasks


def build_segment_tasks(config: SegmentTaskConfig) -> list[AudioTask]:
    """Build one AudioTask per ground-truth segment for a deterministic shard."""
    session_ids = load_session_ids(config.data_root, config.sessions_file)
    selected_ids = [
        session_id
        for index, session_id in enumerate(session_ids)
        if index % config.shard_count == config.shard_index
    ]
    tasks: list[AudioTask] = []
    for session_id in selected_ids:
        tasks.extend(_build_session_tasks(config, session_id))
    return tasks


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        key: round(float(np.percentile(values, percentile)), 6)
        for key, percentile in (("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90), ("p95", 95), ("p99", 99))
    }


def build_wer_distribution_from_values(
    finite: np.ndarray,
    *,
    total_segments: int,
    applied_threshold_pct: float,
) -> dict[str, Any]:
    """Compute histogram, percentiles, and a robust proposed WER threshold."""
    if finite.size == 0:
        return {
            "segments": total_segments,
            "segments_with_wer": 0,
            "applied_threshold_pct": applied_threshold_pct,
            "recommended_threshold_pct": applied_threshold_pct,
            "histogram": [],
            "percentiles": {},
        }

    percentiles = _percentiles(finite)
    iqr = percentiles["p75"] - percentiles["p25"]
    tukey_upper = percentiles["p75"] + 1.5 * iqr
    recommended = round(min(100.0, max(25.0, percentiles["p95"], tukey_upper)), 6)
    histogram: list[dict[str, Any]] = []
    for lower, upper in pairwise(HISTOGRAM_BOUNDS):
        count = int(np.sum((finite >= lower) & (finite < upper)))
        histogram.append(
            {
                "lower_pct": lower,
                "upper_pct": None if np.isinf(upper) else upper,
                "count": count,
            }
        )
    return {
        "segments": total_segments,
        "segments_with_wer": int(finite.size),
        "segments_without_reference_wer": total_segments - int(finite.size),
        "min_pct": round(float(finite.min()), 6),
        "max_pct": round(float(finite.max()), 6),
        "mean_pct": round(float(finite.mean()), 6),
        "percentiles": percentiles,
        "tukey_upper_fence_pct": round(tukey_upper, 6),
        "recommended_threshold_pct": recommended,
        "applied_threshold_pct": applied_threshold_pct,
        "histogram": histogram,
    }


def build_wer_distribution(rows: list[dict[str, Any]], applied_threshold_pct: float) -> dict[str, Any]:
    """Compute a WER report from segment dictionaries."""
    finite = np.asarray(
        [float(row["wer_pct"]) for row in rows if row.get("wer_pct") is not None],
        dtype=np.float64,
    )
    return build_wer_distribution_from_values(
        finite,
        total_segments=len(rows),
        applied_threshold_pct=applied_threshold_pct,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_pipeline_outputs(
    tasks: list[AudioTask],
    *,
    output_dir: Path,
    threshold_pct: float,
    require_fastmss_alignment: bool,
) -> dict[str, Any]:
    """Write audit rows, filtered per-speaker manifests, and WER analytics."""
    rows: list[dict[str, Any]] = []
    kept_by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        row = dict(task.data)
        row.pop("segment_audio_filepath", None)
        row.pop("source_audio_filepath", None)
        reasons: list[str] = []
        if row.get("wer_pct") is None:
            reasons.append("empty_reference")
        elif float(row["wer_pct"]) > threshold_pct:
            reasons.append("wer_above_threshold")
        if require_fastmss_alignment and not row.get("alignment"):
            reasons.append("missing_fastmss_alignment")
        row["keep"] = not reasons
        row["rejection_reasons"] = reasons
        rows.append(row)
        if row["keep"]:
            kept_by_recording[str(row["recording_id"])].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "segments_with_wer.jsonl", rows)
    manifests_dir = output_dir / "manifests"
    for rec_id, recording_rows in kept_by_recording.items():
        manifest_rows = [
            {
                "audio_filepath": row["audio_filepath"],
                "offset": row["start"],
                "duration": row["duration"],
                "text": row["text"],
                "text_raw": row["text_raw"],
                "pred_text": row["pred_text"],
                "wer_pct": row["wer_pct"],
                "session_id": row["session_id"],
                "speaker_id": row["speaker_id"],
                "recording_id": row["recording_id"],
                "segment_index": row["segment_index"],
                "alignment": row["alignment"],
                "alignment_source": "fastmss_textgrid",
            }
            for row in recording_rows
        ]
        _write_jsonl(manifests_dir / f"{rec_id}.jsonl", manifest_rows)

    report = build_wer_distribution(rows, threshold_pct)
    report.update(
        {
            "kept_segments": sum(bool(row["keep"]) for row in rows),
            "rejected_segments": sum(not row["keep"] for row in rows),
            "recording_manifests": len(kept_by_recording),
            "require_fastmss_alignment": require_fastmss_alignment,
        }
    )
    (output_dir / "wer_distribution.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
