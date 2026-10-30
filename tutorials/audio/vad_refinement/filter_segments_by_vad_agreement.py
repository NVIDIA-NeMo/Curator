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

"""Score and filter text segments by frame-level VAD agreement with a reference TextGrid."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tutorials.audio.vad_refinement.textgrid_rttm import SILENCE_MARKS, read_textgrid

Interval = tuple[float, float]
DEFAULT_FRAME_DURATION = 0.01
DEFAULT_HISTOGRAM_BIN_WIDTH = 0.05
RATIO_EPSILON = 1e-12
INTERVAL_ENDPOINT_COUNT = 2
ALIGNMENT_ITEM_MIN_FIELDS = 3
FALLBACK_ALIGNMENT_MARKS = {"[timestamp alignment gt labels]"}
SUMMARY_QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 1.0)

INTERVAL_KEYS = {
    "raw": {
        "silero": (
            "silero_raw_intervals",
            "silero_vad_raw_intervals",
            "silero_intervals",
            "silero_vad_intervals",
        ),
        "sortformer": (
            "sortformer_raw_intervals",
            "sortformer_vad_raw_intervals",
            "sortformer_intervals",
            "sortformer_vad_intervals",
        ),
    },
    "processed": {
        "silero": ("silero_intervals", "silero_vad_intervals"),
        "sortformer": ("sortformer_intervals", "sortformer_vad_intervals"),
    },
}


@dataclass(frozen=True)
class IntervalIndex:
    intervals: tuple[Interval, ...]
    starts: tuple[float, ...]

    @classmethod
    def build(cls, intervals: list[Interval]) -> IntervalIndex:
        merged = merge_intervals(intervals)
        return cls(tuple(merged), tuple(start for start, _ in merged))

    def contains(self, timestamp: float) -> bool:
        interval_index = bisect_right(self.starts, timestamp) - 1
        return interval_index >= 0 and timestamp < self.intervals[interval_index][1]


@dataclass(frozen=True)
class ManifestScore:
    report_rows: list[dict[str, Any]]
    selected_rows: list[dict[str, Any]]
    counts: Counter[str]
    session_id: str
    language: str


@dataclass(frozen=True)
class FilterConfig:
    frame_duration: float = DEFAULT_FRAME_DURATION
    histogram_bin_width: float = DEFAULT_HISTOGRAM_BIN_WIDTH
    vad_variant: str = "raw"
    reference_source: str = "refined"
    max_bad_ratio: float | None = None
    missing_reference_policy: str = "skip"


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted((float(start), float(end)) for start, end in intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + RATIO_EPSILON:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            msg = f"{path}:{line_number}: each JSONL value must be an object"
            raise TypeError(msg)
        rows.append(value)
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _write_distribution_csv(path: Path, distribution: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames = (
        "lower_bound",
        "upper_bound",
        "count",
        "percent",
        "cumulative_count",
        "cumulative_percent",
    )
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(distribution["bins"])
    os.replace(temporary, path)


def _text(row: dict[str, Any]) -> str:
    value = row.get("text") if "text" in row else row.get("text_raw")
    return " ".join(str(value or "").split())


def _embedded_alignment_intervals(row: dict[str, Any]) -> list[Interval] | None:
    alignment = row.get("alignment")
    if not isinstance(alignment, dict) or not isinstance(alignment.get("word"), list):
        return None
    offset = float(row["offset"])
    duration = float(row["duration"])
    time_base = str(row.get("alignment_time_base") or "global")
    if time_base not in {"global", "segment"}:
        msg = f"unsupported alignment_time_base: {time_base!r}"
        raise ValueError(msg)
    intervals: list[Interval] = []
    for item_index, item in enumerate(alignment["word"]):
        if not isinstance(item, (list, tuple)) or len(item) < ALIGNMENT_ITEM_MIN_FIELDS:
            msg = f"alignment.word item {item_index} must contain symbol, start, and duration"
            raise TypeError(msg)
        mark = str(item[0]).strip().casefold()
        if mark in SILENCE_MARKS or mark in FALLBACK_ALIGNMENT_MARKS:
            continue
        start = float(item[1])
        word_duration = float(item[2])
        local_start = start - offset if time_base == "global" else start
        local_end = local_start + word_duration
        intervals.append((max(0.0, local_start), min(duration, local_end)))
    return merge_intervals(intervals)


def reference_speech_intervals(row: dict[str, Any], source: str = "refined") -> list[Interval]:
    """Return segment-local speech intervals from the selected reference alignment."""
    if source == "refined":
        path_value = row.get("vad_refined_textgrid") or row.get("rttm_textgrid")
    elif source == "mfa":
        path_value = row.get("mfa_textgrid")
    else:
        msg = f"unsupported reference source: {source!r}"
        raise ValueError(msg)
    if path_value:
        path = Path(str(path_value))
        if path.is_file():
            duration = float(row["duration"])
            textgrid = read_textgrid(path)
            intervals = [
                (
                    max(0.0, float(interval.minTime)),
                    min(duration, float(interval.maxTime)),
                )
                for interval in textgrid.getFirst("words").intervals
                if interval.mark.strip().casefold() not in SILENCE_MARKS
            ]
            return merge_intervals(intervals)
    embedded = _embedded_alignment_intervals(row)
    if embedded is not None:
        return embedded
    if path_value:
        msg = f"{source} reference TextGrid does not exist: {path_value}"
        raise FileNotFoundError(msg)
    msg = f"segment has neither a {source} reference TextGrid nor alignment.word"
    raise ValueError(msg)


def _normalize_audit_intervals(value: object, key: str) -> list[Interval]:
    if not isinstance(value, (list, tuple)):
        msg = f"audit field {key!r} must be a list of [start, end] intervals"
        raise TypeError(msg)
    intervals: list[Interval] = []
    for index, interval in enumerate(value):
        if not isinstance(interval, (list, tuple)) or len(interval) != INTERVAL_ENDPOINT_COUNT:
            msg = f"audit field {key!r} interval {index} is not [start, end]"
            raise TypeError(msg)
        intervals.append((float(interval[0]), float(interval[1])))
    return merge_intervals(intervals)


def detector_intervals(
    speaker_audit: dict[str, Any],
    detector: str,
    variant: str,
) -> tuple[list[Interval], str]:
    interval_variant = "raw" if variant == "raw_event_excluded" else variant
    for key in INTERVAL_KEYS[interval_variant][detector]:
        if key in speaker_audit:
            return _normalize_audit_intervals(speaker_audit[key], key), key
    expected = ", ".join(INTERVAL_KEYS[interval_variant][detector])
    msg = f"speaker audit has no {detector} intervals; expected one of: {expected}"
    raise KeyError(msg)


def sound_event_intervals(speaker_audit: dict[str, Any]) -> tuple[list[Interval], str]:
    for key in ("sound_event_intervals", "sound_expert_event_intervals"):
        if key in speaker_audit:
            return _normalize_audit_intervals(speaker_audit[key], key), key
    return [], "none"


def score_segment_frames(  # noqa: PLR0913
    row: dict[str, Any],
    reference_intervals: list[Interval],
    silero_intervals: list[Interval],
    sortformer_intervals: list[Interval],
    *,
    frame_duration: float = DEFAULT_FRAME_DURATION,
    event_intervals: list[Interval] | None = None,
) -> dict[str, int | float | None]:
    """Count frame states; bad means both VADs disagree with the reference TextGrid."""
    if frame_duration <= 0:
        msg = "frame_duration must be positive"
        raise ValueError(msg)
    duration = float(row["duration"])
    offset = float(row["offset"])
    if duration <= 0:
        msg = "segment duration must be positive"
        raise ValueError(msg)

    reference = IntervalIndex.build(reference_intervals)
    silero = IntervalIndex.build(silero_intervals)
    sortformer = IntervalIndex.build(sortformer_intervals)
    events = IntervalIndex.build(event_intervals or [])
    total_frames = max(1, math.ceil(duration / frame_duration - RATIO_EPSILON))
    counts: Counter[str] = Counter()

    for frame_index in range(total_frames):
        local_time = min((frame_index + 0.5) * frame_duration, duration - RATIO_EPSILON)
        global_time = offset + local_time
        event_excluded = events.contains(global_time)
        if event_excluded:
            reference_speech = silero_speech = sortformer_speech = False
            counts["event_excluded_frames"] += 1
        else:
            reference_speech = reference.contains(local_time)
            silero_speech = silero.contains(global_time)
            sortformer_speech = sortformer.contains(global_time)

        counts["reference_speech_frames" if reference_speech else "reference_silence_frames"] += 1
        counts["silero_speech_frames" if silero_speech else "silero_silence_frames"] += 1
        counts["sortformer_speech_frames" if sortformer_speech else "sortformer_silence_frames"] += 1

        silero_agrees = silero_speech == reference_speech
        sortformer_agrees = sortformer_speech == reference_speech
        if silero_agrees and sortformer_agrees:
            counts["both_agree_frames"] += 1
        elif silero_agrees:
            counts["silero_only_agrees_frames"] += 1
        elif sortformer_agrees:
            counts["sortformer_only_agrees_frames"] += 1
        else:
            counts["bad_frames"] += 1
            counts["both_miss_speech_frames" if reference_speech else "both_false_alarm_frames"] += 1

    bad_frames = counts["bad_frames"]
    good_frames = total_frames - bad_frames
    return {
        "frame_duration": frame_duration,
        "total_frames": total_frames,
        "good_frames": good_frames,
        "bad_frames": bad_frames,
        "good_ratio": good_frames / total_frames,
        "bad_ratio": bad_frames / total_frames,
        "bad_to_good_ratio": bad_frames / good_frames if good_frames else None,
        "good_to_bad_ratio": good_frames / bad_frames if bad_frames else None,
        "both_agree_frames": counts["both_agree_frames"],
        "silero_only_agrees_frames": counts["silero_only_agrees_frames"],
        "sortformer_only_agrees_frames": counts["sortformer_only_agrees_frames"],
        "both_miss_speech_frames": counts["both_miss_speech_frames"],
        "both_false_alarm_frames": counts["both_false_alarm_frames"],
        "reference_speech_frames": counts["reference_speech_frames"],
        "reference_silence_frames": counts["reference_silence_frames"],
        "silero_speech_frames": counts["silero_speech_frames"],
        "silero_silence_frames": counts["silero_silence_frames"],
        "sortformer_speech_frames": counts["sortformer_speech_frames"],
        "sortformer_silence_frames": counts["sortformer_silence_frames"],
        "event_excluded_frames": counts["event_excluded_frames"],
    }


def _report_identity(
    row: dict[str, Any],
    manifest_path: Path,
    row_index: int,
) -> dict[str, Any]:
    return {
        "status": "scored",
        "source_manifest": str(manifest_path.resolve()),
        "session_id": str(row.get("session_id") or manifest_path.parent.name),
        "speaker_id": str(row.get("speaker_id") or ""),
        "segment_id": str(row.get("segment_id") or row.get("segment_index") or row_index),
        "segment_index": row.get("segment_index", row_index),
        "language": str(row.get("language") or ""),
        "audio_filepath": str(row.get("audio_filepath") or ""),
        "offset": float(row["offset"]),
        "duration": float(row["duration"]),
        "text": _text(row),
    }


def _agreement_payload(report_row: dict[str, Any], variant: str) -> dict[str, Any]:
    excluded = {
        "status",
        "source_manifest",
        "session_id",
        "speaker_id",
        "segment_id",
        "segment_index",
        "language",
        "audio_filepath",
        "offset",
        "duration",
        "text",
        "vad_variant",
        "silero_interval_source",
        "sortformer_interval_source",
        "event_interval_source",
    }
    return {
        "vad_variant": variant,
        **{key: value for key, value in report_row.items() if key not in excluded},
    }


def _segment_scoring_inputs(
    row: dict[str, Any],
    speaker_audits: dict[str, Any],
    vad_variant: str,
    reference_source: str,
) -> tuple[list[Interval], list[Interval], list[Interval], list[Interval], str, str, str]:
    reference_intervals = reference_speech_intervals(row, reference_source)
    if not reference_intervals:
        msg = f"{reference_source} reference TextGrid contains no lexical speech intervals"
        raise ValueError(msg)
    speaker_id = str(row.get("speaker_id") or "")
    speaker_audit = speaker_audits.get(speaker_id)
    if not isinstance(speaker_audit, dict):
        msg = f"audit has no speaker {speaker_id!r}"
        raise KeyError(msg)
    silero_intervals, silero_key = detector_intervals(speaker_audit, "silero", vad_variant)
    sortformer_intervals, sortformer_key = detector_intervals(speaker_audit, "sortformer", vad_variant)
    events, event_key = sound_event_intervals(speaker_audit) if vad_variant == "raw_event_excluded" else ([], "none")
    return reference_intervals, silero_intervals, sortformer_intervals, events, silero_key, sortformer_key, event_key


def score_manifest(
    manifest_path: Path,
    audit_path: Path,
    config: FilterConfig,
) -> ManifestScore:
    rows = _load_jsonl(manifest_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    speaker_audits = audit.get("speakers")
    if not isinstance(speaker_audits, dict):
        msg = f"{audit_path}: missing speakers object"
        raise TypeError(msg)

    report_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    session_id = str(rows[0].get("session_id") or manifest_path.parent.name) if rows else manifest_path.parent.name
    language = str(rows[0].get("language") or "") if rows else ""

    for row_index, row in enumerate(rows):
        counts["input_segments"] += 1
        text = _text(row)
        if not text:
            counts["skipped_no_text"] += 1
            continue
        counts["segments_with_text"] += 1
        identity = _report_identity(row, manifest_path, row_index)
        try:
            (
                reference_intervals,
                silero_intervals,
                sortformer_intervals,
                event_intervals,
                silero_key,
                sortformer_key,
                event_key,
            ) = _segment_scoring_inputs(
                row,
                speaker_audits,
                config.vad_variant,
                config.reference_source,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            if config.missing_reference_policy == "error":
                raise
            counts["skipped_reference_unavailable"] += 1
            report_rows.append(
                {
                    **identity,
                    "status": "skipped_reference_unavailable",
                    "reason": str(exc),
                }
            )
            continue

        metrics = score_segment_frames(
            row,
            reference_intervals,
            silero_intervals,
            sortformer_intervals,
            frame_duration=config.frame_duration,
            event_intervals=event_intervals,
        )
        report_row = {
            **identity,
            "vad_variant": config.vad_variant,
            "reference_source": config.reference_source,
            "silero_interval_source": silero_key,
            "sortformer_interval_source": sortformer_key,
            "event_interval_source": event_key,
            **metrics,
        }
        report_rows.append(report_row)
        counts["scored_segments"] += 1
        if (
            config.max_bad_ratio is not None
            and float(metrics["bad_ratio"] or 0.0) <= config.max_bad_ratio + RATIO_EPSILON
        ):
            selected = dict(row)
            selected["vad_frame_agreement"] = _agreement_payload(report_row, config.vad_variant)
            selected_rows.append(selected)
            counts["selected_segments"] += 1

    return ManifestScore(
        report_rows=report_rows,
        selected_rows=selected_rows,
        counts=counts,
        session_id=session_id,
        language=language,
    )


def resolve_audit_path(manifest_path: Path) -> Path:
    candidates = (
        manifest_path.parent / "audit.json",
        manifest_path.parent / "vad_boundaries.json",
    )
    match = next((path for path in candidates if path.is_file()), None)
    if match is None:
        msg = f"no audit.json or vad_boundaries.json beside {manifest_path}"
        raise FileNotFoundError(msg)
    return match


def _filtered_manifest_path(
    output_dir: Path,
    manifest_path: Path,
    work_root: Path | None,
    score: ManifestScore,
) -> Path:
    if work_root is not None:
        try:
            relative = manifest_path.resolve().relative_to(work_root.resolve())
        except ValueError:
            relative = Path(score.language or "unknown") / score.session_id / manifest_path.name
    else:
        relative = Path(score.language or "unknown") / score.session_id / manifest_path.name
    return output_dir / "filtered_manifests" / relative


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, int | float | None]:
    scored = [row for row in rows if row["status"] == "scored"]
    total_frames = sum(int(row["total_frames"]) for row in scored)
    good_frames = sum(int(row["good_frames"]) for row in scored)
    bad_frames = sum(int(row["bad_frames"]) for row in scored)
    bad_ratios = [float(row["bad_ratio"]) for row in scored]
    return {
        "segments": len(scored),
        "total_frames": total_frames,
        "good_frames": good_frames,
        "bad_frames": bad_frames,
        "pooled_good_ratio": good_frames / total_frames if total_frames else None,
        "pooled_bad_ratio": bad_frames / total_frames if total_frames else None,
        "pooled_bad_to_good_ratio": bad_frames / good_frames if good_frames else None,
        "mean_segment_bad_ratio": statistics.fmean(bad_ratios) if bad_ratios else None,
        "median_segment_bad_ratio": statistics.median(bad_ratios) if bad_ratios else None,
        "both_miss_speech_frames": sum(int(row["both_miss_speech_frames"]) for row in scored),
        "both_false_alarm_frames": sum(int(row["both_false_alarm_frames"]) for row in scored),
        "event_excluded_frames": sum(int(row["event_excluded_frames"]) for row in scored),
    }


def _quantile(sorted_values: list[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return sorted_values[lower_index] * (1 - fraction) + sorted_values[upper_index] * fraction


def bad_ratio_distribution(
    rows: list[dict[str, Any]],
    *,
    bin_width: float = DEFAULT_HISTOGRAM_BIN_WIDTH,
) -> dict[str, Any]:
    """Build a bounded histogram and quantiles for bad_frames / total_frames."""
    if not 0 < bin_width <= 1:
        msg = "bin_width must be in (0, 1]"
        raise ValueError(msg)
    values = sorted(float(row["bad_ratio"]) for row in rows if row["status"] == "scored")
    bin_count = math.ceil(1 / bin_width)
    counts = [0] * bin_count
    for value in values:
        clamped = min(1.0, max(0.0, value))
        bin_index = min(int((clamped + RATIO_EPSILON) / bin_width), bin_count - 1)
        counts[bin_index] += 1

    bins: list[dict[str, int | float]] = []
    cumulative_count = 0
    for bin_index, count in enumerate(counts):
        lower_bound = bin_index * bin_width
        upper_bound = min(1.0, (bin_index + 1) * bin_width)
        cumulative_count += count
        bins.append(
            {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "count": count,
                "percent": count / len(values) if values else 0.0,
                "cumulative_count": cumulative_count,
                "cumulative_percent": cumulative_count / len(values) if values else 0.0,
            }
        )
    return {
        "metric": "bad_frames / total_frames",
        "count": len(values),
        "bin_width": bin_width,
        "minimum": values[0] if values else None,
        "maximum": values[-1] if values else None,
        "mean": statistics.fmean(values) if values else None,
        "quantiles": {f"p{probability * 100:g}": _quantile(values, probability) for probability in SUMMARY_QUANTILES},
        "bins": bins,
    }


def build_summary(
    report_rows: list[dict[str, Any]],
    counts: Counter[str],
    config: FilterConfig,
    manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in report_rows:
        if row["status"] == "scored":
            by_language[str(row.get("language") or "unknown")].append(row)
    distribution = bad_ratio_distribution(report_rows, bin_width=config.histogram_bin_width)
    return {
        "definition": {
            "good_frame": ("Silero or Sortformer has the same speech/silence state as the reference TextGrid."),
            "bad_frame": (
                "Silero and Sortformer both have the opposite speech/silence state from the reference TextGrid."
            ),
            "frame_label_time": "Frame center; intervals use start-inclusive, end-exclusive bounds.",
            "raw_event_excluded": (
                "Start from raw Silero and Sortformer; set the reference, Silero, and Sortformer "
                "to silence inside Sound Expert event intervals."
            ),
        },
        "frame_duration": config.frame_duration,
        "vad_variant": config.vad_variant,
        "reference_source": config.reference_source,
        "max_bad_ratio": config.max_bad_ratio,
        "bad_ratio_distribution": distribution,
        "counts": dict(sorted(counts.items())),
        "pooled": _aggregate(report_rows),
        "by_language": {
            language: _aggregate(language_rows) for language, language_rows in sorted(by_language.items())
        },
        "manifests": manifests,
    }


def discover_manifests(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.glob(pattern) if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, action="append")
    source.add_argument("--work-root", type=Path)
    parser.add_argument("--manifest-pattern", default="**/segments.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-duration", type=float, default=DEFAULT_FRAME_DURATION)
    parser.add_argument("--histogram-bin-width", type=float, default=DEFAULT_HISTOGRAM_BIN_WIDTH)
    parser.add_argument(
        "--vad-variant",
        choices=("raw", "processed", "raw_event_excluded"),
        default="raw",
    )
    parser.add_argument("--reference-source", choices=("refined", "mfa"), default="refined")
    parser.add_argument("--max-bad-ratio", type=float, default=None)
    parser.add_argument("--missing-reference-policy", choices=("skip", "error"), default="skip")
    args = parser.parse_args()
    if args.frame_duration <= 0:
        parser.error("--frame-duration must be positive")
    if not 0 < args.histogram_bin_width <= 1:
        parser.error("--histogram-bin-width must be in (0, 1]")
    if args.max_bad_ratio is not None and not 0 <= args.max_bad_ratio <= 1:
        parser.error("--max-bad-ratio must be in [0, 1]")
    if args.work_root is not None and not args.work_root.is_dir():
        parser.error(f"--work-root is not a directory: {args.work_root}")
    return args


def main() -> int:
    args = parse_args()
    manifests = (
        sorted(set(args.manifest))
        if args.manifest is not None
        else discover_manifests(args.work_root, args.manifest_pattern)
    )
    if not manifests:
        print("No manifests found")
        return 0

    config = FilterConfig(
        frame_duration=args.frame_duration,
        histogram_bin_width=args.histogram_bin_width,
        vad_variant=args.vad_variant,
        reference_source=args.reference_source,
        max_bad_ratio=args.max_bad_ratio,
        missing_reference_policy=args.missing_reference_policy,
    )
    report_rows: list[dict[str, Any]] = []
    total_counts: Counter[str] = Counter()
    manifest_summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for completed, manifest_path in enumerate(manifests, start=1):
        try:
            audit_path = resolve_audit_path(manifest_path)
            score = score_manifest(manifest_path, audit_path, config)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append({"manifest": str(manifest_path.resolve()), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{completed}/{len(manifests)}] Failed {manifest_path}: {exc}")
            continue
        report_rows.extend(score.report_rows)
        total_counts.update(score.counts)
        filtered_path: Path | None = None
        if args.max_bad_ratio is not None:
            filtered_path = _filtered_manifest_path(args.output_dir, manifest_path, args.work_root, score)
            _write_jsonl(filtered_path, score.selected_rows)
        manifest_summaries.append(
            {
                "manifest": str(manifest_path.resolve()),
                "audit": str(audit_path.resolve()),
                "session_id": score.session_id,
                "language": score.language,
                "scored_segments": score.counts["scored_segments"],
                "selected_segments": score.counts["selected_segments"],
                "filtered_manifest": str(filtered_path.resolve()) if filtered_path else None,
            }
        )
        print(
            f"[{completed}/{len(manifests)}] {score.session_id}: "
            f"scored={score.counts['scored_segments']} selected={score.counts['selected_segments']}"
        )

    report_path = args.output_dir / "segment_frame_agreement.jsonl"
    summary_path = args.output_dir / "frame_agreement_summary.json"
    distribution_path = args.output_dir / "bad_ratio_distribution.csv"
    _write_jsonl(report_path, report_rows)
    summary = build_summary(
        report_rows,
        total_counts,
        config,
        manifests=manifest_summaries,
    )
    summary["errors"] = errors
    summary["complete"] = not errors
    _write_json(
        summary_path,
        summary,
    )
    _write_distribution_csv(distribution_path, summary["bad_ratio_distribution"])
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")
    print(f"Distribution: {distribution_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
