# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ALM Data Overlap Stage - Native NeMo Curator Implementation.

Filters overlapping windows based on threshold.
Follows the exact pattern from NeMo Curator:
https://github.com/NVIDIA-NeMo/Curator/blob/main/nemo_curator/stages/audio/common.py

Produces identical output to SDP implementation.
"""

import time
from dataclasses import dataclass
from typing import Any

from nemo_curator.stages.audio.common import LegacySpeechStage
from nemo_curator.tasks import AudioBatch

MAX_OVERLAP_PERCENTAGE = 100


def _calculate_total_dur(windows: list[dict[str, Any]]) -> float:
    """Calculate total duration from windows data."""
    try:
        return sum(
            seg[-1]["end"] - seg[0]["start"] for window in windows for seg in [window.get("segments", [])] if seg
        )
    except (KeyError, IndexError, TypeError):
        return 0.0


def _calculate_duration_list(windows: list[dict[str, Any]]) -> list[float]:
    """Calculate list of durations from windows data."""
    try:
        return [
            seg[-1]["end"] - seg[0]["start"] for window in windows for seg in [window.get("segments", [])] if seg
        ]
    except (KeyError, IndexError, TypeError):
        return []


def _calculate_timestamps(windows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Calculate (end, start) timestamp pairs from windows data."""
    try:
        return [
            (seg[-1]["end"], seg[0]["start"]) for window in windows for seg in [window.get("segments", [])] if seg
        ]
    except (KeyError, IndexError, TypeError):
        return []


def _overlap_ratio(seg1: tuple[float, float], seg2: tuple[float, float]) -> float:
    """Calculate overlap ratio between two segments (stored as (end, start) tuples)."""
    start1, end1 = seg1[1], seg1[0]
    start2, end2 = seg2[1], seg2[0]
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_duration = max(0, overlap_end - overlap_start)
    smaller_duration = min(end1 - start1, end2 - start2)
    return overlap_duration / smaller_duration if smaller_duration else 0


def _filter_segments(
    segments: list[tuple[float, float]], threshold: float, target_duration: float
) -> list[tuple[float, float]]:
    """Filter out segments that have overlap greater than threshold."""
    sorted_segs = sorted(segments, key=lambda x: (x[1], x[0]))
    remove_indices: set[int] = set()

    for i in range(len(sorted_segs)):
        if i in remove_indices:
            continue
        seg_i = sorted_segs[i]
        start_i, end_i = seg_i[1], seg_i[0]
        dur_i = end_i - start_i

        for j in range(i + 1, len(sorted_segs)):
            if j in remove_indices:
                continue
            seg_j = sorted_segs[j]
            start_j, end_j = seg_j[1], seg_j[0]
            dur_j = end_j - start_j

            if start_j >= end_i:
                break

            ratio = _overlap_ratio(seg_i, seg_j)
            if ratio >= threshold:
                diff_i = abs(dur_i - target_duration)
                diff_j = abs(dur_j - target_duration)

                if diff_i < diff_j:
                    remove_indices.add(j)
                elif diff_j < diff_i:
                    remove_indices.add(i)
                    break
                elif dur_i >= dur_j:
                    remove_indices.add(j)
                else:
                    remove_indices.add(i)
                    break

    return [seg for idx, seg in enumerate(sorted_segs) if idx not in remove_indices]


def _process_filtered_dur(timestamps: list[tuple[float, float]]) -> float:
    """Get total duration of qualified segments."""
    return sum(end - start for end, start in timestamps)


def _process_filtered_dur_list(timestamps: list[tuple[float, float]]) -> list[float]:
    """Get duration list of qualified segments."""
    return [end - start for end, start in timestamps]


def _get_filtered_windows(
    windows: list[dict[str, Any]], filtered_timestamps: list[tuple[float, float]]
) -> list[dict[str, Any]]:
    """Get complete window objects that correspond to filtered timestamps."""
    if not windows or not filtered_timestamps:
        return []

    filtered_set = {(round(end, 6), round(start, 6)) for end, start in filtered_timestamps}

    filtered_windows = []
    for window in windows:
        segments = window.get("segments", [])
        if not segments:
            continue

        window_timestamp = (round(segments[-1]["end"], 6), round(segments[0]["start"], 6))
        if window_timestamp in filtered_set:
            filtered_windows.append(window)

    return filtered_windows


def _get_filepath_from_stats(stats: dict[str, Any] | None, key: str) -> str | None:
    return stats.get(key) if isinstance(stats, dict) else None


@dataclass
class ALMDataOverlapStage(LegacySpeechStage):
    """
    Filter overlapping ALM windows.

    Native NeMo Curator stage that removes windows with overlap exceeding
    the threshold, keeping windows closest to target duration.

    This follows the exact pattern from nemo_curator.stages.audio.common:
    - Inherits from LegacySpeechStage
    - Uses @dataclass decorator
    - Implements process_dataset_entry() method
    - Returns list[AudioBatch] from process_dataset_entry

    Produces identical output to SDP implementation.
    """

    # Processing parameters (EXACT match to SDP)
    overlap_percentage: int = 0
    target_duration: float = 120.0

    # Stage metadata
    name: str = "alm_data_overlap"

    def __post_init__(self) -> None:
        """Validate parameters."""
        if not (0 <= self.overlap_percentage <= MAX_OVERLAP_PERCENTAGE):
            msg = f"overlap_percentage must be 0-100, got {self.overlap_percentage}"
            raise ValueError(msg)
        if self.target_duration <= 0:
            msg = "target_duration must be positive"
            raise ValueError(msg)

    def process_dataset_entry(self, data_entry: dict[str, Any]) -> list[AudioBatch]:
        """
        Process a single manifest entry and filter overlapping windows.

        Args:
            data_entry: Single entry from manifest (dict with windows, etc.)

        Returns:
            list[AudioBatch] - Always returns entry (matching SDP behavior)
        """
        t0 = time.perf_counter()
        input_windows = len(data_entry.get("windows", []))
        result = self._filter_overlaps(data_entry)
        filter_time = time.perf_counter() - t0

        output_windows = len(result.get("filtered_windows", []))
        filtered_dur = result.get("filtered_dur", 0.0)
        self._log_metrics(
            {
                "filter_time": filter_time,
                "input_windows": input_windows,
                "output_windows": output_windows,
                "filtered_dur": filtered_dur,
            }
        )

        return [AudioBatch(data=[result])]

    def _filter_overlaps(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Filter overlapping windows from entry."""
        threshold = self.overlap_percentage / MAX_OVERLAP_PERCENTAGE

        windows = entry.get("windows", [])
        if not windows:
            return entry

        total_dur_window = _calculate_total_dur(windows)
        total_dur_list_window = _calculate_duration_list(windows)
        total_dur_list_window_timestamps = _calculate_timestamps(windows)

        filtered_timestamps = _filter_segments(
            total_dur_list_window_timestamps, threshold=threshold, target_duration=self.target_duration
        )
        filtered_windows = _get_filtered_windows(windows, filtered_timestamps)

        filtered_dur = _process_filtered_dur(filtered_timestamps)
        filtered_dur_list = _process_filtered_dur_list(filtered_timestamps)

        stats = entry.get("stats", {})
        manifest_filepath = _get_filepath_from_stats(stats, "manifest_path")
        swift_filepath = _get_filepath_from_stats(stats, "swift_path")

        result = entry.copy()
        result["total_dur_window"] = total_dur_window
        result["total_dur_list_window"] = total_dur_list_window
        result["total_dur_list_window_timestamps"] = total_dur_list_window_timestamps
        result["filtered"] = filtered_timestamps
        result["filtered_windows"] = filtered_windows
        result["filtered_dur"] = filtered_dur
        result["filtered_dur_list"] = filtered_dur_list
        result["manifest_filepath"] = manifest_filepath
        result["swift_filepath"] = swift_filepath
        return result
