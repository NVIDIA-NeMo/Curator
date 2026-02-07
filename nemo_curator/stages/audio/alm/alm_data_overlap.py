# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
ALM Data Overlap Stage - Native NeMo Curator Implementation.

Filters overlapping windows based on threshold.
Follows the exact pattern from NeMo Curator:
https://github.com/NVIDIA-NeMo/Curator/blob/main/nemo_curator/stages/audio/common.py

Produces identical output to SDP implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from nemo_curator.stages.audio.common import LegacySpeechStage
from nemo_curator.tasks import AudioBatch

# Constants for validation
MAX_OVERLAP_PERCENTAGE = 100


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

    # Parallelism (used by runner, passed via config)
    max_workers: int = -1

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

    # ========================================
    # SDP HELPER METHODS
    # ========================================

    def calculate_total_dur(self, windows: list[dict[str, Any]]) -> float:
        """Calculate total duration from windows data."""
        try:
            return sum(
                seg[-1]["end"] - seg[0]["start"] for window in windows for seg in [window.get("segments", [])] if seg
            )
        except (KeyError, IndexError, TypeError):
            return 0.0

    def calculate_duration_list(self, windows: list[dict[str, Any]]) -> list[float]:
        """Calculate list of durations from windows data."""
        try:
            return [
                seg[-1]["end"] - seg[0]["start"] for window in windows for seg in [window.get("segments", [])] if seg
            ]
        except (KeyError, IndexError, TypeError):
            return []

    def calculate_timestamps(self, windows: list[dict[str, Any]]) -> list[tuple[float, float]]:
        """Calculate timestamp pairs from windows data."""
        try:
            return [
                (seg[-1]["end"], seg[0]["start"]) for window in windows for seg in [window.get("segments", [])] if seg
            ]
        except (KeyError, IndexError, TypeError):
            return []

    def overlap_ratio(self, seg1: tuple[float, float], seg2: tuple[float, float]) -> float:
        """Calculate overlap ratio between two segments."""
        start1, end1 = seg1[1], seg1[0]
        start2, end2 = seg2[1], seg2[0]
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_duration = max(0, overlap_end - overlap_start)
        duration1 = end1 - start1
        duration2 = end2 - start2
        smaller_duration = min(duration1, duration2)
        return overlap_duration / smaller_duration if smaller_duration else 0

    def filter_segments(
        self, segments: list[tuple[float, float]], threshold: float = 0.9
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

                ratio = self.overlap_ratio(seg_i, seg_j)
                if ratio >= threshold:
                    diff_i = abs(dur_i - self.target_duration)
                    diff_j = abs(dur_j - self.target_duration)

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

    def process_filtered(self, timestamps: list[tuple[float, float]]) -> float:
        """Get total duration of qualified segments."""
        total = 0.0
        for end, start in timestamps:
            dur = end - start
            total += dur
        return total

    def process_filtered_list(self, timestamps: list[tuple[float, float]]) -> list[float]:
        """Get duration lists of qualified segments."""
        return [end - start for end, start in timestamps]

    def get_manifest_filepath(self, stats: dict[str, Any] | None) -> str | None:
        """Extract manifest filepath from stats."""
        return stats.get("manifest_path") if isinstance(stats, dict) else None

    def get_swift_filepath(self, stats: dict[str, Any] | None) -> str | None:
        """Extract Swift filepath from stats."""
        return stats.get("swift_path") if isinstance(stats, dict) else None

    def get_filtered_windows(
        self, windows: list[dict[str, Any]], filtered_timestamps: list[tuple[float, float]]
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

            window_end = segments[-1]["end"]
            window_start = segments[0]["start"]
            window_timestamp = (round(window_end, 6), round(window_start, 6))

            if window_timestamp in filtered_set:
                filtered_windows.append(window)

        return filtered_windows

    # ========================================
    # MAIN PROCESSING METHODS
    # ========================================

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

        # Log timing metrics for regression tracking
        output_windows = len(result.get("filtered_windows", []))
        self._log_metrics(
            {
                "filter_time": filter_time,
                "input_windows": input_windows,
                "output_windows": output_windows,
            }
        )

        return [AudioBatch(data=[result])]

    def _filter_overlaps(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Filter overlapping windows from entry."""
        threshold = self.overlap_percentage / MAX_OVERLAP_PERCENTAGE

        actual_data = entry
        windows = actual_data.get("windows", [])

        if not windows:
            return actual_data

        total_dur_window = self.calculate_total_dur(windows)
        total_dur_list_window = self.calculate_duration_list(windows)
        total_dur_list_window_timestamps = self.calculate_timestamps(windows)

        filtered_timestamps = self.filter_segments(total_dur_list_window_timestamps, threshold=threshold)
        filtered_windows = self.get_filtered_windows(windows, filtered_timestamps)

        filtered_dur = self.process_filtered(filtered_timestamps)
        filtered_dur_list = self.process_filtered_list(filtered_timestamps)

        stats = actual_data.get("stats", {})
        manifest_filepath = self.get_manifest_filepath(stats)
        swift_filepath = self.get_swift_filepath(stats)

        # Always return consistent schema when input has windows
        filtered_metadata = actual_data.copy()
        filtered_metadata["windows"] = filtered_windows
        filtered_metadata["total_dur_window"] = total_dur_window
        filtered_metadata["total_dur_list_window"] = total_dur_list_window
        filtered_metadata["total_dur_list_window_timestamps"] = total_dur_list_window_timestamps
        filtered_metadata["filtered"] = filtered_timestamps
        filtered_metadata["filtered_windows"] = filtered_windows
        filtered_metadata["filtered_dur"] = filtered_dur
        filtered_metadata["filtered_dur_list"] = filtered_dur_list
        filtered_metadata["manifest_filepath"] = manifest_filepath
        filtered_metadata["swift_filepath"] = swift_filepath
        return filtered_metadata
