# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
ALM Data Overlap Stage - Native NeMo Curator Implementation.

Filters overlapping windows based on threshold.
Follows the exact pattern from NeMo Curator:
https://github.com/NVIDIA-NeMo/Curator/blob/main/nemo_curator/stages/audio/common.py

Produces identical output to SDP implementation.
"""

import json
import logging
import os
from dataclasses import dataclass

from nemo_curator.stages.audio.common import LegacySpeechStage
from nemo_curator.tasks import AudioBatch


def log_output_statistics(output_file: str) -> dict:
    """
    Log cumulative statistics by reading the output file.

    This follows the SDP pattern where finalize() reads
    the output manifest to calculate and log comprehensive statistics.

    Args:
        output_file: Path to the output JSONL file

    Returns:
        dict with statistics
    """
    stats = {
        'total_entries': 0,
        'entries_with_windows': 0,
        'entries_filtered': 0,
        'total_windows_input': 0,
        'total_windows_output': 0,
        'total_duration_seconds': 0.0,
        'processing_errors': 0,
    }

    if not os.path.exists(output_file):
        logging.error(f"❌ Output file not found: {output_file}")
        return stats

    total_duration = 0.0

    with open(output_file) as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line.strip())
                    stats['total_entries'] += 1

                    windows = entry.get('windows', [])
                    filtered_windows = entry.get('filtered_windows', [])
                    filtered_dur = entry.get('filtered_dur', 0)

                    if windows:
                        stats['entries_with_windows'] += 1
                        # Fall back to len(windows) if total_dur_list_window is absent
                        # (handles entries returned unchanged when no filtered windows pass)
                        total_dur_list = entry.get('total_dur_list_window', [])
                        stats['total_windows_input'] += len(total_dur_list) if total_dur_list else len(windows)

                    if filtered_windows:
                        stats['entries_filtered'] += 1
                        stats['total_windows_output'] += len(filtered_windows)
                        total_duration += filtered_dur

                except json.JSONDecodeError:
                    stats['processing_errors'] += 1
                    continue

    stats['total_duration_seconds'] = total_duration

    # Calculate derived stats
    filter_rate = (stats['entries_filtered'] / max(stats['entries_with_windows'], 1)) * 100
    window_reduction = stats['total_windows_input'] - stats['total_windows_output']

    # Log comprehensive statistics
    logging.info("=" * 80)
    logging.info("📊 ALM DATA OVERLAP - PROCESSING STATISTICS")
    logging.info("=" * 80)
    logging.info(f"  📁 Total entries processed: {stats['total_entries']:,}")
    logging.info(f"  ✅ Entries with windows: {stats['entries_with_windows']:,}")
    logging.info(f"  🔄 Entries filtered: {stats['entries_filtered']:,} ({filter_rate:.1f}%)")
    logging.info(f"  📥 Input windows: {stats['total_windows_input']:,}")
    logging.info(f"  📤 Output windows: {stats['total_windows_output']:,}")
    logging.info(f"  🗑️  Windows removed: {window_reduction:,}")
    logging.info(f"  ⏱️  Total duration: {stats['total_duration_seconds']:.2f} seconds")
    logging.info(f"  ❌ Processing errors: {stats['processing_errors']:,}")
    logging.info("=" * 80)

    return stats


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

    # Stage metadata (derived from class name if not set)
    name: str = None

    # Output directory for intermediate results (optional)
    # If set, writes stage output to {output_dir}/{name}_output.jsonl
    output_dir: str = None

    def __post_init__(self):
        """Validate parameters and derive name."""
        # Derive name from class name (e.g., ALMDataOverlapStage -> ALMDataOverlap)
        if self.name is None:
            self.name = self.__class__.__name__.replace("Stage", "")

        if not (0 <= self.overlap_percentage <= 100):
            raise ValueError(f"overlap_percentage must be 0-100, got {self.overlap_percentage}")
        if self.target_duration <= 0:
            raise ValueError(f"target_duration must be positive")

    # ========================================
    # SDP HELPER METHODS
    # (EXACT copies from generic_sdp/processors/alm_data_overlap.py)
    # ========================================

    def calculate_total_dur(self, x):
        """Calculate total duration from windows data. EXACT match to SDP line 99-104."""
        try:
            return sum(seg[-1]['end'] - seg[0]['start'] for window in x for seg in [window.get('segments', [])] if seg)
        except Exception:
            return 0

    def calculate_duration_list(self, x):
        """Calculate list of durations from windows data. EXACT match to SDP line 106-111."""
        try:
            return [seg[-1]['end'] - seg[0]['start'] for window in x for seg in [window.get('segments', [])] if seg]
        except Exception:
            return []

    def calculate_timestamps(self, x):
        """Calculate timestamp pairs from windows data. EXACT match to SDP line 113-118."""
        try:
            return [(seg[-1]['end'], seg[0]['start']) for window in x for seg in [window.get('segments', [])] if seg]
        except Exception:
            return []

    def overlap_ratio(self, seg1, seg2):
        """Calculate overlap ratio between two segments. EXACT match to SDP line 120-130."""
        start1, end1 = seg1[1], seg1[0]
        start2, end2 = seg2[1], seg2[0]
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_duration = max(0, overlap_end - overlap_start)
        duration1 = end1 - start1
        duration2 = end2 - start2
        smaller_duration = min(duration1, duration2)
        return overlap_duration / smaller_duration if smaller_duration else 0

    def filter_segments(self, segments, threshold=0.9):
        """Filter out segments that have overlap greater than threshold. EXACT match to SDP line 132-172."""
        sorted_segs = sorted(segments, key=lambda x: (x[1], x[0]))
        remove_indices = set()

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
                    else:
                        # If absolute difference is same, keep the longer segment
                        if dur_i >= dur_j:
                            remove_indices.add(j)
                        else:
                            remove_indices.add(i)
                            break

        return [seg for idx, seg in enumerate(sorted_segs) if idx not in remove_indices]

    def process_filtered(self, x):
        """Get total duration of qualified segments. EXACT match to SDP line 174-180."""
        total = 0
        for end, start in x:
            dur = end - start
            total += dur
        return total

    def process_filtered_list(self, x):
        """Get duration lists of qualified segments. EXACT match to SDP line 182-188."""
        result = []
        for end, start in x:
            dur = end - start
            result.append(dur)
        return result

    def get_manifest_filepath(self, x):
        """Extract manifest filepath from stats. EXACT match to SDP line 190-192."""
        return x.get('manifest_path') if isinstance(x, dict) else None

    def get_swift_filepath(self, x):
        """Extract Swift filepath from stats. EXACT match to SDP line 194-196."""
        return x.get('swift_path') if isinstance(x, dict) else None

    def get_filtered_windows(self, windows, filtered_timestamps):
        """
        Get complete window objects that correspond to filtered timestamps.
        EXACT match to SDP line 198-232.
        """
        if not windows or not filtered_timestamps:
            return []

        # Convert filtered timestamps to set with 6-digit rounding for fast lookup
        filtered_set = set((round(end, 6), round(start, 6)) for end, start in filtered_timestamps)

        # Find windows that match the filtered timestamps
        filtered_windows = []
        for window in windows:
            segments = window.get('segments', [])
            if not segments:
                continue

            # Calculate timestamp pair for this window (same logic as calculate_timestamps)
            window_end = segments[-1]['end']
            window_start = segments[0]['start']
            # Round both end and start to 6 decimal places for comparison
            window_timestamp = (round(window_end, 6), round(window_start, 6))

            # If this window's rounded timestamp is in the filtered set, include the complete window
            if window_timestamp in filtered_set:
                filtered_windows.append(window)

        return filtered_windows

    # ========================================
    # MAIN PROCESSING METHODS
    # ========================================

    def process_dataset_entry(self, data_entry: dict) -> list[AudioBatch]:
        """
        Process a single manifest entry and filter overlapping windows.
        EXACT match to SDP process_dataset_entry logic.

        Args:
            data_entry: Single entry from manifest (dict with windows, etc.)

        Returns:
            list[AudioBatch] - Always returns entry (matching SDP behavior)
        """
        result = self._filter_overlaps(data_entry)

        # Write to intermediate output file if output_dir is set
        # Uses self.name for file naming (e.g., "ALMDataOverlap" -> "ALMDataOverlap_output.jsonl")
        if self.output_dir:
            import json
            import os
            import fcntl
            os.makedirs(self.output_dir, exist_ok=True)
            output_file = os.path.join(self.output_dir, f"{self.name}_output.jsonl")
            with open(output_file, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

        # SDP always outputs entries
        return [AudioBatch(data=[result])]

    def _filter_overlaps(self, entry: dict) -> dict:
        """
        Filter overlapping windows from entry.
        EXACT match to SDP ALMDataOverlapProcessor.process_dataset_entry lines 234-315
        """
        threshold = self.overlap_percentage / 100.0

        # Get actual data (in NeMo Curator, entry is already a dict)
        actual_data = entry

        windows = actual_data.get('windows', [])

        # If no windows, return original data unchanged (SDP lines 261-264)
        if not windows:
            return actual_data

        # Use exact SDP helper functions for processing (SDP lines 270-272)
        total_dur_window = self.calculate_total_dur(windows)
        total_dur_list_window = self.calculate_duration_list(windows)
        total_dur_list_window_timestamps = self.calculate_timestamps(windows)

        # Apply filtering with current threshold using exact SDP function (SDP line 275)
        filtered_timestamps = self.filter_segments(total_dur_list_window_timestamps, threshold=threshold)

        # Get filtered windows using exact SDP function (SDP line 278)
        filtered_windows = self.get_filtered_windows(windows, filtered_timestamps)

        # Calculate filtered metrics using exact SDP functions (SDP lines 281-282)
        filtered_dur = self.process_filtered(filtered_timestamps)
        filtered_dur_list = self.process_filtered_list(filtered_timestamps)

        # Extract manifest/swift filepaths using exact SDP functions (SDP lines 285-287)
        stats = actual_data.get('stats', {})
        manifest_filepath = self.get_manifest_filepath(stats)
        swift_filepath = self.get_swift_filepath(stats)

        if filtered_windows:
            # Create filtered record with the same structure as SDP (lines 291-307)
            filtered_metadata = actual_data.copy()
            filtered_metadata['windows'] = filtered_windows
            filtered_metadata['total_dur_window'] = total_dur_window
            filtered_metadata['total_dur_list_window'] = total_dur_list_window
            filtered_metadata['total_dur_list_window_timestamps'] = total_dur_list_window_timestamps
            filtered_metadata['filtered'] = filtered_timestamps
            filtered_metadata['filtered_windows'] = filtered_windows
            filtered_metadata['filtered_dur'] = filtered_dur
            filtered_metadata['filtered_dur_list'] = filtered_dur_list
            filtered_metadata['manifest_filepath'] = manifest_filepath
            filtered_metadata['swift_filepath'] = swift_filepath
            return filtered_metadata
        else:
            # No windows passed final filtering - return original data (SDP lines 308-310)
            return actual_data
