# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
ALM Data Builder Stage - Native NeMo Curator Implementation.

Creates training windows from audio segments.
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
        'total_windows_output': 0,
        'total_duration_seconds': 0.0,
        'min_windows_per_entry': 0,
        'max_windows_per_entry': 0,
        # Loss tracking (matches SDP)
        'total_lost_bw': 0,
        'total_dur_lost_bw': 0.0,
        'total_lost_sr': 0,
        'total_dur_lost_sr': 0.0,
        'total_lost_spk': 0,
        'total_dur_lost_spk': 0.0,
        'total_lost_win': 0,
        'total_dur_lost_win': 0.0,
        'total_lost_no_spkr': 0,
        'total_dur_lost_no_spkr': 0.0,
        'total_lost_next_seg_bm': 0,
        'total_dur_lost_next_seg_bm': 0.0,
        'total_truncation_events': 0,
    }

    if not os.path.exists(output_file):
        logging.error(f"❌ Output file not found: {output_file}")
        return stats

    windows_per_entry = []
    total_duration = 0.0

    with open(output_file, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line.strip())
                    stats['total_entries'] += 1

                    windows = entry.get('windows', [])
                    stats['entries_with_windows'] += 1 if windows else 0
                    stats['total_windows_output'] += len(windows)
                    windows_per_entry.append(len(windows))

                    # Calculate duration from segments (SDP pattern - no 'duration' field)
                    for w in windows:
                        segs = w.get('segments', [])
                        if segs:
                            total_duration += segs[-1]['end'] - segs[0]['start']

                    # Aggregate loss stats from entry
                    entry_stats = entry.get('stats', {})
                    stats['total_lost_bw'] += entry_stats.get('lost_bw', 0)
                    stats['total_dur_lost_bw'] += entry_stats.get('dur_lost_bw', 0.0)
                    stats['total_lost_sr'] += entry_stats.get('lost_sr', 0)
                    stats['total_dur_lost_sr'] += entry_stats.get('dur_lost_sr', 0.0)
                    stats['total_lost_spk'] += entry_stats.get('lost_spk', 0)
                    stats['total_dur_lost_spk'] += entry_stats.get('dur_lost_spk', 0.0)
                    stats['total_lost_win'] += entry_stats.get('lost_win', 0)
                    stats['total_dur_lost_win'] += entry_stats.get('dur_lost_win', 0.0)
                    stats['total_lost_no_spkr'] += entry_stats.get('lost_no_spkr', 0)
                    stats['total_dur_lost_no_spkr'] += entry_stats.get('dur_lost_no_spkr', 0.0)
                    stats['total_lost_next_seg_bm'] += entry_stats.get('lost_next_seg_bm', 0)
                    stats['total_dur_lost_next_seg_bm'] += entry_stats.get('dur_lost_next_seg_bm', 0.0)
                    stats['total_truncation_events'] += entry.get('truncation_events', 0)

                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping malformed JSON line in {output_file}: {e}")
                    continue
                except (KeyError, TypeError) as e:
                    logging.warning(f"Skipping entry with missing/invalid fields in {output_file}: {e}")
                    continue

    stats['total_duration_seconds'] = total_duration
    if windows_per_entry:
        stats['min_windows_per_entry'] = min(windows_per_entry)
        stats['max_windows_per_entry'] = max(windows_per_entry)

    # Log comprehensive statistics
    logging.info("=" * 80)
    logging.info("📊 ALM DATA BUILDER - PROCESSING STATISTICS")
    logging.info("=" * 80)
    logging.info(f"  📁 Total entries processed: {stats['total_entries']:,}")
    logging.info(f"  ✅ Entries with windows: {stats['entries_with_windows']:,}")
    logging.info(f"  🪟 Total windows generated: {stats['total_windows_output']:,}")
    logging.info(f"  ⏱️  Total duration: {stats['total_duration_seconds']:.2f} seconds")
    if stats['total_entries'] > 0:
        avg_windows = stats['total_windows_output'] / stats['total_entries']
        logging.info(f"  📈 Average windows per entry: {avg_windows:.2f}")
    logging.info(f"  📉 Min/Max windows per entry: {stats['min_windows_per_entry']}/{stats['max_windows_per_entry']}")
    logging.info("")
    logging.info("  LOSS STATISTICS:")
    logging.info(f"    - Lost to bandwidth: {stats['total_lost_bw']:,} segs ({stats['total_dur_lost_bw']:.2f}s)")
    logging.info(f"    - Lost to sample rate: {stats['total_lost_sr']:,} segs ({stats['total_dur_lost_sr']:.2f}s)")
    logging.info(f"    - Lost to speaker count: {stats['total_lost_spk']:,} segs ({stats['total_dur_lost_spk']:.2f}s)")
    logging.info(f"    - Lost to window constraints: {stats['total_lost_win']:,} segs ({stats['total_dur_lost_win']:.2f}s)")
    logging.info(f"    - Lost to no-speaker: {stats['total_lost_no_spkr']:,} segs ({stats['total_dur_lost_no_spkr']:.2f}s)")
    logging.info(f"    - Lost to next seg bandwidth: {stats['total_lost_next_seg_bm']:,} segs ({stats['total_dur_lost_next_seg_bm']:.2f}s)")
    logging.info(f"    - Truncation events: {stats['total_truncation_events']:,}")
    logging.info("=" * 80)

    return stats


@dataclass
class ALMDataBuilderStage(LegacySpeechStage):
    """
    Build ALM training windows from audio segments.

    Native NeMo Curator stage that filters segments by sample rate,
    bandwidth, speaker count, and duration to create valid training windows.

    This follows the exact pattern from nemo_curator.stages.audio.common:
    - Inherits from LegacySpeechStage
    - Uses @dataclass decorator
    - Implements process_dataset_entry() method
    - Returns list[AudioBatch] from process_dataset_entry

    Produces identical output to SDP implementation.
    """

    # Processing parameters (EXACT match to SDP)
    target_window_duration: float = 120.0
    tolerance: float = 0.1
    min_bandwidth: int = 8000
    min_sample_rate: int = 16000
    min_speakers: int = 2
    max_speakers: int = 5
    truncation: bool = True

    # Fields to drop from output segments (comma-separated)
    # e.g., "words" removes word-level timestamps from each segment in windows
    drop_fields: str = "words"

    # Top-level fields to drop from output entry (comma-separated)
    # e.g., "words,segments" removes these columns from the output entry
    # All other input fields are preserved
    drop_fields_top_level: str = "words,segments"

    # Parallelism (used by runner, passed via config)
    max_workers: int = -1

    # Stage metadata (derived from class name if not set)
    name: str = None

    # Output directory for intermediate results (optional)
    # If set, writes stage output to {output_dir}/{name}_output.jsonl
    output_dir: str = None

    def __post_init__(self):
        """Compute derived parameters - EXACT match to SDP."""
        # Derive name from class name (e.g., ALMDataBuilderStage -> ALMDataBuilder)
        if self.name is None:
            self.name = self.__class__.__name__.replace("Stage", "")

        tol = self.target_window_duration * self.tolerance
        self.min_duration = self.target_window_duration - tol
        self.max_duration = self.target_window_duration + tol
        # Parse drop_fields into a set for fast lookup (for segment-level fields)
        self._drop_fields_set = set(f.strip() for f in self.drop_fields.split(',') if f.strip())
        # Parse drop_fields_top_level into a set (for entry-level fields)
        self._drop_fields_top_level_set = set(f.strip() for f in self.drop_fields_top_level.split(',') if f.strip())

    def process_dataset_entry(self, data_entry: dict) -> list[AudioBatch]:
        """
        Process a single manifest entry and build windows.
        EXACT match to SDP process_dataset_entry logic.

        Args:
            data_entry: Single entry from manifest (dict with audio_filepath, segments, etc.)

        Returns:
            list[AudioBatch] - Always returns entry (even with empty windows, matching SDP)
        """
        result = self._process_single_entry(data_entry)

        # Write to intermediate output file if output_dir is set
        # Uses self.name for file naming (e.g., "ALMDataBuilder" -> "ALMDataBuilder_output.jsonl")
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

        # SDP always outputs entries, even with empty windows
        return [AudioBatch(data=[result])]

    def _process_single_entry(self, entry_data: dict) -> dict:
        """
        Process a single entry and extract valid training windows.

        Returns dict with:
        - All input fields EXCEPT those in drop_fields_top_level (default: words, segments)
        - windows: list of valid training windows
        - stats: processing statistics
        - truncation_events: count of truncation events
        """
        # Initialize variables (EXACT match to SDP lines 210-219)
        total_truncation_events = 0

        audio_file = entry_data.get('audio_filepath')
        segments = entry_data.get('segments', [])
        total_dur = sum(seg['end'] - seg['start'] for seg in segments)

        # Initialize statistics for this audio file (EXACT match to SDP lines 222-234)
        stat = {
            'total_segments': len(segments),
            'total_dur': total_dur,
            'swift_path': entry_data.get('swift_audio_filepath', ''),
            'audio_sample_rate': entry_data.get('audio_sample_rate', 0),
            'lost_bw': 0, 'dur_lost_bw': 0.0,
            'lost_sr': 0, 'dur_lost_sr': 0.0,
            'lost_spk': 0, 'dur_lost_spk': 0.0,
            'lost_win': 0, 'dur_lost_win': 0.0,
            'lost_no_spkr': 0, 'dur_lost_no_spkr': 0.0,
            'lost_next_seg_bm': 0, 'dur_lost_next_seg_bm': 0.0,
            'lost_win_full_data': [],
        }

        # Check sample rate requirement (EXACT match to SDP lines 237-242)
        if entry_data.get('audio_sample_rate', 0) < self.min_sample_rate:
            stat['lost_sr'] = len(segments)
            stat['dur_lost_sr'] = total_dur
            # Return with empty windows (SDP pattern)
            return {
                'audio_filepath': audio_file,
                'windows': [],
                'stats': stat,
                'truncation_events': total_truncation_events
            }

        valid_windows = []

        # Process each segment as potential window start (EXACT match to SDP lines 246-381)
        for start_idx, seg in enumerate(segments):
            # Check bandwidth requirement (SDP lines 248-252)
            bandwidth = seg.get('metrics', {}).get('bandwidth', 0)
            if bandwidth < self.min_bandwidth:
                stat['lost_bw'] += 1
                stat['dur_lost_bw'] += seg['end'] - seg['start']
                continue

            # Build window starting from this segment (SDP lines 255-257)
            window_segs = []
            window_start = seg['start']
            window_end = seg['end']

            # Track curr_idx for later use in lost_win stats
            curr_idx = start_idx

            for curr_idx in range(start_idx, len(segments)):
                curr_seg = segments[curr_idx]

                # Check bandwidth (SDP lines 261-262)
                if curr_seg.get('metrics', {}).get('bandwidth', 0) < self.min_bandwidth:
                    break

                potential_end = curr_seg['end']
                potential_duration = potential_end - window_start

                # Check if adding this segment would exceed target duration (SDP lines 267-312)
                if potential_duration > self.max_duration:
                    if self.truncation:
                        truncated_end = window_start + self.max_duration
                        if curr_seg['start'] >= truncated_end:
                            break
                        if curr_seg.get('metrics', {}).get('bandwidth', 0) < self.min_bandwidth:
                            break

                        total_truncation_events += 1

                        part_curr_seg = curr_seg.copy()
                        # Filter and keep only words that fully end before the truncation point
                        truncated_words = []
                        actual_end = curr_seg['start']  # fallback in case no words selected

                        for w in curr_seg.get('words', []):
                            if w['end'] <= truncated_end:
                                truncated_words.append(w.copy())
                                actual_end = w['end']

                        # Update fields based on truncated words (SDP lines 288-293)
                        # NOTE: SDP line 289 has 'if w.get('words')' - matching exactly
                        part_curr_seg['words'] = truncated_words
                        words_text = [w.get('word', '') for w in truncated_words if w.get('word')]
                        truncated_text = ' '.join(words_text)
                        part_curr_seg['text'] = truncated_text
                        part_curr_seg['end'] = actual_end

                        # Check speaker count before adding truncated segment (SDP lines 296-305)
                        temp_window_segs = window_segs.copy()
                        temp_window_segs.append(part_curr_seg)
                        temp_spk_durs = {}
                        for s in temp_window_segs:
                            spk = s.get('speaker')
                            if spk:
                                temp_spk_durs[spk] = temp_spk_durs.get(spk, 0) + (s['end'] - s['start'])

                        if len(temp_spk_durs) > self.max_speakers or 'no-speaker' in temp_spk_durs.keys():
                            break

                        # Drop fields specified in drop_fields for truncated segment
                        window_segs.append({k: v for k, v in part_curr_seg.items() if k not in self._drop_fields_set})
                        window_end = actual_end
                        break
                    else:
                        break

                # Check speaker count before adding current segment (SDP lines 315-324)
                temp_window_segs = window_segs.copy()
                temp_window_segs.append(curr_seg)
                temp_spk_durs = {}
                for s in temp_window_segs:
                    spk = s.get('speaker')
                    if spk:
                        temp_spk_durs[spk] = temp_spk_durs.get(spk, 0) + (s['end'] - s['start'])

                if len(temp_spk_durs) > self.max_speakers or 'no-speaker' in temp_spk_durs.keys():
                    break

                window_end = curr_seg['end']
                # Drop fields specified in drop_fields (e.g., "words" removes word timestamps from segments)
                window_segs.append({k: v for k, v in curr_seg.items() if k not in self._drop_fields_set})

            window_dur = window_end - window_start

            # Check window duration constraints (SDP lines 332-343)
            if not (self.min_duration <= window_dur <= self.max_duration):
                stat['lost_win'] += 1
                stat['dur_lost_win'] += seg['end'] - seg['start']
                next_segment = segments[curr_idx]
                if next_segment.get('speaker', 'no-speaker') == 'no-speaker':
                    stat['lost_no_spkr'] += 1
                    stat['dur_lost_no_spkr'] += seg['end'] - seg['start']
                elif next_segment.get('metrics', {}).get('bandwidth', 0) < self.min_bandwidth:
                    stat['lost_next_seg_bm'] += 1
                    stat['dur_lost_next_seg_bm'] += seg['end'] - seg['start']
                stat['lost_win_full_data'].append({
                    'index': start_idx,
                    'window_segs': window_segs,
                    'next_seg': {k: v for k, v in next_segment.items() if k not in self._drop_fields_set},
                    'prev_seg': {k: v for k, v in segments[max(start_idx - 1, 0)].items() if k not in self._drop_fields_set}
                })
                continue

            # Check minimum segments and bandwidth for all segments in window (SDP lines 346-357)
            if len(window_segs) < 2 or any(s.get('metrics', {}).get('bandwidth', 0) < self.min_bandwidth for s in window_segs):
                stat['lost_win'] += 1
                stat['dur_lost_win'] += seg['end'] - seg['start']
                next_segment = segments[curr_idx]
                if next_segment.get('speaker', 'no-speaker') == 'no-speaker':
                    stat['lost_no_spkr'] += 1
                    stat['dur_lost_no_spkr'] += seg['end'] - seg['start']
                elif next_segment.get('metrics', {}).get('bandwidth', 0) < self.min_bandwidth:
                    stat['lost_next_seg_bm'] += 1
                    stat['dur_lost_next_seg_bm'] += seg['end'] - seg['start']
                stat['lost_win_full_data'].append({
                    'index': start_idx,
                    'window_segs': window_segs,
                    'next_seg': {k: v for k, v in next_segment.items() if k not in self._drop_fields_set},
                    'prev_seg': {k: v for k, v in segments[max(start_idx - 1, 0)].items() if k not in self._drop_fields_set}
                })
                continue

            # Calculate speaker durations (SDP lines 360-364)
            spk_durs = {}
            for s in window_segs:
                spk = s.get('speaker')
                if spk:
                    spk_durs[spk] = spk_durs.get(spk, 0) + (s['end'] - s['start'])

            # Check speaker count requirement (SDP lines 367-370)
            if not self.min_speakers <= len(spk_durs) <= self.max_speakers or 'no-speaker' in spk_durs.keys():
                stat['lost_spk'] += 1
                stat['dur_lost_spk'] += seg['end'] - seg['start']
                continue

            # Create window (SDP lines 374-381 - NO 'duration' field)
            spk_durations = sorted(spk_durs.values(), reverse=True)[:5]
            spk_durations += [0.0] * (5 - len(spk_durations))

            window_info = {
                "segments": window_segs,
                "speaker_durations": spk_durations
            }
            valid_windows.append(window_info)

        # Build result: keep all input fields EXCEPT those in drop_fields_top_level
        result = {k: v for k, v in entry_data.items() if k not in self._drop_fields_top_level_set}

        # Add computed fields
        result['windows'] = valid_windows
        result['stats'] = stat
        result['truncation_events'] = total_truncation_events

        return result
