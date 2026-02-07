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

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from nemo_curator.stages.audio.common import LegacySpeechStage
from nemo_curator.tasks import AudioBatch

# Constants for validation
MIN_SEGMENTS_PER_WINDOW = 2


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
    drop_fields: str = "words"

    # Top-level fields to drop from output entry (comma-separated)
    drop_fields_top_level: str = "words,segments"

    # Parallelism (used by runner, passed via config)
    max_workers: int = -1

    # Stage metadata
    name: str = "alm_data_builder"

    def __post_init__(self) -> None:
        """Compute derived parameters - EXACT match to SDP."""

        tol = self.target_window_duration * self.tolerance
        self.min_duration = self.target_window_duration - tol
        self.max_duration = self.target_window_duration + tol
        self._drop_fields_set = {f.strip() for f in self.drop_fields.split(",") if f.strip()}
        self._drop_fields_top_level_set = {f.strip() for f in self.drop_fields_top_level.split(",") if f.strip()}

    def process_dataset_entry(self, data_entry: dict[str, Any]) -> list[AudioBatch]:
        """
        Process a single manifest entry and build windows.

        Args:
            data_entry: Single entry from manifest (dict with audio_filepath, segments, etc.)

        Returns:
            list[AudioBatch] - Always returns entry (even with empty windows, matching SDP)
        """
        t0 = time.perf_counter()
        result = self._process_single_entry(data_entry)
        process_time = time.perf_counter() - t0

        # Log timing metrics for regression tracking
        num_segments = len(data_entry.get("segments", []))
        num_windows = len(result.get("windows", []))
        self._log_metrics(
            {
                "process_entry_time": process_time,
                "segments_processed": num_segments,
                "windows_created": num_windows,
            }
        )

        return [AudioBatch(data=[result])]

    def _process_single_entry(  # noqa: C901, PLR0912, PLR0915
        self, entry_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Process a single entry and extract valid training windows.

        Note: This method is complex (C901, PLR0912, PLR0915) because it follows
        the exact SDP implementation logic for bit-exact compatibility.
        Refactoring would risk behavioral differences.

        Returns dict with:
        - All input fields EXCEPT those in drop_fields_top_level
        - windows: list of valid training windows
        - stats: processing statistics
        - truncation_events: count of truncation events
        """
        total_truncation_events = 0

        audio_file = entry_data.get("audio_filepath")
        segments = entry_data.get("segments", [])
        total_dur = sum(seg["end"] - seg["start"] for seg in segments)

        stat: dict[str, Any] = {
            "total_segments": len(segments),
            "total_dur": total_dur,
            "swift_path": entry_data.get("swift_audio_filepath", ""),
            "audio_sample_rate": entry_data.get("audio_sample_rate", 0),
            "lost_bw": 0,
            "dur_lost_bw": 0.0,
            "lost_sr": 0,
            "dur_lost_sr": 0.0,
            "lost_spk": 0,
            "dur_lost_spk": 0.0,
            "lost_win": 0,
            "dur_lost_win": 0.0,
            "lost_no_spkr": 0,
            "dur_lost_no_spkr": 0.0,
            "lost_next_seg_bm": 0,
            "dur_lost_next_seg_bm": 0.0,
            "lost_win_full_data": [],
        }

        if entry_data.get("audio_sample_rate", 0) < self.min_sample_rate:
            stat["lost_sr"] = len(segments)
            stat["dur_lost_sr"] = total_dur
            return {
                "audio_filepath": audio_file,
                "windows": [],
                "stats": stat,
                "truncation_events": total_truncation_events,
            }

        valid_windows: list[dict[str, Any]] = []

        for start_idx, seg in enumerate(segments):
            bandwidth = seg.get("metrics", {}).get("bandwidth", 0)
            if bandwidth < self.min_bandwidth:
                stat["lost_bw"] += 1
                stat["dur_lost_bw"] += seg["end"] - seg["start"]
                continue

            window_segs: list[dict[str, Any]] = []
            window_start = seg["start"]
            window_end = seg["end"]
            curr_idx = start_idx

            for curr_idx in range(start_idx, len(segments)):
                curr_seg = segments[curr_idx]

                if curr_seg.get("metrics", {}).get("bandwidth", 0) < self.min_bandwidth:
                    break

                potential_end = curr_seg["end"]
                potential_duration = potential_end - window_start

                if potential_duration > self.max_duration:
                    if self.truncation:
                        truncated_end = window_start + self.max_duration
                        if curr_seg["start"] >= truncated_end:
                            break
                        if curr_seg.get("metrics", {}).get("bandwidth", 0) < self.min_bandwidth:
                            break

                        total_truncation_events += 1

                        part_curr_seg = curr_seg.copy()
                        truncated_words = []
                        actual_end = curr_seg["start"]

                        for w in curr_seg.get("words", []):
                            if w["end"] <= truncated_end:
                                truncated_words.append(w.copy())
                                actual_end = w["end"]

                        part_curr_seg["words"] = truncated_words
                        words_text = [w.get("word", "") for w in truncated_words if w.get("word")]
                        truncated_text = " ".join(words_text)
                        part_curr_seg["text"] = truncated_text
                        part_curr_seg["end"] = actual_end

                        temp_window_segs = [*window_segs, part_curr_seg]
                        temp_spk_durs: dict[str, float] = {}
                        for s in temp_window_segs:
                            spk = s.get("speaker")
                            if spk:
                                temp_spk_durs[spk] = temp_spk_durs.get(spk, 0) + (s["end"] - s["start"])

                        if len(temp_spk_durs) > self.max_speakers or "no-speaker" in temp_spk_durs:
                            break

                        window_segs.append({k: v for k, v in part_curr_seg.items() if k not in self._drop_fields_set})
                        window_end = actual_end
                        break
                    else:
                        break

                temp_window_segs = [*window_segs, curr_seg]
                temp_spk_durs = {}
                for s in temp_window_segs:
                    spk = s.get("speaker")
                    if spk:
                        temp_spk_durs[spk] = temp_spk_durs.get(spk, 0) + (s["end"] - s["start"])

                if len(temp_spk_durs) > self.max_speakers or "no-speaker" in temp_spk_durs:
                    break

                window_end = curr_seg["end"]
                window_segs.append({k: v for k, v in curr_seg.items() if k not in self._drop_fields_set})

            window_dur = window_end - window_start

            if not (self.min_duration <= window_dur <= self.max_duration):
                stat["lost_win"] += 1
                stat["dur_lost_win"] += seg["end"] - seg["start"]
                # Bounds check: curr_idx may equal len(segments) if loop completed without break
                next_seg_idx = min(curr_idx, len(segments) - 1)
                next_segment = segments[next_seg_idx]
                if next_segment.get("speaker", "no-speaker") == "no-speaker":
                    stat["lost_no_spkr"] += 1
                    stat["dur_lost_no_spkr"] += seg["end"] - seg["start"]
                elif next_segment.get("metrics", {}).get("bandwidth", 0) < self.min_bandwidth:
                    stat["lost_next_seg_bm"] += 1
                    stat["dur_lost_next_seg_bm"] += seg["end"] - seg["start"]
                stat["lost_win_full_data"].append(
                    {
                        "index": start_idx,
                        "window_segs": window_segs,
                        "next_seg": {k: v for k, v in next_segment.items() if k not in self._drop_fields_set},
                        "prev_seg": {
                            k: v for k, v in segments[max(start_idx - 1, 0)].items() if k not in self._drop_fields_set
                        },
                    }
                )
                continue

            if len(window_segs) < MIN_SEGMENTS_PER_WINDOW or any(
                s.get("metrics", {}).get("bandwidth", 0) < self.min_bandwidth for s in window_segs
            ):
                stat["lost_win"] += 1
                stat["dur_lost_win"] += seg["end"] - seg["start"]
                # Bounds check: curr_idx may equal len(segments) if loop completed without break
                next_seg_idx = min(curr_idx, len(segments) - 1)
                next_segment = segments[next_seg_idx]
                if next_segment.get("speaker", "no-speaker") == "no-speaker":
                    stat["lost_no_spkr"] += 1
                    stat["dur_lost_no_spkr"] += seg["end"] - seg["start"]
                elif next_segment.get("metrics", {}).get("bandwidth", 0) < self.min_bandwidth:
                    stat["lost_next_seg_bm"] += 1
                    stat["dur_lost_next_seg_bm"] += seg["end"] - seg["start"]
                stat["lost_win_full_data"].append(
                    {
                        "index": start_idx,
                        "window_segs": window_segs,
                        "next_seg": {k: v for k, v in next_segment.items() if k not in self._drop_fields_set},
                        "prev_seg": {
                            k: v for k, v in segments[max(start_idx - 1, 0)].items() if k not in self._drop_fields_set
                        },
                    }
                )
                continue

            spk_durs: dict[str, float] = {}
            for s in window_segs:
                spk = s.get("speaker")
                if spk:
                    spk_durs[spk] = spk_durs.get(spk, 0) + (s["end"] - s["start"])

            if not self.min_speakers <= len(spk_durs) <= self.max_speakers or "no-speaker" in spk_durs:
                stat["lost_spk"] += 1
                stat["dur_lost_spk"] += seg["end"] - seg["start"]
                continue

            spk_durations = sorted(spk_durs.values(), reverse=True)[:5]
            spk_durations += [0.0] * (5 - len(spk_durations))

            window_info: dict[str, Any] = {
                "segments": window_segs,
                "speaker_durations": spk_durations,
            }
            valid_windows.append(window_info)

        result = {k: v for k, v in entry_data.items() if k not in self._drop_fields_top_level_set}
        result["windows"] = valid_windows
        result["stats"] = stat
        result["truncation_events"] = total_truncation_events

        return result
