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

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from nemo_curator.stages.audio.common import LegacySpeechStage
from nemo_curator.tasks import AudioBatch

MIN_SEGMENTS_PER_WINDOW = 2


@dataclass
class BuilderStats:
    """Tracks segment loss reasons and counts during window building."""

    total_segments: int = 0
    total_dur: float = 0.0
    swift_path: str = ""
    audio_sample_rate: int = 0
    lost_bw: int = 0
    dur_lost_bw: float = 0.0
    lost_sr: int = 0
    dur_lost_sr: float = 0.0
    lost_spk: int = 0
    dur_lost_spk: float = 0.0
    lost_win: int = 0
    dur_lost_win: float = 0.0
    lost_no_spkr: int = 0
    dur_lost_no_spkr: float = 0.0
    lost_next_seg_bm: int = 0
    dur_lost_next_seg_bm: float = 0.0
    lost_win_full_data: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_bandwidth(seg: dict[str, Any]) -> int:
    return seg.get("metrics", {}).get("bandwidth", 0)


def _compute_speaker_durations(segments: list[dict[str, Any]]) -> dict[str, float]:
    spk_durs: dict[str, float] = {}
    for s in segments:
        spk = s.get("speaker")
        if spk:
            spk_durs[spk] = spk_durs.get(spk, 0) + (s["end"] - s["start"])
    return spk_durs


def _truncate_segment(seg: dict[str, Any], truncated_end: float) -> dict[str, Any]:
    """Truncate a segment's words at the given end time, returning a modified copy."""
    part = seg.copy()
    truncated_words = []
    actual_end = seg["start"]

    for w in seg.get("words", []):
        if w["end"] <= truncated_end:
            truncated_words.append(w.copy())
            actual_end = w["end"]

    part["words"] = truncated_words
    part["text"] = " ".join(w.get("word", "") for w in truncated_words if w.get("word"))
    part["end"] = actual_end
    return part


def _record_window_loss(
    stat: BuilderStats,
    seg: dict[str, Any],
    segments: list[dict[str, Any]],
    start_idx: int,
    curr_idx: int,
    window_segs: list[dict[str, Any]],
    drop_fields: set[str],
) -> None:
    """Record statistics for a rejected window."""
    seg_dur = seg["end"] - seg["start"]
    stat.lost_win += 1
    stat.dur_lost_win += seg_dur

    next_seg_idx = min(curr_idx, len(segments) - 1)
    next_segment = segments[next_seg_idx]

    if next_segment.get("speaker", "no-speaker") == "no-speaker":
        stat.lost_no_spkr += 1
        stat.dur_lost_no_spkr += seg_dur
    elif _get_bandwidth(next_segment) < 8000:
        stat.lost_next_seg_bm += 1
        stat.dur_lost_next_seg_bm += seg_dur

    stat.lost_win_full_data.append(
        {
            "index": start_idx,
            "window_segs": window_segs,
            "next_seg": {k: v for k, v in next_segment.items() if k not in drop_fields},
            "prev_seg": {k: v for k, v in segments[max(start_idx - 1, 0)].items() if k not in drop_fields},
        }
    )


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

    def process(self, task: AudioBatch) -> list[AudioBatch]:
        """Process a batch, propagating parent task metadata and perf stats."""
        results = []
        for entry in task.data:
            for child in self.process_dataset_entry(entry):
                child._metadata = task._metadata.copy()
                child._stage_perf = task._stage_perf.copy()
                results.append(child)
        return results

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

    def _process_single_entry(self, entry_data: dict[str, Any]) -> dict[str, Any]:
        """Process a single entry and extract valid training windows."""
        total_truncation_events = 0

        audio_file = entry_data.get("audio_filepath")
        segments = entry_data.get("segments", [])
        total_dur = sum(seg["end"] - seg["start"] for seg in segments)

        stat = BuilderStats(
            total_segments=len(segments),
            total_dur=total_dur,
            swift_path=entry_data.get("swift_audio_filepath", ""),
            audio_sample_rate=entry_data.get("audio_sample_rate", 0),
        )

        if entry_data.get("audio_sample_rate", 0) < self.min_sample_rate:
            stat.lost_sr = len(segments)
            stat.dur_lost_sr = total_dur
            return {
                "audio_filepath": audio_file,
                "windows": [],
                "stats": stat.to_dict(),
                "truncation_events": total_truncation_events,
            }

        valid_windows: list[dict[str, Any]] = []

        for start_idx, seg in enumerate(segments):
            if _get_bandwidth(seg) < self.min_bandwidth:
                stat.lost_bw += 1
                stat.dur_lost_bw += seg["end"] - seg["start"]
                continue

            window_segs: list[dict[str, Any]] = []
            window_start = seg["start"]
            window_end = seg["end"]
            curr_idx = start_idx

            for curr_idx in range(start_idx, len(segments)):
                curr_seg = segments[curr_idx]

                if _get_bandwidth(curr_seg) < self.min_bandwidth:
                    break

                potential_duration = curr_seg["end"] - window_start

                if potential_duration > self.max_duration:
                    if not self.truncation:
                        break
                    truncated_end = window_start + self.max_duration
                    if curr_seg["start"] >= truncated_end:
                        break
                    if _get_bandwidth(curr_seg) < self.min_bandwidth:
                        break

                    total_truncation_events += 1
                    part = _truncate_segment(curr_seg, truncated_end)

                    spk_durs = _compute_speaker_durations([*window_segs, part])
                    if len(spk_durs) > self.max_speakers or "no-speaker" in spk_durs:
                        break

                    window_segs.append({k: v for k, v in part.items() if k not in self._drop_fields_set})
                    window_end = part["end"]
                    break

                spk_durs = _compute_speaker_durations([*window_segs, curr_seg])
                if len(spk_durs) > self.max_speakers or "no-speaker" in spk_durs:
                    break

                window_end = curr_seg["end"]
                window_segs.append({k: v for k, v in curr_seg.items() if k not in self._drop_fields_set})

            window_dur = window_end - window_start

            if not (self.min_duration <= window_dur <= self.max_duration):
                _record_window_loss(stat, seg, segments, start_idx, curr_idx, window_segs, self._drop_fields_set)
                continue

            if len(window_segs) < MIN_SEGMENTS_PER_WINDOW or any(
                _get_bandwidth(s) < self.min_bandwidth for s in window_segs
            ):
                _record_window_loss(stat, seg, segments, start_idx, curr_idx, window_segs, self._drop_fields_set)
                continue

            spk_durs = _compute_speaker_durations(window_segs)
            if not self.min_speakers <= len(spk_durs) <= self.max_speakers or "no-speaker" in spk_durs:
                stat.lost_spk += 1
                stat.dur_lost_spk += seg["end"] - seg["start"]
                continue

            spk_durations = sorted(spk_durs.values(), reverse=True)[:5]
            spk_durations += [0.0] * (5 - len(spk_durations))

            valid_windows.append({
                "segments": window_segs,
                "speaker_durations": spk_durations,
            })

        result = {k: v for k, v in entry_data.items() if k not in self._drop_fields_top_level_set}
        result["windows"] = valid_windows
        result["stats"] = stat.to_dict()
        result["truncation_events"] = total_truncation_events

        return result
