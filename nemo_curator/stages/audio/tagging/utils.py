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

from typing import Any


def add_non_speaker_segments(
    segments: list[dict[str, Any]],
    audio_duration: float,
    max_length: float | None = None,
) -> None:
    """Add non-speaker segments to the segments list with speaker id 'no-speaker'.

    If max_length is provided, splits non-speaker regions into chunks of that length;
    otherwise adds one segment per gap. Modifies segments in-place and sorts by start time.

    Args:
        segments: List of segment dicts with 'start' and 'end'.
        audio_duration: Total audio duration in seconds.
        max_length: Optional max length for each non-speaker segment.
    """
    non_speaker_segments = []
    last_end_time = 0
    for seg in sorted(segments, key=lambda s: s["start"]):
        start = seg["start"]
        end = seg["end"]
        if start > last_end_time:
            non_speaker_segments.append((last_end_time, start))
        last_end_time = end

    if last_end_time < audio_duration:
        non_speaker_segments.append((last_end_time, audio_duration))

    for start, end in non_speaker_segments:
        speaker_id = "no-speaker"
        if max_length is not None:
            current_start = start
            while current_start < end:
                current_end = min(current_start + max_length, end)
                segment_data_entry = {
                    "speaker": speaker_id,
                    "start": current_start,
                    "end": current_end,
                }
                segments.append(segment_data_entry)
                current_start = current_end
        else:
            segment_data_entry = {
                "speaker": speaker_id,
                "start": start,
                "end": end,
            }
            segments.append(segment_data_entry)

    segments.sort(key=lambda x: x["start"])
