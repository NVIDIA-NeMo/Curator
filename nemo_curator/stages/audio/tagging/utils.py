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

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nemo_curator.tasks import AudioTask

_REQUIRED_SEGMENT_FIELDS = frozenset({"speaker", "start", "end", "text", "words", "metrics"})
_REQUIRED_WORD_FIELDS = frozenset({"word", "start", "end"})


def _finite_float(value: object, description: str) -> float:
    """Convert a numeric output field to a finite float."""
    try:
        number = float(value)
    except (TypeError, ValueError) as e:
        msg = f"{description} must be a finite number, got {value!r}"
        raise RuntimeError(msg) from e
    if not math.isfinite(number):
        msg = f"{description} must be a finite number, got {value!r}"
        raise RuntimeError(msg)
    return number


def _validate_prepared_words(words: object, location: str) -> None:
    """Validate the word list for one prepared segment."""
    if not isinstance(words, list) or not words:
        msg = f"{location} must contain a non-empty words list"
        raise TypeError(msg)
    for word_index, word in enumerate(words):
        word_location = f"{location} word {word_index}"
        if not isinstance(word, Mapping):
            msg = f"{word_location} must be a mapping, got {type(word).__name__}"
            raise TypeError(msg)
        missing_word_fields = _REQUIRED_WORD_FIELDS.difference(word)
        if missing_word_fields:
            msg = f"{word_location} is missing required fields: {', '.join(sorted(missing_word_fields))}"
            raise RuntimeError(msg)
        if not isinstance(word["word"], str):
            msg = f"{word_location} has invalid text: {word['word']!r}"
            raise TypeError(msg)
        word_start = _finite_float(word["start"], f"{word_location} start")
        word_end = _finite_float(word["end"], f"{word_location} end")
        if word_start < 0 or word_end <= word_start:
            msg = f"{word_location} must satisfy 0 <= start < end, got {word_start} and {word_end}"
            raise RuntimeError(msg)


def _validate_prepared_segment(segment: object, task_index: int, segment_index: int) -> float:
    """Validate one PrepareModuleSegments output and return its duration."""
    location = f"Audio tagging output task {task_index} segment {segment_index}"
    if not isinstance(segment, Mapping):
        msg = f"{location} must be a mapping, got {type(segment).__name__}"
        raise TypeError(msg)

    missing_fields = _REQUIRED_SEGMENT_FIELDS.difference(segment)
    if missing_fields:
        msg = f"{location} is missing required fields: {', '.join(sorted(missing_fields))}"
        raise RuntimeError(msg)

    speaker = segment["speaker"]
    if not isinstance(speaker, str) or not speaker.strip():
        msg = f"{location} has an invalid speaker: {speaker!r}"
        raise TypeError(msg)

    text = segment["text"]
    if not isinstance(text, str) or not text.strip():
        msg = f"{location} has invalid text: {text!r}"
        raise TypeError(msg)

    _validate_prepared_words(segment["words"], location)

    if not isinstance(segment["metrics"], Mapping):
        msg = f"{location} metrics must be a mapping"
        raise TypeError(msg)

    start = _finite_float(segment["start"], f"{location} start")
    end = _finite_float(segment["end"], f"{location} end")
    if start < 0 or end <= start:
        msg = f"{location} must satisfy 0 <= start < end, got {start} and {end}"
        raise RuntimeError(msg)
    return end - start


def validate_tagging_outputs(tasks: Sequence[AudioTask] | None) -> dict[str, int | float]:
    """Validate final audio-tagging tasks and return stable output metrics.

    A tagging run is useful only when it emits tasks with positive-duration audio
    and at least one prepared segment. Every counted segment must match the
    ``PrepareModuleSegmentsStage`` contract: speaker, finite ordered timestamps,
    non-empty text and word entries, and a metrics mapping. Individual short clips
    may legitimately contain no prepared segments, so the segment requirement
    applies to the run as a whole.
    """
    output_tasks = list(tasks or [])
    if not output_tasks:
        msg = "Audio tagging pipeline produced no output tasks"
        raise RuntimeError(msg)

    total_duration_s = 0.0
    tagged_duration_s = 0.0
    num_tasks_with_segments = 0
    num_segments_processed = 0

    for index, task in enumerate(output_tasks):
        duration = task.data.get("duration")
        duration_s = _finite_float(duration, f"Audio tagging output task {index} duration")
        if duration_s <= 0:
            msg = f"Audio tagging output task {index} has non-positive duration: {duration!r}"
            raise RuntimeError(msg)
        total_duration_s += duration_s

        segments = task.data.get("segments")
        if not isinstance(segments, list):
            msg = f"Audio tagging output task {index} is missing a valid segments list"
            raise TypeError(msg)
        if segments:
            num_tasks_with_segments += 1
        for segment_index, segment in enumerate(segments):
            tagged_duration_s += _validate_prepared_segment(segment, index, segment_index)
            num_segments_processed += 1

    if num_segments_processed == 0:
        msg = "Audio tagging pipeline produced no tagged segments"
        raise RuntimeError(msg)

    return {
        "num_tasks_processed": len(output_tasks),
        "num_tasks_with_segments": num_tasks_with_segments,
        "num_segments_processed": num_segments_processed,
        "segment_task_coverage_ratio": num_tasks_with_segments / len(output_tasks),
        "total_audio_duration_hours": total_duration_s / 3600,
        "tagged_audio_duration_hours": tagged_duration_s / 3600,
    }


def load_vocab_file(filepath: str) -> set[str]:
    """Read a vocabulary file and return its characters as a set.

    The file can be in one of two formats:

    * **one-char-per-line** — each line holds a single allowed character
      (blank lines are treated as the space character).
    * **single line** — the entire first line is treated as the character set.
    """
    with open(filepath) as f:
        content = f.read()
    lines = content.splitlines()
    if len(lines) <= 1:
        return set(content.strip())
    chars: set[str] = set()
    for line in lines:
        ch = line.strip()
        if ch == "":
            chars.add(" ")
        else:
            if len(ch) > 1:
                logger.warning(f"Vocab file line has multiple characters: '{ch}' in {filepath}")
            chars.add(ch)
    logger.info(f"Loaded {len(chars)} vocab characters from {filepath}")
    return chars


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
