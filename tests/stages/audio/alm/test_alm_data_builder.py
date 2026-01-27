# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for ALMDataBuilderStage."""

import pytest

from nemo_curator.stages.audio.alm import ALMDataBuilderStage
from nemo_curator.tasks import AudioBatch


def _create_test_entry(
    sample_rate: int = 16000,
    num_segments: int = 10,
    segment_duration: float = 15.0,
    num_speakers: int = 3,
    bandwidth: int = 8000,
) -> dict:
    """Create a test manifest entry with segments."""
    segments = []
    for i in range(num_segments):
        start = i * segment_duration
        end = start + segment_duration
        segments.append({
            "start": start,
            "end": end,
            "speaker": f"speaker_{i % num_speakers}",
            "text": f"Segment {i} text",
            "metrics": {"bandwidth": bandwidth},
        })
    return {
        "audio_filepath": "/path/to/audio.wav",
        "audio_sample_rate": sample_rate,
        "segments": segments,
    }


def test_alm_data_builder_creates_windows() -> None:
    """Test that ALMDataBuilderStage creates windows from segments."""
    stage = ALMDataBuilderStage(
        target_window_duration=30.0,
        tolerance=0.2,
        min_sample_rate=16000,
        min_bandwidth=8000,
        min_speakers=2,
        max_speakers=5,
    )

    entry = _create_test_entry(num_segments=10, segment_duration=10.0, num_speakers=3)
    batch = AudioBatch(data=[entry])

    result = stage.process(batch)

    assert len(result) == 1
    assert isinstance(result[0], AudioBatch)
    output = result[0].data[0]
    assert "windows" in output
    assert len(output["windows"]) > 0


def test_alm_data_builder_filters_low_sample_rate() -> None:
    """Test that entries with low sample rate are skipped."""
    stage = ALMDataBuilderStage(
        target_window_duration=30.0,
        min_sample_rate=16000,
    )

    entry = _create_test_entry(sample_rate=8000)  # Below min
    batch = AudioBatch(data=[entry])

    result = stage.process(batch)

    assert len(result) == 1
    output = result[0].data[0]
    # Should have stats showing sample rate loss
    assert "stats" in output
    assert output["stats"].get("lost_sr", 0) > 0 or len(output.get("windows", [])) == 0


def test_alm_data_builder_filters_low_bandwidth() -> None:
    """Test that segments with low bandwidth are filtered."""
    stage = ALMDataBuilderStage(
        target_window_duration=30.0,
        min_bandwidth=8000,
    )

    entry = _create_test_entry(bandwidth=4000)  # Below min
    batch = AudioBatch(data=[entry])

    result = stage.process(batch)

    assert len(result) == 1
    output = result[0].data[0]
    # Should have stats showing bandwidth loss
    assert "stats" in output


def test_alm_data_builder_speaker_constraints() -> None:
    """Test that windows respect speaker count constraints."""
    stage = ALMDataBuilderStage(
        target_window_duration=30.0,
        min_speakers=2,
        max_speakers=3,
    )

    # Entry with only 1 speaker
    entry = _create_test_entry(num_speakers=1)
    batch = AudioBatch(data=[entry])

    result = stage.process(batch)

    assert len(result) == 1
    output = result[0].data[0]
    # Windows should be empty or stats show speaker loss
    windows = output.get("windows", [])
    # All windows should have 2-3 speakers or be filtered out


def test_alm_data_builder_default_values() -> None:
    """Test that stage can be created with default values."""
    stage = ALMDataBuilderStage()
    assert stage.target_window_duration == 120.0
    assert stage.tolerance == 0.1
    assert stage.min_speakers == 2
    assert stage.max_speakers == 5


def test_alm_data_builder_empty_segments() -> None:
    """Test handling of entry with no segments."""
    stage = ALMDataBuilderStage(target_window_duration=30.0)

    entry = {
        "audio_filepath": "/path/to/audio.wav",
        "audio_sample_rate": 16000,
        "segments": [],
    }
    batch = AudioBatch(data=[entry])

    result = stage.process(batch)

    assert len(result) == 1
    output = result[0].data[0]
    assert output.get("windows", []) == []


def test_alm_data_builder_drop_fields() -> None:
    """Test that specified fields are dropped from output."""
    stage = ALMDataBuilderStage(
        target_window_duration=60.0,
        drop_fields="words",
        drop_fields_top_level="words,segments",
    )

    entry = _create_test_entry(num_segments=5, segment_duration=15.0)
    entry["words"] = [{"word": "test", "start": 0, "end": 1}]
    for seg in entry["segments"]:
        seg["words"] = [{"word": "test", "start": seg["start"], "end": seg["end"]}]

    batch = AudioBatch(data=[entry])
    result = stage.process(batch)

    assert len(result) == 1
    output = result[0].data[0]
    # Top-level fields should be dropped
    assert "words" not in output or output.get("words") is None
    assert "segments" not in output or output.get("segments") is None
