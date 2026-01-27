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

"""Tests for ALMDataBuilderStage using sample data fixtures."""

import json
from pathlib import Path

import pytest

from nemo_curator.stages.audio.alm import ALMDataBuilderStage
from nemo_curator.tasks import AudioBatch

# Path to sample data fixture
FIXTURE_PATH = Path(__file__).parent.parent.parent.parent / "fixtures" / "audio" / "alm" / "sample_input.jsonl"


@pytest.fixture
def sample_entries() -> list[dict]:
    """Load sample entries from fixture file."""
    entries = []
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
    return entries


@pytest.fixture
def sample_entry(sample_entries: list[dict]) -> dict:
    """Get first sample entry (conversation_001.wav)."""
    return sample_entries[0]


def test_alm_data_builder_creates_windows_from_sample(sample_entry: dict) -> None:
    """Test that ALMDataBuilderStage creates windows from sample data."""
    stage = ALMDataBuilderStage(
        target_window_duration=120.0,
        tolerance=0.1,
        min_sample_rate=16000,
        min_bandwidth=8000,
        min_speakers=2,
        max_speakers=5,
    )

    batch = AudioBatch(data=[sample_entry])
    result = stage.process(batch)

    assert len(result) == 1
    assert isinstance(result[0], AudioBatch)
    output = result[0].data[0]
    assert "windows" in output
    assert len(output["windows"]) > 0
    assert "stats" in output


def test_alm_data_builder_processes_all_sample_entries(sample_entries: list[dict]) -> None:
    """Test that ALMDataBuilderStage processes all 5 sample entries."""
    stage = ALMDataBuilderStage(
        target_window_duration=120.0,
        tolerance=0.1,
        min_sample_rate=16000,
        min_bandwidth=8000,
        min_speakers=2,
        max_speakers=5,
    )

    total_windows = 0
    for entry in sample_entries:
        batch = AudioBatch(data=[entry])
        result = stage.process(batch)
        output = result[0].data[0]
        total_windows += len(output.get("windows", []))

    # Based on actual test run, expect 181 windows from 5 entries
    assert total_windows == 181


def test_alm_data_builder_filters_low_sample_rate(sample_entries: list[dict]) -> None:
    """Test that entries with low sample rate produce fewer windows."""
    # sample_entries[0] has 16000 Hz sample rate
    entry = sample_entries[0].copy()
    entry["audio_sample_rate"] = 8000  # Below min

    stage = ALMDataBuilderStage(
        target_window_duration=120.0,
        min_sample_rate=16000,
    )

    batch = AudioBatch(data=[entry])
    result = stage.process(batch)

    output = result[0].data[0]
    # Should have stats showing sample rate loss
    assert "stats" in output
    assert output["stats"].get("lost_sr", 0) > 0 or len(output.get("windows", [])) == 0


def test_alm_data_builder_filters_low_bandwidth(sample_entries: list[dict]) -> None:
    """Test that segments with low bandwidth are filtered."""
    entry = sample_entries[0].copy()
    # Modify all segments to have low bandwidth
    entry["segments"] = [{**seg, "metrics": {"bandwidth": 4000}} for seg in entry["segments"]]

    stage = ALMDataBuilderStage(
        target_window_duration=120.0,
        min_bandwidth=8000,
    )

    batch = AudioBatch(data=[entry])
    result = stage.process(batch)

    output = result[0].data[0]
    assert "stats" in output
    # All segments should be filtered due to low bandwidth
    assert output["stats"].get("lost_bw", 0) > 0


def test_alm_data_builder_speaker_constraints(sample_entries: list[dict]) -> None:
    """Test that windows respect speaker count constraints."""
    # Modify entry to have only 1 speaker
    entry = sample_entries[0].copy()
    entry["segments"] = [{**seg, "speaker": "single_speaker"} for seg in entry["segments"]]

    stage = ALMDataBuilderStage(
        target_window_duration=120.0,
        min_speakers=2,
        max_speakers=3,
    )

    batch = AudioBatch(data=[entry])
    result = stage.process(batch)

    output = result[0].data[0]
    # Windows should be empty since we need at least 2 speakers
    assert len(output.get("windows", [])) == 0


def test_alm_data_builder_default_values() -> None:
    """Test that stage can be created with default values."""
    stage = ALMDataBuilderStage()
    assert stage.target_window_duration == 120.0
    assert stage.tolerance == 0.1
    assert stage.min_speakers == 2
    assert stage.max_speakers == 5


def test_alm_data_builder_empty_segments() -> None:
    """Test handling of entry with no segments."""
    stage = ALMDataBuilderStage(target_window_duration=120.0)

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


def test_alm_data_builder_drop_fields(sample_entry: dict) -> None:
    """Test that specified fields are dropped from output."""
    # Add words to segments
    entry = sample_entry.copy()
    entry["words"] = [{"word": "test", "start": 0, "end": 1}]
    entry["segments"] = [
        {**seg, "words": [{"word": "test", "start": seg["start"], "end": seg["end"]}]} for seg in entry["segments"]
    ]

    stage = ALMDataBuilderStage(
        target_window_duration=120.0,
        drop_fields="words",
        drop_fields_top_level="words,segments",
    )

    batch = AudioBatch(data=[entry])
    result = stage.process(batch)

    output = result[0].data[0]
    # Top-level fields should be dropped
    assert "words" not in output or output.get("words") is None
    assert "segments" not in output or output.get("segments") is None


def test_alm_data_builder_different_sample_rates(sample_entries: list[dict]) -> None:
    """Test processing entries with different sample rates."""
    stage = ALMDataBuilderStage(
        target_window_duration=120.0,
        min_sample_rate=16000,
    )

    # Check that entries with different sample rates are handled
    # sample_entries have: 16000, 22050, 16000, 48000, 44100
    for entry in sample_entries:
        batch = AudioBatch(data=[entry])
        result = stage.process(batch)
        assert len(result) == 1
        output = result[0].data[0]
        # All should pass since all are >= 16000
        assert "windows" in output
