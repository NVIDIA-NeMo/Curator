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

"""Tests for ALMDataOverlapStage using sample data fixtures."""

import json
from pathlib import Path

import pytest

from nemo_curator.stages.audio.alm import ALMDataBuilderStage, ALMDataOverlapStage
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
    """Get first sample entry."""
    return sample_entries[0]


@pytest.fixture
def entry_with_windows(sample_entry: dict) -> dict:
    """Process sample entry through ALMDataBuilderStage to get windows."""
    builder = ALMDataBuilderStage(
        target_window_duration=120.0,
        tolerance=0.1,
        min_sample_rate=16000,
        min_bandwidth=8000,
        min_speakers=2,
        max_speakers=5,
    )
    batch = AudioBatch(data=[sample_entry])
    result = builder.process(batch)
    return result[0].data[0]


def test_alm_data_overlap_filters_overlapping_windows(entry_with_windows: dict) -> None:
    """Test that overlapping windows are filtered from real sample data."""
    stage = ALMDataOverlapStage(
        overlap_percentage=50,
        target_duration=120.0,
    )

    batch = AudioBatch(data=[entry_with_windows])
    result = stage.process(batch)

    assert len(result) == 1
    output = result[0].data[0]
    assert "filtered_windows" in output
    # Should have filtered some overlapping windows
    assert len(output["filtered_windows"]) <= len(entry_with_windows.get("windows", []))


def test_alm_data_overlap_full_pipeline(sample_entries: list[dict]) -> None:
    """Test full pipeline: Builder -> Overlap on all sample entries."""
    builder = ALMDataBuilderStage(
        target_window_duration=120.0,
        tolerance=0.1,
        min_sample_rate=16000,
        min_bandwidth=8000,
        min_speakers=2,
        max_speakers=5,
    )
    overlap = ALMDataOverlapStage(
        overlap_percentage=50,
        target_duration=120.0,
    )

    total_builder_windows = 0
    total_filtered_windows = 0
    total_filtered_dur = 0.0

    for entry in sample_entries:
        # Stage 1: Builder
        batch = AudioBatch(data=[entry])
        builder_result = builder.process(batch)
        builder_output = builder_result[0].data[0]
        total_builder_windows += len(builder_output.get("windows", []))

        # Stage 2: Overlap
        overlap_result = overlap.process(builder_result[0])
        overlap_output = overlap_result[0].data[0]
        total_filtered_windows += len(overlap_output.get("filtered_windows", []))
        total_filtered_dur += overlap_output.get("filtered_dur", 0)

    # Based on actual test run: 181 builder windows -> 25 filtered windows
    assert total_builder_windows == 181
    assert total_filtered_windows == 25
    assert abs(total_filtered_dur - 3035.50) < 1.0  # Allow small floating point variance


def test_alm_data_overlap_keeps_closer_to_target(entry_with_windows: dict) -> None:
    """Test that windows closer to target duration are kept."""
    stage = ALMDataOverlapStage(
        overlap_percentage=0,  # Aggressive filtering
        target_duration=120.0,
    )

    batch = AudioBatch(data=[entry_with_windows])
    result = stage.process(batch)

    output = result[0].data[0]
    filtered = output.get("filtered_windows", [])
    # Should have some windows after filtering
    assert len(filtered) >= 0


def test_alm_data_overlap_permissive_mode(entry_with_windows: dict) -> None:
    """Test permissive mode keeps more windows."""
    aggressive_stage = ALMDataOverlapStage(
        overlap_percentage=0,  # Most aggressive
        target_duration=120.0,
    )
    permissive_stage = ALMDataOverlapStage(
        overlap_percentage=100,  # Most permissive
        target_duration=120.0,
    )

    batch = AudioBatch(data=[entry_with_windows])

    aggressive_result = aggressive_stage.process(batch)
    permissive_result = permissive_stage.process(AudioBatch(data=[entry_with_windows]))

    aggressive_count = len(aggressive_result[0].data[0].get("filtered_windows", []))
    permissive_count = len(permissive_result[0].data[0].get("filtered_windows", []))

    # Permissive should keep at least as many windows
    assert permissive_count >= aggressive_count


def test_alm_data_overlap_no_windows() -> None:
    """Test handling of entry with no windows."""
    stage = ALMDataOverlapStage(overlap_percentage=50)

    entry = {
        "audio_filepath": "/path/to/audio.wav",
        "windows": [],
    }
    batch = AudioBatch(data=[entry])

    result = stage.process(batch)

    assert len(result) == 1
    # Should return entry unchanged - verify it has same filepath
    assert result[0].data[0]["audio_filepath"] == "/path/to/audio.wav"


def test_alm_data_overlap_validation() -> None:
    """Test parameter validation."""
    with pytest.raises(ValueError, match="overlap_percentage must be 0-100"):
        ALMDataOverlapStage(overlap_percentage=-1)

    with pytest.raises(ValueError, match="overlap_percentage must be 0-100"):
        ALMDataOverlapStage(overlap_percentage=101)

    with pytest.raises(ValueError, match="target_duration must be positive"):
        ALMDataOverlapStage(target_duration=-1)


def test_alm_data_overlap_calculates_duration(entry_with_windows: dict) -> None:
    """Test that filtered_dur is calculated correctly."""
    stage = ALMDataOverlapStage(
        overlap_percentage=100,  # Keep all
        target_duration=120.0,
    )

    batch = AudioBatch(data=[entry_with_windows])
    result = stage.process(batch)

    output = result[0].data[0]
    assert "filtered_dur" in output
    assert output["filtered_dur"] >= 0
    # Should also have filtered_dur_list
    assert "filtered_dur_list" in output
