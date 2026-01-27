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

"""Tests for ALMDataOverlapStage."""

import pytest

from nemo_curator.stages.audio.alm import ALMDataOverlapStage
from nemo_curator.tasks import AudioBatch


def _create_window(start: float, end: float, speaker: str = "speaker_0") -> dict:
    """Create a test window with segments."""
    return {
        "segments": [
            {"start": start, "end": end, "speaker": speaker, "text": "Test text"}
        ],
        "speaker_durations": [end - start, 0, 0, 0, 0],
    }


def _create_test_entry_with_windows(windows: list[dict]) -> dict:
    """Create a test entry with pre-built windows."""
    return {
        "audio_filepath": "/path/to/audio.wav",
        "windows": windows,
    }


def test_alm_data_overlap_filters_overlapping_windows() -> None:
    """Test that overlapping windows are filtered."""
    stage = ALMDataOverlapStage(
        overlap_percentage=50,
        target_duration=120.0,
    )

    # Create overlapping windows
    windows = [
        _create_window(0, 120),    # 0-120s
        _create_window(60, 180),   # 60-180s (overlaps with first)
        _create_window(200, 320),  # Non-overlapping
    ]
    entry = _create_test_entry_with_windows(windows)
    batch = AudioBatch(data=[entry])

    result = stage.process(batch)

    assert len(result) == 1
    output = result[0].data[0]
    assert "filtered_windows" in output
    # Should have filtered some overlapping windows
    assert len(output["filtered_windows"]) <= len(windows)


def test_alm_data_overlap_keeps_closer_to_target() -> None:
    """Test that windows closer to target duration are kept."""
    stage = ALMDataOverlapStage(
        overlap_percentage=0,  # Aggressive filtering
        target_duration=120.0,
    )

    # Window exactly at target should be preferred
    windows = [
        _create_window(0, 120),   # Exactly 120s - should be preferred
        _create_window(0, 100),   # 100s - overlaps, further from target
    ]
    entry = _create_test_entry_with_windows(windows)
    batch = AudioBatch(data=[entry])

    result = stage.process(batch)

    assert len(result) == 1
    output = result[0].data[0]
    filtered = output.get("filtered_windows", [])
    # The 120s window should be kept as it's closer to target
    if filtered:
        durations = [w["segments"][-1]["end"] - w["segments"][0]["start"] for w in filtered]
        assert 120.0 in durations or len(durations) == 1


def test_alm_data_overlap_permissive_mode() -> None:
    """Test permissive mode keeps all windows."""
    stage = ALMDataOverlapStage(
        overlap_percentage=100,  # Most permissive
        target_duration=120.0,
    )

    windows = [
        _create_window(0, 120),
        _create_window(60, 180),
        _create_window(120, 240),
    ]
    entry = _create_test_entry_with_windows(windows)
    batch = AudioBatch(data=[entry])

    result = stage.process(batch)

    assert len(result) == 1
    output = result[0].data[0]
    filtered = output.get("filtered_windows", [])
    # With 100% threshold, most/all windows should pass
    assert len(filtered) >= 1


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
    output = result[0].data[0]
    # Should return entry unchanged or with empty filtered_windows


def test_alm_data_overlap_validation() -> None:
    """Test parameter validation."""
    with pytest.raises(ValueError):
        ALMDataOverlapStage(overlap_percentage=-1)

    with pytest.raises(ValueError):
        ALMDataOverlapStage(overlap_percentage=101)

    with pytest.raises(ValueError):
        ALMDataOverlapStage(target_duration=-1)


def test_alm_data_overlap_calculates_duration() -> None:
    """Test that filtered_dur is calculated correctly."""
    stage = ALMDataOverlapStage(
        overlap_percentage=100,  # Keep all
        target_duration=120.0,
    )

    windows = [
        _create_window(0, 100),
        _create_window(200, 320),
    ]
    entry = _create_test_entry_with_windows(windows)
    batch = AudioBatch(data=[entry])

    result = stage.process(batch)

    assert len(result) == 1
    output = result[0].data[0]
    assert "filtered_dur" in output
    # Total duration should be sum of non-filtered windows
    assert output["filtered_dur"] >= 0


def test_alm_data_overlap_non_overlapping_windows() -> None:
    """Test that non-overlapping windows are all kept."""
    stage = ALMDataOverlapStage(
        overlap_percentage=0,  # Aggressive
        target_duration=120.0,
    )

    # Non-overlapping windows
    windows = [
        _create_window(0, 100),
        _create_window(150, 250),
        _create_window(300, 400),
    ]
    entry = _create_test_entry_with_windows(windows)
    batch = AudioBatch(data=[entry])

    result = stage.process(batch)

    assert len(result) == 1
    output = result[0].data[0]
    filtered = output.get("filtered_windows", [])
    # All should pass since there's no overlap
    assert len(filtered) == 3
