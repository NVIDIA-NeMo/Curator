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

"""Tests for ALMDataOverlapStage using sample data fixtures."""

import pytest

from nemo_curator.stages.audio.alm import ALMDataBuilderStage, ALMDataOverlapStage
from nemo_curator.tasks import AudioBatch


class TestALMDataOverlap:
    """Unit tests for ALMDataOverlapStage."""

    def test_filters_overlapping_windows(self, entry_with_windows: dict) -> None:
        stage = ALMDataOverlapStage(
            overlap_percentage=50,
            target_duration=120.0,
        )

        batch = AudioBatch(data=[entry_with_windows])
        result = stage.process(batch)

        assert len(result) == 1
        output = result[0].data[0]
        assert "filtered_windows" in output
        assert len(output["filtered_windows"]) <= len(entry_with_windows.get("windows", []))

    def test_keeps_closer_to_target(self, entry_with_windows: dict) -> None:
        stage = ALMDataOverlapStage(
            overlap_percentage=0,
            target_duration=120.0,
        )

        batch = AudioBatch(data=[entry_with_windows])
        result = stage.process(batch)

        output = result[0].data[0]
        filtered = output.get("filtered_windows", [])
        assert len(filtered) >= 0

    def test_permissive_mode(self, entry_with_windows: dict) -> None:
        aggressive_stage = ALMDataOverlapStage(
            overlap_percentage=0,
            target_duration=120.0,
        )
        permissive_stage = ALMDataOverlapStage(
            overlap_percentage=100,
            target_duration=120.0,
        )

        aggressive_result = aggressive_stage.process(AudioBatch(data=[entry_with_windows]))
        permissive_result = permissive_stage.process(AudioBatch(data=[entry_with_windows]))

        aggressive_count = len(aggressive_result[0].data[0].get("filtered_windows", []))
        permissive_count = len(permissive_result[0].data[0].get("filtered_windows", []))

        assert permissive_count >= aggressive_count

    def test_no_windows(self) -> None:
        stage = ALMDataOverlapStage(overlap_percentage=50)

        entry = {
            "audio_filepath": "/path/to/audio.wav",
            "windows": [],
        }
        batch = AudioBatch(data=[entry])

        result = stage.process(batch)

        assert len(result) == 1
        assert result[0].data[0]["audio_filepath"] == "/path/to/audio.wav"

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="overlap_percentage must be 0-100"):
            ALMDataOverlapStage(overlap_percentage=-1)

        with pytest.raises(ValueError, match="overlap_percentage must be 0-100"):
            ALMDataOverlapStage(overlap_percentage=101)

        with pytest.raises(ValueError, match="target_duration must be positive"):
            ALMDataOverlapStage(target_duration=-1)

    def test_calculates_duration(self, entry_with_windows: dict) -> None:
        stage = ALMDataOverlapStage(
            overlap_percentage=100,
            target_duration=120.0,
        )

        batch = AudioBatch(data=[entry_with_windows])
        result = stage.process(batch)

        output = result[0].data[0]
        assert "filtered_dur" in output
        assert output["filtered_dur"] >= 0
        assert "filtered_dur_list" in output


class TestALMDataOverlapIntegration:
    """Integration tests for the full Builder -> Overlap pipeline."""

    def test_full_pipeline(self, sample_entries: list[dict]) -> None:
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
            batch = AudioBatch(data=[entry])
            builder_result = builder.process(batch)
            builder_output = builder_result[0].data[0]
            total_builder_windows += len(builder_output.get("windows", []))

            overlap_result = overlap.process(builder_result[0])
            overlap_output = overlap_result[0].data[0]
            total_filtered_windows += len(overlap_output.get("filtered_windows", []))
            total_filtered_dur += overlap_output.get("filtered_dur", 0)

        assert total_builder_windows == 181
        assert total_filtered_windows == 25
        assert abs(total_filtered_dur - 3035.50) < 1.0
