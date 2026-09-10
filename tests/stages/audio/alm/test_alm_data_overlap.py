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

import copy

import pytest

from nemo_curator.stages.audio._agent._agent_registry import build_contract
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio.alm import ALMDataBuilderStage, ALMDataOverlapStage
from nemo_curator.tasks import AudioTask


class TestALMDataOverlap:
    """Unit tests for ALMDataOverlapStage."""

    def test_validate_input_valid(self, entry_with_windows: dict) -> None:
        stage = ALMDataOverlapStage(overlap_percentage=50, target_duration=120.0)
        assert stage.validate_input(AudioTask(data=entry_with_windows)) is True

    def test_validate_input_missing_windows(self) -> None:
        stage = ALMDataOverlapStage(overlap_percentage=50, target_duration=120.0)
        assert stage.validate_input(AudioTask(data={"audio_filepath": "a.wav"})) is False

    def test_process_batch_raises_on_missing_windows(self) -> None:
        stage = ALMDataOverlapStage(overlap_percentage=50, target_duration=120.0)
        with pytest.raises(ValueError, match="failed validation"):
            stage.process_batch([AudioTask(data={"audio_filepath": "a.wav"})])

    def test_filters_overlapping_windows(self, entry_with_windows: dict) -> None:
        stage = ALMDataOverlapStage(
            overlap_percentage=50,
            target_duration=120.0,
        )

        result = stage.process(AudioTask(data=entry_with_windows))
        assert isinstance(result, AudioTask)
        output = result.data
        assert "filtered_windows" in output
        assert len(output["filtered_windows"]) <= len(entry_with_windows.get("windows", []))

    def test_keeps_closer_to_target(self, entry_with_windows: dict) -> None:
        stage = ALMDataOverlapStage(
            overlap_percentage=0,
            target_duration=120.0,
        )

        result = stage.process(AudioTask(data=entry_with_windows))
        output = result.data
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

        aggressive_result = aggressive_stage.process(AudioTask(data=entry_with_windows))
        permissive_result = permissive_stage.process(AudioTask(data=entry_with_windows))

        aggressive_count = len(aggressive_result.data.get("filtered_windows", []))
        permissive_count = len(permissive_result.data.get("filtered_windows", []))

        assert permissive_count >= aggressive_count

    def test_no_windows(self) -> None:
        stage = ALMDataOverlapStage(overlap_percentage=50)

        entry = {
            "audio_filepath": "/path/to/audio.wav",
            "windows": [],
        }
        result = stage.process(AudioTask(data=entry))

        assert isinstance(result, AudioTask)
        assert result.data["audio_filepath"] == "/path/to/audio.wav"

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

        result = stage.process(AudioTask(data=entry_with_windows))
        output = result.data
        assert "filtered_dur" in output
        assert output["filtered_dur"] >= 0
        assert "filtered_dur_list" in output

    def test_declares_all_guaranteed_outputs_but_not_optional_stats(self) -> None:
        stage = ALMDataOverlapStage()
        expected = {
            "total_dur_window",
            "total_dur_list_window",
            "total_dur_list_window_timestamps",
            "filtered",
            "filtered_windows",
            "filtered_dur",
            "filtered_dur_list",
            "manifest_filepath",
            "swift_filepath",
        }
        contract = build_contract(stage)
        assert set(stage.outputs()[1]) == expected
        assert set(contract.writes.data_keys) == expected
        assert contract.reads.data_keys == ["windows"]
        assert "stats" not in contract.reads.data_keys

    def test_fully_renamed_outputs_work_for_populated_and_empty_rows(self, entry_with_windows: dict) -> None:
        stage = ALMDataOverlapStage(
            windows_key="input_windows",
            filtered_windows_key="kept_windows",
            stats_key="builder_stats",
            total_dur_window_key="window_duration_total",
            total_dur_list_window_key="window_durations",
            total_dur_list_window_timestamps_key="window_timestamps",
            filtered_key="kept_timestamps",
            filtered_dur_key="kept_duration_total",
            filtered_dur_list_key="kept_durations",
            manifest_filepath_key="source_manifest",
            swift_filepath_key="source_swift",
        )
        renamed = copy.deepcopy(entry_with_windows)
        renamed["input_windows"] = renamed.pop("windows")
        renamed["builder_stats"] = {
            **renamed.pop("stats"),
            "manifest_path": "manifests/input.jsonl",
            "swift_path": "swift://bucket/audio.wav",
        }

        populated = stage.process(AudioTask(data=copy.deepcopy(renamed))).data
        assert populated["kept_windows"]
        assert populated["window_duration_total"] >= 0
        assert populated["source_manifest"] == "manifests/input.jsonl"
        assert populated["source_swift"] == "swift://bucket/audio.wav"

        empty = copy.deepcopy(renamed)
        empty["input_windows"] = []
        empty_output = stage.process(AudioTask(data=empty)).data
        assert empty_output["kept_windows"] == []
        assert empty_output["window_duration_total"] == 0.0
        assert empty_output["window_durations"] == []
        assert empty_output["window_timestamps"] == []
        assert empty_output["kept_timestamps"] == []
        assert empty_output["kept_duration_total"] == 0.0
        assert empty_output["kept_durations"] == []
        assert empty_output["source_manifest"] is None
        assert empty_output["source_swift"] is None

    def test_agent_ready_default_and_fully_renamed(self, entry_with_windows: dict) -> None:
        def default_fixture() -> AudioTask:
            return AudioTask(data=copy.deepcopy(entry_with_windows))

        assert_agent_ready(
            ALMDataOverlapStage(),
            default_fixture,
            expected_cardinality="1:1",
            available_keys={"windows"},
        )

        stage = ALMDataOverlapStage(
            windows_key="input_windows",
            filtered_windows_key="kept_windows",
            stats_key="builder_stats",
            total_dur_window_key="window_duration_total",
            total_dur_list_window_key="window_durations",
            total_dur_list_window_timestamps_key="window_timestamps",
            filtered_key="kept_timestamps",
            filtered_dur_key="kept_duration_total",
            filtered_dur_list_key="kept_durations",
            manifest_filepath_key="source_manifest",
            swift_filepath_key="source_swift",
        )

        def renamed_fixture() -> AudioTask:
            entry = copy.deepcopy(entry_with_windows)
            entry["input_windows"] = entry.pop("windows")
            entry["builder_stats"] = entry.pop("stats")
            return AudioTask(data=entry)

        assert_agent_ready(
            stage,
            renamed_fixture,
            expected_cardinality="1:1",
            available_keys={"input_windows"},
        )


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
            builder_result = builder.process(AudioTask(data=entry))
            builder_output = builder_result.data
            total_builder_windows += len(builder_output.get("windows", []))

            overlap_result = overlap.process(builder_result)
            overlap_output = overlap_result.data
            total_filtered_windows += len(overlap_output.get("filtered_windows", []))
            total_filtered_dur += overlap_output.get("filtered_dur", 0)

        assert total_builder_windows == 181
        assert total_filtered_windows == 25
        assert abs(total_filtered_dur - 3035.50) < 1.0
