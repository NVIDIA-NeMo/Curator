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

"""Tests for ALMDataBuilderStage using sample data fixtures."""

import copy

import pytest

from nemo_curator.stages.audio._agent._agent_registry import build_contract
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.audio.alm import ALMDataBuilderStage
from nemo_curator.stages.audio.preprocessing import MonoConversionStage
from nemo_curator.tasks import AudioTask


class TestALMDataBuilder:
    """Unit tests for ALMDataBuilderStage."""

    def test_validate_input_valid(self, sample_entry: dict) -> None:
        stage = ALMDataBuilderStage()
        assert stage.validate_input(AudioTask(data=sample_entry)) is True

    def test_validate_input_missing_segments(self) -> None:
        stage = ALMDataBuilderStage()
        assert stage.validate_input(AudioTask(data={"audio_filepath": "a.wav", "audio_sample_rate": 16000})) is False

    def test_validate_input_missing_sample_rate(self) -> None:
        stage = ALMDataBuilderStage()
        assert stage.validate_input(AudioTask(data={"audio_filepath": "a.wav", "segments": []})) is False

    def test_process_batch_raises_on_missing_segments(self) -> None:
        stage = ALMDataBuilderStage()
        with pytest.raises(ValueError, match="failed validation"):
            stage.process_batch([AudioTask(data={"audio_filepath": "a.wav", "audio_sample_rate": 16000})])

    def test_process_batch_raises_on_missing_sample_rate(self) -> None:
        stage = ALMDataBuilderStage()
        with pytest.raises(ValueError, match="failed validation"):
            stage.process_batch([AudioTask(data={"audio_filepath": "a.wav", "segments": []})])

    def test_creates_windows_from_sample(self, sample_entry: dict) -> None:
        stage = ALMDataBuilderStage(
            target_window_duration=120.0,
            tolerance=0.1,
            min_sample_rate=16000,
            min_bandwidth=8000,
            min_speakers=2,
            max_speakers=5,
        )

        result = stage.process(AudioTask(data=sample_entry))

        assert isinstance(result, AudioTask)
        output = result.data
        assert "windows" in output
        assert len(output["windows"]) > 0
        assert "stats" in output

    def test_filters_low_sample_rate(self, sample_entries: list[dict]) -> None:
        entry = sample_entries[0].copy()
        entry["audio_sample_rate"] = 8000

        stage = ALMDataBuilderStage(
            target_window_duration=120.0,
            min_sample_rate=16000,
        )

        result = stage.process(AudioTask(data=entry))
        output = result.data
        assert "stats" in output
        assert output["stats"].get("lost_sr", 0) > 0 or len(output.get("windows", [])) == 0

    def test_filters_low_bandwidth(self, sample_entries: list[dict]) -> None:
        entry = sample_entries[0].copy()
        entry["segments"] = [{**seg, "metrics": {"bandwidth": 4000}} for seg in entry["segments"]]

        stage = ALMDataBuilderStage(
            target_window_duration=120.0,
            min_bandwidth=8000,
        )

        result = stage.process(AudioTask(data=entry))
        output = result.data
        assert "stats" in output
        assert output["stats"].get("lost_bw", 0) > 0

    def test_speaker_constraints(self, sample_entries: list[dict]) -> None:
        entry = sample_entries[0].copy()
        entry["segments"] = [{**seg, "speaker": "single_speaker"} for seg in entry["segments"]]

        stage = ALMDataBuilderStage(
            target_window_duration=120.0,
            min_speakers=2,
            max_speakers=3,
        )

        result = stage.process(AudioTask(data=entry))
        output = result.data
        assert len(output.get("windows", [])) == 0

    def test_empty_segments(self) -> None:
        stage = ALMDataBuilderStage(target_window_duration=120.0)

        entry = {
            "audio_filepath": "/path/to/audio.wav",
            "audio_sample_rate": 16000,
            "segments": [],
        }
        result = stage.process(AudioTask(data=entry))

        assert isinstance(result, AudioTask)
        assert result.data.get("windows", []) == []

    def test_drop_fields(self, sample_entry: dict) -> None:
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

        result = stage.process(AudioTask(data=entry))
        output = result.data
        assert "words" not in output or output.get("words") is None
        assert "segments" not in output or output.get("segments") is None

    def test_different_sample_rates(self, sample_entries: list[dict]) -> None:
        stage = ALMDataBuilderStage(
            target_window_duration=120.0,
            min_sample_rate=16000,
        )

        for entry in sample_entries:
            result = stage.process(AudioTask(data=entry))
            assert isinstance(result, AudioTask)
            assert "windows" in result.data

    def test_audio_filepath_contract_follows_explicit_top_level_drop(self, sample_entry: dict) -> None:
        default = ALMDataBuilderStage()
        custom = ALMDataBuilderStage(audio_filepath_key="source_path")
        dropped = ALMDataBuilderStage(drop_fields_top_level="words,segments,audio_filepath")

        assert "audio_filepath" in default.outputs()[1]
        assert "audio_filepath" in build_contract(default).writes.data_keys
        assert "source_path" in custom.outputs()[1]
        assert "source_path" in build_contract(custom).writes.data_keys
        assert "audio_filepath" not in dropped.outputs()[1]
        assert "audio_filepath" not in build_contract(dropped).writes.data_keys

        normal = dropped.process(AudioTask(data=copy.deepcopy(sample_entry)))
        assert "audio_filepath" not in normal.data

        low_rate_entry = copy.deepcopy(sample_entry)
        low_rate_entry["audio_sample_rate"] = 8000
        low_rate = dropped.process(AudioTask(data=low_rate_entry))
        assert "audio_filepath" not in low_rate.data

    def test_agent_ready_default_custom_and_explicit_drop(self, sample_entry: dict) -> None:
        def default_fixture() -> AudioTask:
            return AudioTask(data=copy.deepcopy(sample_entry))

        assert_agent_ready(
            ALMDataBuilderStage(),
            default_fixture,
            expected_cardinality="1:1",
            available_keys={"audio_filepath", "segments", "audio_sample_rate"},
        )

        custom_data = copy.deepcopy(sample_entry)
        custom_data["source_path"] = custom_data.pop("audio_filepath")

        def custom_fixture() -> AudioTask:
            return AudioTask(data=copy.deepcopy(custom_data))

        assert_agent_ready(
            ALMDataBuilderStage(audio_filepath_key="source_path"),
            custom_fixture,
            expected_cardinality="1:1",
            available_keys={"source_path", "segments", "audio_sample_rate"},
        )
        assert_agent_ready(
            ALMDataBuilderStage(drop_fields_top_level="words,segments,audio_filepath"),
            default_fixture,
            expected_cardinality="1:1",
            available_keys={"audio_filepath", "segments", "audio_sample_rate"},
        )

    def test_file_consumer_planning_tracks_default_custom_and_dropped_path(self) -> None:
        seed = {
            "initial_roles": {"audio_filepath", "segments", "sample_rate"},
            "initial_keys": {"audio_filepath", "segments", "audio_sample_rate"},
        }
        default = validate_pipeline([ALMDataBuilderStage(), MonoConversionStage()], **seed)
        assert default.ok
        assert default.keys_ok

        custom = validate_pipeline(
            [
                ALMDataBuilderStage(audio_filepath_key="source_path"),
                MonoConversionStage(audio_filepath_key="source_path"),
            ],
            initial_roles={"audio_filepath", "segments", "sample_rate"},
            initial_keys={"source_path", "segments", "audio_sample_rate"},
        )
        assert custom.ok
        assert custom.keys_ok

        dropped = validate_pipeline(
            [
                ALMDataBuilderStage(drop_fields_top_level="words,segments,audio_filepath"),
                MonoConversionStage(),
            ],
            **seed,
        )
        assert not dropped.ok
        assert any(issue.code == "key_removed_upstream" for issue in dropped.issues)


class TestALMDataBuilderIntegration:
    """Integration tests for ALMDataBuilderStage across full fixture dataset."""

    def test_processes_all_sample_entries(self, sample_entries: list[dict]) -> None:
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
            result = stage.process(AudioTask(data=entry))
            total_windows += len(result.data.get("windows", []))

        assert total_windows == 181
