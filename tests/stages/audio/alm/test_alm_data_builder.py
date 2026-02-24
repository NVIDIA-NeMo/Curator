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

from nemo_curator.stages.audio.alm import ALMDataBuilderStage
from nemo_curator.tasks import AudioBatch


class TestALMDataBuilder:
    """Unit tests for ALMDataBuilderStage."""

    def test_creates_windows_from_sample(self, sample_entry: dict) -> None:
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

    def test_filters_low_sample_rate(self, sample_entries: list[dict]) -> None:
        entry = sample_entries[0].copy()
        entry["audio_sample_rate"] = 8000

        stage = ALMDataBuilderStage(
            target_window_duration=120.0,
            min_sample_rate=16000,
        )

        batch = AudioBatch(data=[entry])
        result = stage.process(batch)

        output = result[0].data[0]
        assert "stats" in output
        assert output["stats"].get("lost_sr", 0) > 0 or len(output.get("windows", [])) == 0

    def test_filters_low_bandwidth(self, sample_entries: list[dict]) -> None:
        entry = sample_entries[0].copy()
        entry["segments"] = [{**seg, "metrics": {"bandwidth": 4000}} for seg in entry["segments"]]

        stage = ALMDataBuilderStage(
            target_window_duration=120.0,
            min_bandwidth=8000,
        )

        batch = AudioBatch(data=[entry])
        result = stage.process(batch)

        output = result[0].data[0]
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

        batch = AudioBatch(data=[entry])
        result = stage.process(batch)

        output = result[0].data[0]
        assert len(output.get("windows", [])) == 0

    def test_empty_segments(self) -> None:
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

    def test_drop_fields(self, sample_entry: dict) -> None:
        entry = sample_entry.copy()
        entry["words"] = [{"word": "test", "start": 0, "end": 1}]
        entry["segments"] = [
            {**seg, "words": [{"word": "test", "start": seg["start"], "end": seg["end"]}]}
            for seg in entry["segments"]
        ]

        stage = ALMDataBuilderStage(
            target_window_duration=120.0,
            drop_fields="words",
            drop_fields_top_level="words,segments",
        )

        batch = AudioBatch(data=[entry])
        result = stage.process(batch)

        output = result[0].data[0]
        assert "words" not in output or output.get("words") is None
        assert "segments" not in output or output.get("segments") is None

    def test_different_sample_rates(self, sample_entries: list[dict]) -> None:
        stage = ALMDataBuilderStage(
            target_window_duration=120.0,
            min_sample_rate=16000,
        )

        for entry in sample_entries:
            batch = AudioBatch(data=[entry])
            result = stage.process(batch)
            assert len(result) == 1
            output = result[0].data[0]
            assert "windows" in output


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
            batch = AudioBatch(data=[entry])
            result = stage.process(batch)
            output = result[0].data[0]
            total_windows += len(output.get("windows", []))

        assert total_windows == 181
