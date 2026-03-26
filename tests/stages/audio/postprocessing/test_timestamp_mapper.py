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

"""Unit tests for TimestampMapperStage."""

import torch

from nemo_curator.stages.audio.postprocessing.timestamp_mapper import (
    TimestampMapperStage,
    _translate_to_original,
)
from nemo_curator.tasks import AudioTask


SAMPLE_MAPPINGS = [
    {
        "original_file": "/data/audio.wav",
        "original_start_ms": 2000,
        "original_end_ms": 5000,
        "concat_start_ms": 0,
        "concat_end_ms": 3000,
        "segment_index": 0,
    },
    {
        "original_file": "/data/audio.wav",
        "original_start_ms": 12000,
        "original_end_ms": 17000,
        "concat_start_ms": 3500,
        "concat_end_ms": 8500,
        "segment_index": 1,
    },
    {
        "original_file": "/data/audio.wav",
        "original_start_ms": 45000,
        "original_end_ms": 49000,
        "concat_start_ms": 9000,
        "concat_end_ms": 13000,
        "segment_index": 2,
    },
]


def _make_task(item, mappings=None, task_id="test"):
    metadata = {}
    if mappings is not None:
        metadata["segment_mappings"] = mappings
    task = AudioTask(
        data=item,
        task_id=task_id,
        dataset_name="test",
    )
    task._metadata = metadata
    return task


class TestTranslateToOriginal:

    def test_segment_within_single_mapping(self):
        results = _translate_to_original(SAMPLE_MAPPINGS, 500, 2500)
        assert len(results) == 1
        assert results[0]["original_file"] == "/data/audio.wav"
        assert results[0]["original_start_ms"] == 2500
        assert results[0]["original_end_ms"] == 4500
        assert results[0]["duration_ms"] == 2000

    def test_segment_spans_two_mappings(self):
        results = _translate_to_original(SAMPLE_MAPPINGS, 2000, 5000)
        assert len(results) == 2
        assert results[0]["original_start_ms"] == 4000
        assert results[0]["original_end_ms"] == 5000
        assert results[1]["original_start_ms"] == 12000
        assert results[1]["original_end_ms"] == 13500

    def test_segment_in_silence_gap(self):
        results = _translate_to_original(SAMPLE_MAPPINGS, 3000, 3500)
        assert len(results) == 0

    def test_segment_no_overlap(self):
        results = _translate_to_original(SAMPLE_MAPPINGS, 14000, 15000)
        assert len(results) == 0

    def test_segment_covers_entire_mapping(self):
        results = _translate_to_original(SAMPLE_MAPPINGS, 0, 3000)
        assert len(results) == 1
        assert results[0]["original_start_ms"] == 2000
        assert results[0]["original_end_ms"] == 5000
        assert results[0]["duration_ms"] == 3000

    def test_malformed_mapping_skipped(self):
        bad_mappings = [{"concat_start_ms": 0, "concat_end_ms": 1000}]
        results = _translate_to_original(bad_mappings, 0, 500)
        assert len(results) == 0

    def test_empty_mappings(self):
        results = _translate_to_original([], 0, 1000)
        assert len(results) == 0


class TestTimestampMapperWithMappings:

    def test_single_segment_maps_correctly(self):
        stage = TimestampMapperStage()
        task = _make_task(
            {"start_ms": 500, "end_ms": 2500, "speaker_id": "speaker_0", "utmos_mos": 4.2},
            mappings=SAMPLE_MAPPINGS,
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["original_file"] == "/data/audio.wav"
        assert result.data["original_start_ms"] == 2500
        assert result.data["original_end_ms"] == 4500
        assert result.data["duration_ms"] == 2000
        assert result.data["duration_sec"] == 2.0
        assert result.data["speaker_id"] == "speaker_0"
        assert result.data["utmos_mos"] == 4.2

    def test_cross_boundary_segment_rejected(self):
        stage = TimestampMapperStage()
        task = _make_task({"start_ms": 2000, "end_ms": 5000}, mappings=SAMPLE_MAPPINGS)

        result = stage.process(task)

        assert result == []

    def test_segment_in_silence_gap_produces_no_output(self):
        stage = TimestampMapperStage()
        task = _make_task({"start_ms": 3000, "end_ms": 3500}, mappings=SAMPLE_MAPPINGS)

        result = stage.process(task)

        assert result == []

    def test_invalid_range_skipped(self):
        stage = TimestampMapperStage()
        task = _make_task({"start_ms": 5000, "end_ms": 2000}, mappings=SAMPLE_MAPPINGS)

        result = stage.process(task)

        assert result == []

    def test_waveform_stripped_from_output(self):
        stage = TimestampMapperStage()
        task = _make_task(
            {
                "start_ms": 0,
                "end_ms": 3000,
                "waveform": torch.randn(1, 48000),
                "audio": b"fake",
                "audio_filepath": "/data/audio.wav",
                "segment_num": 0,
                "speaker_id": "speaker_0",
            },
            mappings=SAMPLE_MAPPINGS,
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert "waveform" not in result.data
        assert "audio" not in result.data
        assert "audio_filepath" not in result.data
        assert "segment_num" not in result.data
        assert "speaker_id" in result.data


class TestTimestampMapperNoMappings:

    def test_no_mapping_uses_start_end_directly(self):
        stage = TimestampMapperStage()
        task = _make_task(
            {"start_ms": 1000, "end_ms": 4000, "original_file": "/data/audio.wav", "speaker_id": "speaker_0"},
            mappings=None,
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["original_file"] == "/data/audio.wav"
        assert result.data["original_start_ms"] == 1000
        assert result.data["original_end_ms"] == 4000
        assert result.data["duration_ms"] == 3000
        assert result.data["duration_sec"] == 3.0

    def test_no_mapping_falls_back_to_duration_sec(self):
        stage = TimestampMapperStage()
        task = _make_task({"original_file": "/data/audio.wav", "duration_sec": 5.0}, mappings=None)

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["duration_ms"] == 5000

    def test_no_mapping_falls_back_to_waveform_length(self):
        stage = TimestampMapperStage()
        task = _make_task(
            {"original_file": "/data/audio.wav", "waveform": torch.randn(1, 96000), "sample_rate": 48000},
            mappings=None,
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["duration_ms"] == 2000
        assert "waveform" not in result.data

    def test_no_mapping_uses_audio_filepath_as_fallback(self):
        stage = TimestampMapperStage()
        task = _make_task({"audio_filepath": "/data/fallback.wav", "start_ms": 0, "end_ms": 1000}, mappings=None)

        result = stage.process(task)

        assert result.data["original_file"] == "/data/fallback.wav"


class TestPassthroughKeys:

    def test_default_passes_all_non_stripped_keys(self):
        stage = TimestampMapperStage()
        task = _make_task(
            {
                "start_ms": 0, "end_ms": 3000,
                "speaker_id": "speaker_0", "utmos_mos": 4.2,
                "band_prediction": "full_band", "sample_rate": 48000, "is_mono": True,
            },
            mappings=SAMPLE_MAPPINGS,
        )

        result = stage.process(task)

        assert result.data["speaker_id"] == "speaker_0"
        assert result.data["utmos_mos"] == 4.2
        assert result.data["band_prediction"] == "full_band"
        assert result.data["sample_rate"] == 48000
        assert result.data["is_mono"] is True

    def test_explicit_passthrough_keys_filters_output(self):
        stage = TimestampMapperStage(passthrough_keys=["speaker_id", "utmos_mos"])
        task = _make_task(
            {
                "start_ms": 0, "end_ms": 3000,
                "speaker_id": "speaker_0", "utmos_mos": 4.2,
                "band_prediction": "full_band", "sample_rate": 48000, "is_mono": True,
            },
            mappings=SAMPLE_MAPPINGS,
        )

        result = stage.process(task)

        assert result.data["speaker_id"] == "speaker_0"
        assert result.data["utmos_mos"] == 4.2
        assert "band_prediction" not in result.data
        assert "sample_rate" not in result.data
        assert "is_mono" not in result.data
        assert "original_file" in result.data
        assert "duration_ms" in result.data

    def test_passthrough_keys_missing_key_ignored(self):
        stage = TimestampMapperStage(passthrough_keys=["speaker_id", "nonexistent_key"])
        task = _make_task({"start_ms": 0, "end_ms": 3000, "speaker_id": "speaker_0"}, mappings=SAMPLE_MAPPINGS)

        result = stage.process(task)

        assert result.data["speaker_id"] == "speaker_0"
        assert "nonexistent_key" not in result.data


class TestTimestampMapperParams:

    def test_default_passthrough_keys(self):
        stage = TimestampMapperStage()
        assert stage.passthrough_keys is None

    def test_custom_passthrough_keys(self):
        stage = TimestampMapperStage(passthrough_keys=["a", "b"])
        assert stage.passthrough_keys == ["a", "b"]


class TestEdgeCases:

    def test_none_values_not_passed_through(self):
        stage = TimestampMapperStage()
        task = _make_task(
            {"start_ms": 0, "end_ms": 3000, "speaker_id": None, "utmos_mos": 4.0},
            mappings=SAMPLE_MAPPINGS,
        )

        result = stage.process(task)

        assert "speaker_id" not in result.data
        assert result.data["utmos_mos"] == 4.0
