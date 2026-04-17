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

import torch

from nemo_curator.stages.audio.postprocessing.timestamp_mapper import (
    _NEVER_PASS_KEYS,
    TimestampMapperStage,
    _translate_to_original,
)
from nemo_curator.tasks import AudioTask


def _make_task(data: dict, task_id: str = "test", metadata: dict | None = None) -> AudioTask:
    t = AudioTask(data=data, task_id=task_id, dataset_name="test_ds")
    if metadata:
        t._metadata = metadata
    return t


class TestTranslateToOriginal:
    """Unit tests for the pure _translate_to_original() function."""

    MAPPINGS = [
        {"concat_start_ms": 0, "concat_end_ms": 2000, "original_file": "a.wav", "original_start_ms": 5000, "original_end_ms": 7000},
        {"concat_start_ms": 2000, "concat_end_ms": 5000, "original_file": "b.wav", "original_start_ms": 0, "original_end_ms": 3000},
        {"concat_start_ms": 5000, "concat_end_ms": 8000, "original_file": "c.wav", "original_start_ms": 10000, "original_end_ms": 13000},
    ]

    def test_single_mapping_exact_match(self) -> None:
        """Segment exactly matches one mapping."""
        results = _translate_to_original(self.MAPPINGS, 0, 2000)
        assert len(results) == 1
        assert results[0]["original_file"] == "a.wav"
        assert results[0]["original_start_ms"] == 5000
        assert results[0]["original_end_ms"] == 7000
        assert results[0]["duration_ms"] == 2000

    def test_single_mapping_partial_overlap(self) -> None:
        """Segment partially overlaps one mapping."""
        results = _translate_to_original(self.MAPPINGS, 500, 1500)
        assert len(results) == 1
        assert results[0]["original_file"] == "a.wav"
        assert results[0]["original_start_ms"] == 5500
        assert results[0]["original_end_ms"] == 6500
        assert results[0]["duration_ms"] == 1000

    def test_cross_boundary_span(self) -> None:
        """Segment spans two mappings — returns both."""
        results = _translate_to_original(self.MAPPINGS, 1500, 3000)
        assert len(results) == 2
        assert results[0]["original_file"] == "a.wav"
        assert results[0]["original_start_ms"] == 6500
        assert results[0]["original_end_ms"] == 7000
        assert results[0]["duration_ms"] == 500
        assert results[1]["original_file"] == "b.wav"
        assert results[1]["original_start_ms"] == 0
        assert results[1]["original_end_ms"] == 1000
        assert results[1]["duration_ms"] == 1000

    def test_silence_gap_no_overlap(self) -> None:
        """Segment falls entirely in a gap between mappings."""
        mappings = [
            {"concat_start_ms": 0, "concat_end_ms": 1000, "original_file": "a.wav", "original_start_ms": 0, "original_end_ms": 1000},
            {"concat_start_ms": 3000, "concat_end_ms": 5000, "original_file": "b.wav", "original_start_ms": 0, "original_end_ms": 2000},
        ]
        results = _translate_to_original(mappings, 1000, 3000)
        assert len(results) == 0

    def test_malformed_mapping_missing_key(self) -> None:
        """Malformed mapping (missing key) is skipped gracefully."""
        mappings = [
            {"concat_start_ms": 0, "concat_end_ms": 2000},
            {"concat_start_ms": 2000, "concat_end_ms": 4000, "original_file": "b.wav", "original_start_ms": 0, "original_end_ms": 2000},
        ]
        results = _translate_to_original(mappings, 0, 4000)
        assert len(results) == 1
        assert results[0]["original_file"] == "b.wav"

    def test_empty_mappings(self) -> None:
        """Empty mappings list returns empty results."""
        results = _translate_to_original([], 0, 1000)
        assert results == []

    def test_no_overlap_before_all_mappings(self) -> None:
        """Segment ends before any mapping starts."""
        mappings = [
            {"concat_start_ms": 5000, "concat_end_ms": 8000, "original_file": "a.wav", "original_start_ms": 0, "original_end_ms": 3000},
        ]
        results = _translate_to_original(mappings, 0, 1000)
        assert results == []


def test_combo4_with_segment_mappings() -> None:
    """Full pipeline: remaps concat-space timestamps to original file positions."""
    mappings = [
        {
            "concat_start_ms": 0,
            "concat_end_ms": 2000,
            "original_file": "test.wav",
            "original_start_ms": 5000,
            "original_end_ms": 7000,
        },
    ]
    task = _make_task(
        {"waveform": torch.randn(1, 48000), "sample_rate": 48000, "start_ms": 100, "end_ms": 1500, "utmos_mos": 4.2},
        metadata={"segment_mappings": mappings},
    )
    stage = TimestampMapperStage()
    result = stage.process(task)

    assert result.data["original_file"] == "test.wav"
    assert result.data["original_start_ms"] == 5100
    assert result.data["original_end_ms"] == 6500
    assert result.data["duration_ms"] == 1400
    assert result.data["utmos_mos"] == 4.2
    assert result.data["sample_rate"] == 48000
    assert "waveform" not in result.data
    assert "start_ms" not in result.data


def test_combo2_vad_fanout_start_end() -> None:
    """VAD fan-out: uses start_ms/end_ms directly."""
    task = _make_task(
        {
            "waveform": torch.randn(1, 48000),
            "sample_rate": 48000,
            "start_ms": 5200,
            "end_ms": 15400,
            "segment_num": 0,
            "duration": 10.2,
            "original_file": "/a.wav",
            "utmos_mos": 4.2,
        }
    )
    stage = TimestampMapperStage()
    result = stage.process(task)

    assert result.data["original_file"] == "/a.wav"
    assert result.data["original_start_ms"] == 5200
    assert result.data["original_end_ms"] == 15400
    assert result.data["duration_ms"] == 10200
    assert abs(result.data["duration"] - 10.2) < 0.01
    assert result.data["utmos_mos"] == 4.2
    assert "waveform" not in result.data
    assert "start_ms" not in result.data
    assert "segment_num" not in result.data


def test_combo3_diar_segments() -> None:
    """Speaker-only: computes span from diar_segments."""
    task = _make_task(
        {
            "waveform": torch.randn(1, 48000),
            "sample_rate": 48000,
            "speaker_id": "speaker_0",
            "num_speakers": 3,
            "duration": 42.6,
            "diar_segments": [(5.2, 15.4), (30.1, 42.8), (100.0, 120.5)],
            "audio_filepath": "/a.wav",
            "sigmos_noise": 4.5,
        }
    )
    stage = TimestampMapperStage()
    result = stage.process(task)

    assert result.data["original_file"] == "/a.wav"
    assert result.data["original_start_ms"] == 5200
    assert result.data["original_end_ms"] == 120500
    assert result.data["duration_ms"] == 115300
    assert abs(result.data["speaking_duration"] - 43.4) < 0.01
    assert len(result.data["diar_segments"]) == 3
    assert result.data["speaker_id"] == "speaker_0"
    assert result.data["num_speakers"] == 3
    assert result.data["sigmos_noise"] == 4.5
    assert "waveform" not in result.data


def test_combo1_duration_fallback() -> None:
    """Filters-only: uses duration from MonoConversion."""
    task = _make_task(
        {
            "audio_filepath": "/a.wav",
            "waveform": torch.randn(1, 48000),
            "sample_rate": 48000,
            "duration": 10.5,
            "is_mono": True,
            "num_samples": 504000,
            "sigmos_ovrl": 3.5,
        }
    )
    stage = TimestampMapperStage()
    result = stage.process(task)

    assert result.data["original_file"] == "/a.wav"
    assert result.data["original_start_ms"] == 0
    assert result.data["original_end_ms"] == 10500
    assert result.data["duration"] == 10.5
    assert result.data["sigmos_ovrl"] == 3.5
    assert result.data["sample_rate"] == 48000
    assert "waveform" not in result.data
    assert "is_mono" not in result.data
    assert "num_samples" not in result.data


def test_never_pass_keys_blocked() -> None:
    """Non-serializable keys are blocked even if in passthrough_keys."""
    task = _make_task(
        {
            "audio_filepath": "/a.wav",
            "waveform": torch.randn(1, 48000),
            "segments": [{"waveform": torch.randn(1, 100)}],
            "duration": 1.0,
            "sigmos_ovrl": 3.0,
        }
    )
    stage = TimestampMapperStage(passthrough_keys=["waveform", "segments", "sigmos_ovrl"])
    result = stage.process(task)

    for key in _NEVER_PASS_KEYS:
        assert key not in result.data, f"{key!r} must never pass through"
    assert result.data["sigmos_ovrl"] == 3.0


def test_default_passthrough_covers_all_filters() -> None:
    """Default passthrough_keys includes all built-in filter scores."""
    task = _make_task(
        {
            "audio_filepath": "/a.wav",
            "duration": 1.0,
            "utmos_mos": 4.2,
            "sigmos_noise": 4.0,
            "sigmos_ovrl": 3.5,
            "sigmos_sig": 3.8,
            "sigmos_col": 4.0,
            "sigmos_disc": 4.2,
            "sigmos_loud": 3.7,
            "sigmos_reverb": 4.9,
            "band_prediction": "full_band",
            "sample_rate": 48000,
        }
    )
    stage = TimestampMapperStage()
    result = stage.process(task)

    assert result.data["utmos_mos"] == 4.2
    assert result.data["sigmos_noise"] == 4.0
    assert result.data["sigmos_ovrl"] == 3.5
    assert result.data["band_prediction"] == "full_band"
    assert result.data["sample_rate"] == 48000


def test_custom_passthrough_keys() -> None:
    """User can restrict output to only specific keys."""
    task = _make_task(
        {
            "audio_filepath": "/a.wav",
            "duration": 1.0,
            "sigmos_ovrl": 3.0,
            "sigmos_noise": 4.0,
            "utmos_mos": 4.2,
            "book_id": "123",
        }
    )
    stage = TimestampMapperStage(passthrough_keys=["sigmos_ovrl", "book_id"])
    result = stage.process(task)

    assert result.data["sigmos_ovrl"] == 3.0
    assert result.data["book_id"] == "123"
    assert "sigmos_noise" not in result.data
    assert "utmos_mos" not in result.data


def test_dataset_metadata_not_in_default_output() -> None:
    """Dataset-specific keys (text, book_id) are excluded by default passthrough."""
    task = _make_task(
        {
            "audio_filepath": "/a.wav",
            "duration": 1.0,
            "text": "hello world",
            "book_id": "123",
            "reader_id": "456",
            "sigmos_ovrl": 3.0,
        }
    )
    stage = TimestampMapperStage()
    result = stage.process(task)

    assert result.data["sigmos_ovrl"] == 3.0
    assert "text" not in result.data
    assert "book_id" not in result.data
    assert "reader_id" not in result.data
