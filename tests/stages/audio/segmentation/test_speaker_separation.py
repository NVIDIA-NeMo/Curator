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

import pickle
from unittest.mock import MagicMock, patch

import torch
from pydub import AudioSegment

from nemo_curator.stages.audio.segmentation.speaker_separation import SpeakerSeparationStage
from nemo_curator.stages.audio.segmentation.speaker_separation_module.speaker_sep import SpeakerSeparator
from nemo_curator.tasks import AudioTask


def _make_audio_segment(duration_ms: int = 5000, sample_rate: int = 48000) -> AudioSegment:
    return AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)


def _make_task(duration_sec: float = 10.0, sample_rate: int = 48000) -> AudioTask:
    num_samples = int(duration_sec * sample_rate)
    return AudioTask(
        data={"waveform": torch.randn(1, num_samples), "sample_rate": sample_rate},
        task_id="test",
        dataset_name="test",
    )


class TestSpeakerSeparationStage:
    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_process_returns_per_speaker_tasks(self, mock_init: MagicMock) -> None:
        stage = SpeakerSeparationStage(min_duration=0.5)

        separator = MagicMock()
        speaker_data = {
            "speaker_0": (_make_audio_segment(3000), 3.0),
            "speaker_1": (_make_audio_segment(4000), 4.0),
        }
        separator.get_speaker_audio_data.return_value = speaker_data
        stage._separator = separator

        result = stage.process(_make_task())

        assert isinstance(result, list)
        assert len(result) == 2
        for r in result:
            assert isinstance(r, AudioTask)
            assert "speaker_id" in r.data
            assert "num_speakers" in r.data
            assert r.data["num_speakers"] == 2
            assert "duration_sec" in r.data

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_process_output_keys(self, mock_init: MagicMock) -> None:
        stage = SpeakerSeparationStage(min_duration=0.5)

        separator = MagicMock()
        separator.get_speaker_audio_data.return_value = {
            "spk_0": (_make_audio_segment(5000), 5.0),
        }
        stage._separator = separator

        result = stage.process(_make_task())

        assert len(result) == 1
        item = result[0].data
        assert item["speaker_id"] == "spk_0"
        assert item["num_speakers"] == 1
        assert item["duration_sec"] == 5.0
        assert "waveform" in item
        assert "sample_rate" in item

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_min_duration_filters_short_speakers(self, mock_init: MagicMock) -> None:
        stage = SpeakerSeparationStage(min_duration=2.0)

        separator = MagicMock()
        separator.get_speaker_audio_data.return_value = {
            "speaker_0": (_make_audio_segment(5000), 5.0),
            "speaker_1": (_make_audio_segment(1000), 1.0),
        }
        stage._separator = separator

        result = stage.process(_make_task())

        assert len(result) == 1
        assert result[0].data["speaker_id"] == "speaker_0"

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_no_speakers_returns_empty(self, mock_init: MagicMock) -> None:
        stage = SpeakerSeparationStage()

        separator = MagicMock()
        separator.get_speaker_audio_data.return_value = {}
        stage._separator = separator

        result = stage.process(_make_task())

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_no_audio_no_filepath_skipped(self, mock_init: MagicMock) -> None:
        stage = SpeakerSeparationStage()
        stage._separator = MagicMock()

        task = AudioTask(
            data={"some_key": "value"},
            task_id="test",
            dataset_name="test",
        )
        result = stage.process(task)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_separator_not_available(self) -> None:
        stage = SpeakerSeparationStage()
        stage._separator = None

        with patch.object(stage, "_initialize_separator"):
            try:
                result = stage.process(_make_task())
            except RuntimeError:
                result = "raised"

        assert result == "raised"

    def test_pickling(self) -> None:
        stage = SpeakerSeparationStage(min_duration=1.0, exclude_overlaps=False)
        pickled = pickle.dumps(stage)
        restored = pickle.loads(pickled)  # noqa: S301
        assert restored.min_duration == 1.0
        assert restored.exclude_overlaps is False
        assert restored._separator is None

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_separator_exception_skips_task(self, mock_init: MagicMock) -> None:
        stage = SpeakerSeparationStage(min_duration=0.5)

        separator = MagicMock()
        separator.get_speaker_audio_data.side_effect = RuntimeError("Simulated crash")
        stage._separator = separator

        result = stage.process(_make_task())

        assert isinstance(result, list)
        assert len(result) == 0


def _make_separator() -> SpeakerSeparator:
    """Create a SpeakerSeparator with mocked model loading."""
    with patch.object(SpeakerSeparator, "_load_model"):
        return SpeakerSeparator(
            model_name="mock",
            config={"speaker_gap_threshold": 0.1, "speaker_min_duration": 0.5, "speaker_buffer_time": 0.5},
        )


class TestMergeAdjacentSegments:
    def test_empty_segments(self) -> None:
        sep = _make_separator()
        assert sep.merge_adjacent_segments([], gap_threshold=0.1) == []

    def test_single_segment(self) -> None:
        sep = _make_separator()
        result = sep.merge_adjacent_segments([(1.0, 3.0)], gap_threshold=0.1)
        assert result == [(1.0, 3.0)]

    def test_merge_close_segments(self) -> None:
        sep = _make_separator()
        segments = [(0.0, 1.0), (1.05, 2.0), (2.05, 3.0)]
        result = sep.merge_adjacent_segments(segments, gap_threshold=0.1)
        assert len(result) == 1
        assert result[0] == (0.0, 3.0)

    def test_no_merge_distant_segments(self) -> None:
        sep = _make_separator()
        segments = [(0.0, 1.0), (2.0, 3.0)]
        result = sep.merge_adjacent_segments(segments, gap_threshold=0.1)
        assert len(result) == 2

    def test_unsorted_input_gets_sorted(self) -> None:
        sep = _make_separator()
        segments = [(2.0, 3.0), (0.0, 1.0), (1.05, 2.0)]
        result = sep.merge_adjacent_segments(segments, gap_threshold=0.1)
        assert len(result) == 1
        assert result[0] == (0.0, 3.0)

    def test_gap_within_threshold_merges(self) -> None:
        sep = _make_separator()
        segments = [(0.0, 1.0), (1.08, 2.0)]
        result = sep.merge_adjacent_segments(segments, gap_threshold=0.1)
        assert len(result) == 1

    def test_gap_just_over_threshold(self) -> None:
        sep = _make_separator()
        segments = [(0.0, 1.0), (1.2, 2.0)]
        result = sep.merge_adjacent_segments(segments, gap_threshold=0.1)
        assert len(result) == 2


class TestFilterShortSegments:
    def test_all_pass(self) -> None:
        sep = _make_separator()
        segs = {"spk0": [(0.0, 5.0), (6.0, 10.0)]}
        result = sep.filter_short_segments(segs, min_duration=1.0)
        assert len(result["spk0"]) == 2

    def test_all_filtered(self) -> None:
        sep = _make_separator()
        segs = {"spk0": [(0.0, 0.3), (1.0, 1.4)]}
        result = sep.filter_short_segments(segs, min_duration=1.0)
        assert len(result["spk0"]) == 0

    def test_mixed_pass_and_filter(self) -> None:
        sep = _make_separator()
        segs = {"spk0": [(0.0, 0.3), (1.0, 5.0)], "spk1": [(0.0, 0.1)]}
        result = sep.filter_short_segments(segs, min_duration=1.0)
        assert len(result["spk0"]) == 1
        assert result["spk0"][0] == (1.0, 5.0)
        assert len(result["spk1"]) == 0

    def test_exact_min_duration_passes(self) -> None:
        sep = _make_separator()
        segs = {"spk0": [(0.0, 1.0)]}
        result = sep.filter_short_segments(segs, min_duration=1.0)
        assert len(result["spk0"]) == 1


class TestCleanCutOverlappingSegments:
    def test_no_overlap(self) -> None:
        sep = _make_separator()
        segs = {"spk0": [(0.0, 2.0)], "spk1": [(3.0, 5.0)]}
        result = sep.clean_cut_overlapping_segments(segs)
        assert result["spk0"] == [(0.0, 2.0)]
        assert result["spk1"] == [(3.0, 5.0)]

    def test_full_overlap_splits(self) -> None:
        sep = _make_separator()
        segs = {"spk0": [(0.0, 5.0)], "spk1": [(2.0, 4.0)]}
        result = sep.clean_cut_overlapping_segments(segs)
        total_spk0 = sum(e - s for s, e in result["spk0"])
        total_spk1 = sum(e - s for s, e in result["spk1"])
        assert total_spk0 + total_spk1 <= 5.0

    def test_empty_input(self) -> None:
        sep = _make_separator()
        result = sep.clean_cut_overlapping_segments({})
        assert result == {}

    def test_single_speaker_no_change(self) -> None:
        sep = _make_separator()
        segs = {"spk0": [(0.0, 2.0), (3.0, 5.0)]}
        result = sep.clean_cut_overlapping_segments(segs)
        assert result["spk0"] == [(0.0, 2.0), (3.0, 5.0)]

    def test_adjacent_segments_no_cut(self) -> None:
        sep = _make_separator()
        segs = {"spk0": [(0.0, 2.0)], "spk1": [(2.0, 4.0)]}
        result = sep.clean_cut_overlapping_segments(segs)
        assert result["spk0"] == [(0.0, 2.0)]
        assert result["spk1"] == [(2.0, 4.0)]


class TestExcludeOverlappingSegments:
    def test_no_overlap_keeps_all(self) -> None:
        sep = _make_separator()
        segs = {"spk0": [(0.0, 2.0)], "spk1": [(4.0, 6.0)]}
        result = sep.exclude_overlapping_segments(segs, buffer_time=0.0)
        assert result["spk0"] == [(0.0, 2.0)]
        assert result["spk1"] == [(4.0, 6.0)]

    def test_full_overlap_excludes_both(self) -> None:
        sep = _make_separator()
        segs = {"spk0": [(0.0, 5.0)], "spk1": [(0.0, 5.0)]}
        result = sep.exclude_overlapping_segments(segs, buffer_time=0.0)
        assert len(result["spk0"]) == 0
        assert len(result["spk1"]) == 0

    def test_partial_overlap_keeps_non_overlapping_parts(self) -> None:
        sep = _make_separator()
        segs = {"spk0": [(0.0, 3.0)], "spk1": [(2.0, 5.0)]}
        result = sep.exclude_overlapping_segments(segs, buffer_time=0.0)
        spk0_dur = sum(e - s for s, e in result["spk0"])
        spk1_dur = sum(e - s for s, e in result["spk1"])
        assert spk0_dur > 0
        assert spk1_dur > 0
        assert spk0_dur <= 2.0
        assert spk1_dur <= 3.0

    def test_buffer_time_shrinks_segments(self) -> None:
        sep = _make_separator()
        segs = {"spk0": [(0.0, 3.0)], "spk1": [(2.0, 5.0)]}
        result_no_buf = sep.exclude_overlapping_segments(segs, buffer_time=0.0)
        result_with_buf = sep.exclude_overlapping_segments(segs, buffer_time=0.5)
        dur_no_buf = sum(e - s for s, e in result_no_buf["spk0"]) + sum(e - s for s, e in result_no_buf["spk1"])
        dur_with_buf = sum(e - s for s, e in result_with_buf["spk0"]) + sum(e - s for s, e in result_with_buf["spk1"])
        assert dur_with_buf <= dur_no_buf

    def test_empty_input(self) -> None:
        sep = _make_separator()
        result = sep.exclude_overlapping_segments({}, buffer_time=0.0)
        assert result == {}
