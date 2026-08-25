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

import os
import pickle
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from pydub import AudioSegment

from nemo_curator.stages.audio.segmentation.speaker_separation import SpeakerSeparationStage
from nemo_curator.stages.audio.segmentation.speaker_separation_module.speaker_sep import (
    SpeakerResult,
    SpeakerSeparator,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


def _make_audio_segment(duration_ms: int = 5000, sample_rate: int = 48000) -> AudioSegment:
    return AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)


def _make_task(duration_sec: float = 10.0, sample_rate: int = 48000) -> AudioTask:
    num_samples = int(duration_sec * sample_rate)
    return AudioTask(
        data={"waveform": torch.randn(1, num_samples), "sample_rate": sample_rate},
        dataset_name="test",
    )


class TestSpeakerSeparationStage:
    def test_ray_stage_spec(self) -> None:
        stage = SpeakerSeparationStage()

        assert stage.ray_stage_spec()["is_fanout_stage"] is True

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_process_returns_per_speaker_tasks(self, mock_init: MagicMock) -> None:
        stage = SpeakerSeparationStage(min_duration=0.5)

        separator = MagicMock()
        speaker_data = {
            "speaker_0": SpeakerResult(_make_audio_segment(3000), 3.0, [(0.0, 3.0)]),
            "speaker_1": SpeakerResult(_make_audio_segment(4000), 4.0, [(0.0, 4.0)]),
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
            assert "duration" in r.data

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_process_output_keys(self, mock_init: MagicMock) -> None:
        stage = SpeakerSeparationStage(min_duration=0.5)

        separator = MagicMock()
        separator.get_speaker_audio_data.return_value = {
            "spk_0": SpeakerResult(_make_audio_segment(5000), 5.0, [(0.0, 5.0)]),
        }
        stage._separator = separator

        result = stage.process(_make_task())

        assert len(result) == 1
        item = result[0].data
        assert item["speaker_id"] == "spk_0"
        assert item["num_speakers"] == 1
        assert item["duration"] == 5.0
        assert "waveform" in item
        assert "sample_rate" in item

    # --- output residency (write-to-disk extension) ---

    def test_default_output_is_in_memory_only(self) -> None:
        """Regression: default config emits a tensor and sets no disk gate/path."""
        contract = SpeakerSeparationStage().describe()
        assert contract.writes.produces == ["tensor"]
        assert contract.gates.writes_to_disk is False
        assert "audio_filepath" not in contract.writes.data_keys

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_write_to_disk_persists_and_sets_path(self, mock_init: MagicMock, tmp_path) -> None:  # noqa: ANN001
        stage = SpeakerSeparationStage(min_duration=0.5, write_to_disk=True, separated_audio_dir=str(tmp_path / "sep"))
        separator = MagicMock()
        separator.get_speaker_audio_data.return_value = {
            "spk_0": SpeakerResult(_make_audio_segment(3000), 3.0, [(0.0, 3.0)]),
        }
        stage._separator = separator
        item = stage.process(_make_task())[0].data
        # default keep_waveform_in_task=True -> waveform AND a written per-speaker file
        assert "waveform" in item
        assert "audio_filepath" in item
        assert os.path.exists(item["audio_filepath"])

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_write_to_disk_only_drops_waveform(self, mock_init: MagicMock, tmp_path) -> None:  # noqa: ANN001
        stage = SpeakerSeparationStage(
            min_duration=0.5,
            write_to_disk=True,
            separated_audio_dir=str(tmp_path / "sep"),
            keep_waveform_in_task=False,
        )
        separator = MagicMock()
        separator.get_speaker_audio_data.return_value = {
            "spk_0": SpeakerResult(_make_audio_segment(3000), 3.0, [(0.0, 3.0)]),
        }
        stage._separator = separator
        item = stage.process(_make_task())[0].data
        assert "waveform" not in item
        assert os.path.exists(item["audio_filepath"])

    def test_requires_dir_when_write_to_disk(self) -> None:
        with pytest.raises(ValueError, match="separated_audio_dir"):
            SpeakerSeparationStage(write_to_disk=True)

    def test_requires_at_least_one_output_sink(self) -> None:
        with pytest.raises(ValueError, match="keep_waveform_in_task or write_to_disk"):
            SpeakerSeparationStage(keep_waveform_in_task=False)

    @patch("nemo_curator.stages.audio.segmentation.speaker_separation.SpeakerSeparationStage._initialize_separator")
    def test_min_duration_filters_short_speakers(self, mock_init: MagicMock) -> None:
        stage = SpeakerSeparationStage(min_duration=2.0)

        separator = MagicMock()
        separator.get_speaker_audio_data.return_value = {
            "speaker_0": SpeakerResult(_make_audio_segment(5000), 5.0, [(0.0, 5.0)]),
            "speaker_1": SpeakerResult(_make_audio_segment(1000), 1.0, [(0.0, 1.0)]),
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


class _TinyAudioSegment:
    """A pydub-shaped stub: just enough for the separator to hand back audio."""

    sample_width = 2
    channels = 1
    frame_rate = 16000

    def get_array_of_samples(self) -> list[int]:
        return [0, 500, -500, 0] * 100


# Lifted from tests/stages/audio/test_agent_simulation_pipelines.py: it drives only
# SpeakerSeparationStage, and was the sole coverage of fan-out metadata isolation.
def test_agent_fanout_children_have_isolated_metadata() -> None:
    """Fan-out children must own independent _metadata / _stage_perf copies.

    Pins the de-aliasing fix behaviorally: mutating one child must not leak into a
    sibling or the parent (the shared-reference bug class).
    """

    def fake_speaker_audio_data(*_args: Any, **_kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        return {
            "spk0": SimpleNamespace(audio=_TinyAudioSegment(), duration=0.25, diar_segments=[(0.0, 0.25)]),
            "spk1": SimpleNamespace(audio=_TinyAudioSegment(), duration=0.30, diar_segments=[(0.25, 0.55)]),
        }

    stage = SpeakerSeparationStage(
        input_residency="waveform",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        min_duration=0.1,
        resources=Resources(gpus=0.0),
    )
    stage._separator = SimpleNamespace(get_speaker_audio_data=fake_speaker_audio_data)

    parent = AudioTask(
        dataset_name="t",
        data={"agent_waveform": torch.randn(1, 9600), "agent_sr": 16000},
        _metadata={"trace": "kept"},
        _stage_perf=["fanout-input"],
    )
    children = stage.process(parent)
    assert len(children) == 2

    children[0]._metadata["mutated"] = True
    children[0]._stage_perf.append("child0-only")

    assert "mutated" not in children[1]._metadata
    assert "mutated" not in parent._metadata
    assert "child0-only" not in children[1]._stage_perf
    assert "child0-only" not in parent._stage_perf
    assert children[1]._metadata["trace"] == "kept"
