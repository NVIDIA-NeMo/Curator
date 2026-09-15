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

import pytest
import torch

from nemo_curator.stages.audio.preprocessing.concatenation import SegmentConcatenationStage
from nemo_curator.tasks import AudioTask


def _make_segment_dict(duration_ms: int = 1000, sample_rate: int = 48000, segment_num: int = 0) -> dict:
    num_samples = int(sample_rate * duration_ms / 1000)
    return {
        "waveform": torch.randn(1, num_samples),
        "sample_rate": sample_rate,
        "original_file": "test.wav",
        "start_ms": 0,
        "end_ms": duration_ms,
        "segment_num": segment_num,
    }


def _make_nested_task(segments: list[dict]) -> AudioTask:
    return AudioTask(
        data={"segments": segments, "original_file": "test.wav"},
        dataset_name="ds",
    )


class TestSegmentConcatenationStage:
    def test_process_batch_concatenates_segments(self) -> None:
        segments = [
            _make_segment_dict(duration_ms=2000, segment_num=0),
            _make_segment_dict(duration_ms=3000, segment_num=1),
        ]
        task = _make_nested_task(segments)

        stage = SegmentConcatenationStage(silence_duration_sec=1.0)
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        out = result.data
        assert out["num_segments"] == 2
        expected_duration = (2000 + 1000 + 3000) / 1000.0
        assert abs(out["total_duration_sec"] - expected_duration) < 0.1

    def test_process_batch_empty_input(self) -> None:
        stage = SegmentConcatenationStage()
        result = stage.process_batch([])
        assert result == []

    def test_process_batch_single_segment(self) -> None:
        segments = [_make_segment_dict(duration_ms=5000)]
        task = _make_nested_task(segments)

        stage = SegmentConcatenationStage(silence_duration_sec=0.5)
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["num_segments"] == 1
        assert abs(result.data["total_duration_sec"] - 5.0) < 0.1

    def test_silence_duration_in_output(self) -> None:
        segments = [
            _make_segment_dict(duration_ms=1000, segment_num=0),
            _make_segment_dict(duration_ms=1000, segment_num=1),
        ]
        task = _make_nested_task(segments)

        stage = SegmentConcatenationStage(silence_duration_sec=2.0)
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        combined = result.data["waveform"]
        sample_rate = result.data["sample_rate"]
        combined_duration_sec = combined.shape[-1] / sample_rate
        expected = 1.0 + 2.0 + 1.0
        assert abs(combined_duration_sec - expected) < 0.1

    def test_no_waveform_in_tasks(self) -> None:
        task = AudioTask(
            data={"segments": [{"other_key": "value"}]},
            dataset_name="ds",
        )
        stage = SegmentConcatenationStage()
        result = stage.process(task)
        assert result == []

    def test_missing_segments_key_raises(self) -> None:
        task = AudioTask(data={"other_key": "value"}, dataset_name="ds")
        stage = SegmentConcatenationStage()
        with pytest.raises(ValueError):  # noqa: PT011
            stage.process(task)

    def test_empty_segments_returns_empty(self) -> None:
        task = _make_nested_task([])
        stage = SegmentConcatenationStage()
        result = stage.process(task)
        assert result == []

    # --- output residency (write-to-disk extension) ---

    def test_default_output_is_in_memory_only(self) -> None:
        """Regression: default config keeps the in-memory waveform and writes no path."""
        result = SegmentConcatenationStage().process(_make_nested_task([_make_segment_dict(duration_ms=1000)]))
        assert "waveform" in result.data
        assert "audio_filepath" not in result.data

    def test_write_to_disk_persists_and_sets_path(self, tmp_path) -> None:  # noqa: ANN001
        out_dir = tmp_path / "concat"
        stage = SegmentConcatenationStage(write_to_disk=True, output_dir=str(out_dir))
        segments = [
            _make_segment_dict(duration_ms=1000, segment_num=0),
            _make_segment_dict(duration_ms=1000, segment_num=1),
        ]
        result = stage.process(_make_nested_task(segments))
        assert isinstance(result, AudioTask)
        # default keep_waveform_in_task=True -> both the waveform AND a written path
        assert "waveform" in result.data
        assert "audio_filepath" in result.data
        assert os.path.exists(result.data["audio_filepath"])

    def test_write_to_disk_only_drops_waveform(self, tmp_path) -> None:  # noqa: ANN001
        stage = SegmentConcatenationStage(
            write_to_disk=True, output_dir=str(tmp_path / "c"), keep_waveform_in_task=False
        )
        result = stage.process(_make_nested_task([_make_segment_dict(duration_ms=1000)]))
        assert "waveform" not in result.data
        assert os.path.exists(result.data["audio_filepath"])

    def test_requires_output_dir_when_write_to_disk(self) -> None:
        with pytest.raises(ValueError, match="output_dir"):
            SegmentConcatenationStage(write_to_disk=True)

    def test_requires_at_least_one_output_sink(self) -> None:
        with pytest.raises(ValueError, match="keep_waveform_in_task or write_to_disk"):
            SegmentConcatenationStage(keep_waveform_in_task=False)
