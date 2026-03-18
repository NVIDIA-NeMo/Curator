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

from nemo_curator.stages.audio.preprocessing.concatenation import SegmentConcatenationStage
from nemo_curator.tasks import AudioBatch


def _make_waveform_item(duration_ms: int = 1000, sample_rate: int = 48000) -> dict:
    """Create a dict with a torch waveform tensor of a given duration."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return {
        "waveform": torch.randn(1, num_samples),
        "sample_rate": sample_rate,
        "original_file": "test.wav",
        "start_ms": 0,
        "end_ms": duration_ms,
    }


class TestSegmentConcatenationStage:
    """Tests for SegmentConcatenationStage.process()."""

    def test_process_concatenates_segments(self) -> None:
        items = [
            _make_waveform_item(duration_ms=2000),
            _make_waveform_item(duration_ms=3000),
        ]
        batch = AudioBatch(data=items, dataset_name="ds")

        stage = SegmentConcatenationStage(silence_duration_sec=1.0)
        result = stage.process(batch)

        assert isinstance(result, AudioBatch)
        assert len(result.data) == 1
        out = result.data[0]
        assert out["num_segments"] == 2
        expected_duration = (2000 + 1000 + 3000) / 1000.0
        assert abs(out["total_duration_sec"] - expected_duration) < 0.1

    def test_process_empty_input(self) -> None:
        batch = AudioBatch(data=[], dataset_name="ds")
        stage = SegmentConcatenationStage()
        result = stage.process(batch)
        assert isinstance(result, AudioBatch)
        assert len(result.data) == 0

    def test_process_single_segment(self) -> None:
        items = [_make_waveform_item(duration_ms=5000)]
        batch = AudioBatch(data=items, dataset_name="ds")

        stage = SegmentConcatenationStage(silence_duration_sec=0.5)
        result = stage.process(batch)

        assert len(result.data) == 1
        assert result.data[0]["num_segments"] == 1
        assert abs(result.data[0]["total_duration_sec"] - 5.0) < 0.1

    def test_silence_duration_in_output(self) -> None:
        items = [
            _make_waveform_item(duration_ms=1000),
            _make_waveform_item(duration_ms=1000),
        ]
        batch = AudioBatch(data=items, dataset_name="ds")

        stage = SegmentConcatenationStage(silence_duration_sec=2.0)
        result = stage.process(batch)

        combined = result.data[0]["waveform"]
        sample_rate = result.data[0]["sample_rate"]
        combined_duration_sec = combined.shape[-1] / sample_rate
        expected = 1.0 + 2.0 + 1.0
        assert abs(combined_duration_sec - expected) < 0.1

    def test_no_waveform_in_items(self) -> None:
        batch = AudioBatch(data=[{"other_key": "value"}], dataset_name="ds")
        stage = SegmentConcatenationStage()
        result = stage.process(batch)
        assert len(result.data) == 0
