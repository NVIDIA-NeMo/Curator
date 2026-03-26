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
from nemo_curator.tasks import AudioTask


def _make_segment_task(duration_ms: int = 1000, sample_rate: int = 48000, segment_num: int = 0) -> AudioTask:
    num_samples = int(sample_rate * duration_ms / 1000)
    return AudioTask(
        data={
            "waveform": torch.randn(1, num_samples),
            "sample_rate": sample_rate,
            "original_file": "test.wav",
            "start_ms": 0,
            "end_ms": duration_ms,
            "segment_num": segment_num,
        },
        task_id=f"seg_{segment_num}",
        dataset_name="ds",
    )


class TestSegmentConcatenationStage:

    def test_process_batch_concatenates_segments(self) -> None:
        tasks = [
            _make_segment_task(duration_ms=2000, segment_num=0),
            _make_segment_task(duration_ms=3000, segment_num=1),
        ]

        stage = SegmentConcatenationStage(silence_duration_sec=1.0)
        result = stage.process_batch(tasks)

        assert len(result) == 1
        out = result[0].data
        assert out["num_segments"] == 2
        expected_duration = (2000 + 1000 + 3000) / 1000.0
        assert abs(out["total_duration_sec"] - expected_duration) < 0.1

    def test_process_batch_empty_input(self) -> None:
        stage = SegmentConcatenationStage()
        result = stage.process_batch([])
        assert result == []

    def test_process_batch_single_segment(self) -> None:
        tasks = [_make_segment_task(duration_ms=5000)]

        stage = SegmentConcatenationStage(silence_duration_sec=0.5)
        result = stage.process_batch(tasks)

        assert len(result) == 1
        assert result[0].data["num_segments"] == 1
        assert abs(result[0].data["total_duration_sec"] - 5.0) < 0.1

    def test_silence_duration_in_output(self) -> None:
        tasks = [
            _make_segment_task(duration_ms=1000, segment_num=0),
            _make_segment_task(duration_ms=1000, segment_num=1),
        ]

        stage = SegmentConcatenationStage(silence_duration_sec=2.0)
        result = stage.process_batch(tasks)

        combined = result[0].data["waveform"]
        sample_rate = result[0].data["sample_rate"]
        combined_duration_sec = combined.shape[-1] / sample_rate
        expected = 1.0 + 2.0 + 1.0
        assert abs(combined_duration_sec - expected) < 0.1

    def test_no_waveform_in_tasks(self) -> None:
        task = AudioTask(data={"other_key": "value"}, task_id="empty", dataset_name="ds")
        stage = SegmentConcatenationStage()
        result = stage.process_batch([task])
        assert result == []
