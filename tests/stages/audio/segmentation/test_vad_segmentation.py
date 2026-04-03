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

from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.stages.audio.segmentation.vad_segmentation import VADSegmentationStage
from nemo_curator.tasks import AudioTask


class TestVADSegmentationStage:
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_process_returns_segments(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_model = MagicMock()
        mock_load_vad.return_value = mock_model

        sr = 48000
        mock_get_ts.return_value = [
            {"start": 0, "end": sr * 3},
            {"start": sr * 5, "end": sr * 8},
        ]

        waveform = torch.randn(1, sr * 10)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": sr},
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage(min_duration_sec=1.0, max_duration_sec=30.0)
        stage.setup()
        result = stage.process(task)

        assert isinstance(result, list)
        assert len(result) == 2
        for seg in result:
            assert isinstance(seg, AudioTask)
            assert "waveform" in seg.data
            assert "start_ms" in seg.data
            assert "end_ms" in seg.data
            assert "segment_num" in seg.data
            assert "duration_sec" in seg.data

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_process_output_keys(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_load_vad.return_value = MagicMock()

        sr = 48000
        mock_get_ts.return_value = [{"start": 0, "end": sr * 5}]

        waveform = torch.randn(1, sr * 10)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": sr},
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage(min_duration_sec=1.0)
        stage.setup()
        result = stage.process(task)

        assert result[0].data["start_ms"] == 0
        assert result[0].data["segment_num"] == 0
        assert result[0].data["duration_sec"] > 0
        assert result[0].data["sample_rate"] == sr

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_empty_speech_returns_empty(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_load_vad.return_value = MagicMock()
        mock_get_ts.return_value = []

        waveform = torch.randn(1, 48000 * 5)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": 48000},
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage()
        stage.setup()
        result = stage.process(task)

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_segment_numbering(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_load_vad.return_value = MagicMock()

        sr = 48000
        mock_get_ts.return_value = [
            {"start": 0, "end": sr * 2},
            {"start": sr * 3, "end": sr * 5},
            {"start": sr * 6, "end": sr * 8},
        ]

        waveform = torch.randn(1, sr * 10)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": sr},
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage(min_duration_sec=0.5)
        stage.setup()
        result = stage.process(task)

        assert len(result) == 3
        for i, seg in enumerate(result):
            assert seg.data["segment_num"] == i

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_missing_waveform_and_filepath_skipped(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_load_vad.return_value = MagicMock()

        task = AudioTask(
            data={"some_key": "value"},
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage()
        stage.setup()
        result = stage.process(task)

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_nested_mode_returns_single_task(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_load_vad.return_value = MagicMock()

        sr = 48000
        mock_get_ts.return_value = [
            {"start": 0, "end": sr * 3},
            {"start": sr * 5, "end": sr * 8},
        ]

        waveform = torch.randn(1, sr * 10)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": sr},
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage(min_duration_sec=1.0, nested=True)
        stage.setup()
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert "segments" in result.data
        assert len(result.data["segments"]) == 2
        for seg in result.data["segments"]:
            assert "waveform" in seg
            assert "start_ms" in seg
            assert "end_ms" in seg
            assert "segment_num" in seg
            assert "duration_sec" in seg
            assert "original_file" in seg

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_nested_mode_no_speech_returns_empty(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_load_vad.return_value = MagicMock()
        mock_get_ts.return_value = []

        waveform = torch.randn(1, 48000 * 5)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": 48000},
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage(nested=True)
        stage.setup()
        result = stage.process(task)

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_nested_mode_ray_stage_spec_no_fanout(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        stage = VADSegmentationStage(nested=True)
        spec = stage.ray_stage_spec()
        assert spec == {}

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_non_nested_mode_ray_stage_spec_has_fanout(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        stage = VADSegmentationStage(nested=False)
        spec = stage.ray_stage_spec()
        assert spec[RayStageSpecKeys.IS_FANOUT_STAGE] is True

    def test_pickling(self) -> None:
        stage = VADSegmentationStage(min_duration_sec=2.0, threshold=0.6)
        pickled = pickle.dumps(stage)
        restored = pickle.loads(pickled)  # noqa: S301
        assert restored.min_duration_sec == 2.0
        assert restored.threshold == 0.6
        assert restored._vad_model is None
