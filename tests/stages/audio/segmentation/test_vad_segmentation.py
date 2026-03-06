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

from unittest.mock import MagicMock, patch

import torch

from nemo_curator.stages.audio.configs.vad import VADConfig
from nemo_curator.stages.audio.segmentation.vad_segmentation import VADSegmentationStage
from nemo_curator.tasks import AudioBatch


class TestVADConfig:
    """Tests for VADConfig."""

    def test_defaults(self) -> None:
        cfg = VADConfig()
        assert cfg.min_interval_ms == 500
        assert cfg.min_duration_sec == 2.0
        assert cfg.max_duration_sec == 60.0
        assert cfg.threshold == 0.5
        assert cfg.speech_pad_ms == 300

    def test_from_dict(self) -> None:
        cfg = VADConfig.from_dict({"min_duration_sec": 3.0, "threshold": 0.6})
        assert cfg.min_duration_sec == 3.0
        assert cfg.threshold == 0.6
        assert cfg.max_duration_sec == 60.0

    def test_from_dict_none(self) -> None:
        cfg = VADConfig.from_dict(None)
        assert cfg.threshold == 0.5

    def test_from_dict_ignores_unknown(self) -> None:
        cfg = VADConfig.from_dict({"unknown": 99, "threshold": 0.3})
        assert cfg.threshold == 0.3

    def test_to_dict(self) -> None:
        cfg = VADConfig(min_duration_sec=5.0, max_duration_sec=30.0)
        d = cfg.to_dict()
        assert d["min_duration_sec"] == 5.0
        assert d["max_duration_sec"] == 30.0

    def test_roundtrip(self) -> None:
        original = VADConfig(min_duration_sec=1.5, threshold=0.7, speech_pad_ms=200)
        restored = VADConfig.from_dict(original.to_dict())
        assert restored.min_duration_sec == original.min_duration_sec
        assert restored.threshold == original.threshold
        assert restored.speech_pad_ms == original.speech_pad_ms

    def test_get(self) -> None:
        cfg = VADConfig()
        assert cfg.get("threshold") == 0.5
        assert cfg.get("nonexistent", "default") == "default"


class TestVADSegmentationStage:
    """Tests for VADSegmentationStage."""

    def test_stage_properties(self) -> None:
        stage = VADSegmentationStage()
        assert stage.name == "VADSegmentation"
        assert stage.inputs() == (["data"], [])
        _, output_keys = stage.outputs()
        for key in ["audio", "waveform", "sample_rate", "start_ms", "end_ms",
                     "segment_num", "duration_sec"]:
            assert key in output_keys

    def test_config_overrides_params(self) -> None:
        cfg = VADConfig(min_duration_sec=3.0, max_duration_sec=20.0, threshold=0.7)
        stage = VADSegmentationStage(config=cfg)
        assert stage.min_duration_sec == 3.0
        assert stage.max_duration_sec == 20.0
        assert stage.threshold == 0.7

    def test_ray_stage_spec_is_fanout(self) -> None:
        stage = VADSegmentationStage()
        spec = stage.ray_stage_spec()
        from nemo_curator.backends.experimental.utils import RayStageSpecKeys
        assert spec[RayStageSpecKeys.IS_FANOUT_STAGE] is True

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_process_returns_segments(self, mock_load_vad, mock_get_ts) -> None:
        mock_model = MagicMock()
        mock_load_vad.return_value = mock_model

        sr = 48000
        mock_get_ts.return_value = [
            {"start": 0, "end": sr * 3},
            {"start": sr * 5, "end": sr * 8},
        ]

        waveform = torch.randn(1, sr * 10)
        batch = AudioBatch(
            data=[{"waveform": waveform, "sample_rate": sr}],
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage(min_duration_sec=1.0, max_duration_sec=30.0)
        stage.setup()
        result = stage.process(batch)

        assert isinstance(result, list)
        assert len(result) == 2
        for seg in result:
            assert isinstance(seg, AudioBatch)
            item = seg.data[0]
            assert "audio" in item
            assert "waveform" in item
            assert "start_ms" in item
            assert "end_ms" in item
            assert "segment_num" in item
            assert "duration_sec" in item

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_process_output_keys(self, mock_load_vad, mock_get_ts) -> None:
        mock_load_vad.return_value = MagicMock()

        sr = 48000
        mock_get_ts.return_value = [{"start": 0, "end": sr * 5}]

        waveform = torch.randn(1, sr * 10)
        batch = AudioBatch(
            data=[{"waveform": waveform, "sample_rate": sr}],
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage(min_duration_sec=1.0)
        stage.setup()
        result = stage.process(batch)

        item = result[0].data[0]
        assert item["start_ms"] == 0
        assert item["segment_num"] == 0
        assert item["duration_sec"] > 0
        assert item["sample_rate"] == sr

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_empty_speech_returns_empty(self, mock_load_vad, mock_get_ts) -> None:
        mock_load_vad.return_value = MagicMock()
        mock_get_ts.return_value = []

        waveform = torch.randn(1, 48000 * 5)
        batch = AudioBatch(
            data=[{"waveform": waveform, "sample_rate": 48000}],
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage()
        stage.setup()
        result = stage.process(batch)

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_segment_numbering(self, mock_load_vad, mock_get_ts) -> None:
        mock_load_vad.return_value = MagicMock()

        sr = 48000
        mock_get_ts.return_value = [
            {"start": 0, "end": sr * 2},
            {"start": sr * 3, "end": sr * 5},
            {"start": sr * 6, "end": sr * 8},
        ]

        waveform = torch.randn(1, sr * 10)
        batch = AudioBatch(
            data=[{"waveform": waveform, "sample_rate": sr}],
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage(min_duration_sec=0.5)
        stage.setup()
        result = stage.process(batch)

        assert len(result) == 3
        for i, seg in enumerate(result):
            assert seg.data[0]["segment_num"] == i

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_missing_waveform_and_filepath_skipped(self, mock_load_vad, mock_get_ts) -> None:
        mock_load_vad.return_value = MagicMock()

        batch = AudioBatch(
            data=[{"some_key": "value"}],
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage()
        stage.setup()
        result = stage.process(batch)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_pickling(self) -> None:
        """VADSegmentationStage should be picklable (for Ray workers)."""
        import pickle

        stage = VADSegmentationStage(min_duration_sec=2.0, threshold=0.6)
        pickled = pickle.dumps(stage)
        restored = pickle.loads(pickled)
        assert restored.min_duration_sec == 2.0
        assert restored.threshold == 0.6
        assert restored._vad_model is None
