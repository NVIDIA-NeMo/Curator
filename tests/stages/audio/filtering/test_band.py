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

from nemo_curator.stages.audio.configs.band import BandFilterConfig
from nemo_curator.stages.audio.filtering.band import BandFilterStage
from nemo_curator.tasks import AudioBatch


class TestBandFilterConfig:
    """Tests for BandFilterConfig."""

    def test_defaults(self) -> None:
        cfg = BandFilterConfig()
        assert cfg.band_value == "full_band"
        assert cfg.feature_group == "band"
        assert cfg.n_workers == 4
        assert cfg.feature_cache_size == 100

    def test_from_dict(self) -> None:
        cfg = BandFilterConfig.from_dict({"band_value": "narrow_band", "n_workers": 2})
        assert cfg.band_value == "narrow_band"
        assert cfg.n_workers == 2
        assert cfg.feature_group == "band"

    def test_from_dict_none(self) -> None:
        cfg = BandFilterConfig.from_dict(None)
        assert cfg.band_value == "full_band"

    def test_from_dict_ignores_unknown_keys(self) -> None:
        cfg = BandFilterConfig.from_dict({"unknown": 42, "band_value": "narrow_band"})
        assert cfg.band_value == "narrow_band"

    def test_to_dict(self) -> None:
        cfg = BandFilterConfig(band_value="narrow_band")
        d = cfg.to_dict()
        assert d["band_value"] == "narrow_band"
        assert d["feature_group"] == "band"

    def test_roundtrip(self) -> None:
        original = BandFilterConfig(band_value="narrow_band", n_workers=8)
        restored = BandFilterConfig.from_dict(original.to_dict())
        assert restored.band_value == original.band_value
        assert restored.n_workers == original.n_workers

    def test_get(self) -> None:
        cfg = BandFilterConfig()
        assert cfg.get("band_value") == "full_band"
        assert cfg.get("nonexistent", "default") == "default"


class TestBandFilterStage:
    """Tests for BandFilterStage."""

    def test_stage_properties(self) -> None:
        stage = BandFilterStage()
        assert stage.name == "BandFilter"
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == ([], ["band_prediction"])

    def test_config_overrides_params(self) -> None:
        cfg = BandFilterConfig(band_value="narrow_band", n_workers=2)
        stage = BandFilterStage(config=cfg)
        assert stage.band_value == "narrow_band"
        assert stage.n_workers == 2

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_full_band_passes(self, mock_init) -> None:
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        predictor.predict_audio_batch.return_value = ["full_band"]
        stage._predictor = predictor

        waveform = torch.randn(1, 48000)
        batch = AudioBatch(
            data=[{"waveform": waveform, "sample_rate": 48000}],
            task_id="test",
            dataset_name="test",
        )
        result = stage.process(batch)

        assert isinstance(result, AudioBatch)
        assert len(result.data) == 1
        assert result.data[0]["band_prediction"] == "full_band"

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_narrow_band_filtered_out(self, mock_init) -> None:
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        predictor.predict_audio_batch.return_value = ["narrow_band"]
        stage._predictor = predictor

        waveform = torch.randn(1, 48000)
        batch = AudioBatch(
            data=[{"waveform": waveform, "sample_rate": 48000}],
            task_id="test",
            dataset_name="test",
        )
        result = stage.process(batch)

        assert isinstance(result, AudioBatch)
        assert len(result.data) == 0

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_narrow_band_passes_when_configured(self, mock_init) -> None:
        stage = BandFilterStage(band_value="narrow_band")
        predictor = MagicMock()
        predictor.predict_audio_batch.return_value = ["narrow_band"]
        stage._predictor = predictor

        waveform = torch.randn(1, 48000)
        batch = AudioBatch(
            data=[{"waveform": waveform, "sample_rate": 48000}],
            task_id="test",
            dataset_name="test",
        )
        result = stage.process(batch)

        assert len(result.data) == 1
        assert result.data[0]["band_prediction"] == "narrow_band"

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_multiple_items(self, mock_init) -> None:
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        predictor.predict_audio_batch.side_effect = [
            ["full_band"],
            ["narrow_band"],
            ["full_band"],
        ]
        stage._predictor = predictor

        batch = AudioBatch(
            data=[
                {"waveform": torch.randn(1, 48000), "sample_rate": 48000},
                {"waveform": torch.randn(1, 48000), "sample_rate": 48000},
                {"waveform": torch.randn(1, 48000), "sample_rate": 48000},
            ],
            task_id="test",
            dataset_name="test",
        )
        result = stage.process(batch)

        assert len(result.data) == 2
        for item in result.data:
            assert item["band_prediction"] == "full_band"

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_empty_batch(self, mock_init) -> None:
        stage = BandFilterStage(band_value="full_band")
        stage._predictor = MagicMock()

        batch = AudioBatch(data=[], task_id="test", dataset_name="test")
        result = stage.process(batch)

        assert isinstance(result, AudioBatch)
        assert len(result.data) == 0

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_error_prediction_skipped(self, mock_init) -> None:
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        predictor.predict_audio_batch.return_value = ["Error: model failed"]
        stage._predictor = predictor

        batch = AudioBatch(
            data=[{"waveform": torch.randn(1, 48000), "sample_rate": 48000}],
            task_id="test",
            dataset_name="test",
        )
        result = stage.process(batch)

        assert len(result.data) == 0

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_no_waveform_no_filepath_skipped(self, mock_init) -> None:
        stage = BandFilterStage(band_value="full_band")
        stage._predictor = MagicMock()

        batch = AudioBatch(
            data=[{"some_key": "value"}],
            task_id="test",
            dataset_name="test",
        )
        result = stage.process(batch)

        assert len(result.data) == 0

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_preserves_task_metadata(self, mock_init) -> None:
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        predictor.predict_audio_batch.return_value = ["full_band"]
        stage._predictor = predictor

        batch = AudioBatch(
            data=[{"waveform": torch.randn(1, 48000), "sample_rate": 48000}],
            task_id="my-task",
            dataset_name="my-ds",
        )
        result = stage.process(batch)

        assert result.task_id == "my-task"
        assert result.dataset_name == "my-ds"

    def test_predictor_not_available(self) -> None:
        stage = BandFilterStage(band_value="full_band")
        stage._predictor = None

        with patch.object(stage, "_initialize_predictor"):
            batch = AudioBatch(
                data=[{"waveform": torch.randn(1, 48000), "sample_rate": 48000}],
                task_id="test",
                dataset_name="test",
            )
            result = stage.process(batch)

        assert isinstance(result, AudioBatch)
        assert len(result.data) == 0
