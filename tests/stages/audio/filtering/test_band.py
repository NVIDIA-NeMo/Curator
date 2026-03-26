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

from nemo_curator.stages.audio.filtering.band import BandFilterStage
from nemo_curator.tasks import AudioTask


def _make_task(waveform=None, sample_rate=48000):
    if waveform is None:
        waveform = torch.randn(1, sample_rate)
    return AudioTask(
        data={"waveform": waveform, "sample_rate": sample_rate},
        task_id="test",
        dataset_name="test",
    )


class TestBandFilterStage:

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_full_band_passes(self, mock_init) -> None:
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "full_band"
        stage._predictor = predictor

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert result.data["band_prediction"] == "full_band"

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_narrow_band_filtered_out(self, mock_init) -> None:
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "narrow_band"
        stage._predictor = predictor

        result = stage.process(_make_task())

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_narrow_band_passes_when_configured(self, mock_init) -> None:
        stage = BandFilterStage(band_value="narrow_band")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "narrow_band"
        stage._predictor = predictor

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert result.data["band_prediction"] == "narrow_band"

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_error_prediction_skipped(self, mock_init) -> None:
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "Error: model failed"
        stage._predictor = predictor

        result = stage.process(_make_task())

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_no_waveform_no_filepath_skipped(self, mock_init) -> None:
        stage = BandFilterStage(band_value="full_band")
        stage._predictor = MagicMock()

        task = AudioTask(data={"some_key": "value"}, task_id="test", dataset_name="test")
        result = stage.process(task)

        assert result == []

    def test_predictor_not_available(self) -> None:
        stage = BandFilterStage(band_value="full_band")
        stage._predictor = None

        with patch.object(stage, "_initialize_predictor"):
            result = stage.process(_make_task())

        assert result == []
