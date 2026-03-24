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

"""Unit tests for UTMOSFilterStage."""

from unittest.mock import MagicMock, patch

import torch

from nemo_curator.stages.audio.filtering.utmos import UTMOSFilterStage
from nemo_curator.tasks import AudioBatch


def _make_item(duration_s: float = 1.0, sample_rate: int = 16000) -> dict:
    num_samples = int(duration_s * sample_rate)
    return {"waveform": torch.randn(1, num_samples), "sample_rate": sample_rate}


def _mock_model(score: float) -> MagicMock:
    """Create a mock UTMOS model that returns a fixed score."""
    model = MagicMock()
    model.return_value = torch.tensor([score])
    model.parameters = lambda: iter([torch.tensor([0.0])])
    return model


class TestUTMOSFilterStage:
    """Tests for UTMOSFilterStage."""

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_process_passes_above_threshold(self, mock_ensure) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        stage._model = _mock_model(4.5)

        batch = AudioBatch(data=[_make_item()], task_id="test", dataset_name="test")
        result = stage.process(batch)

        assert isinstance(result, AudioBatch)
        assert len(result.data) == 1
        assert abs(result.data[0]["utmos_mos"] - 4.5) < 1e-3

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_process_filters_below_threshold(self, mock_ensure) -> None:
        stage = UTMOSFilterStage(mos_threshold=4.0)
        stage._model = _mock_model(2.5)

        batch = AudioBatch(data=[_make_item()], task_id="test", dataset_name="test")
        result = stage.process(batch)

        assert len(result.data) == 0

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_process_multiple_items(self, mock_ensure) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        model = MagicMock()
        model.side_effect = [torch.tensor([4.0]), torch.tensor([2.0]), torch.tensor([3.5])]
        model.parameters = lambda: iter([torch.tensor([0.0])])
        stage._model = model

        batch = AudioBatch(
            data=[_make_item(), _make_item(), _make_item()],
            task_id="test",
            dataset_name="test",
        )
        result = stage.process(batch)

        assert len(result.data) == 2

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_none_threshold_passes_all(self, mock_ensure) -> None:
        stage = UTMOSFilterStage(mos_threshold=None)
        stage._model = _mock_model(1.0)

        batch = AudioBatch(data=[_make_item()], task_id="test", dataset_name="test")
        result = stage.process(batch)

        assert len(result.data) == 1
        assert abs(result.data[0]["utmos_mos"] - 1.0) < 1e-3

    def test_empty_batch(self) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        stage._model = MagicMock()

        with patch.object(stage, "_ensure_model"):
            batch = AudioBatch(data=[], task_id="test", dataset_name="test")
            result = stage.process(batch)

        assert isinstance(result, AudioBatch)
        assert len(result.data) == 0

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_prediction_error_skips_item(self, mock_ensure) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        model = MagicMock(side_effect=RuntimeError("CUDA error"))
        model.parameters = lambda: iter([torch.tensor([0.0])])
        stage._model = model

        batch = AudioBatch(data=[_make_item()], task_id="test", dataset_name="test")
        result = stage.process(batch)

        assert isinstance(result, AudioBatch)
        assert len(result.data) == 0

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_no_waveform_no_filepath_skipped(self, mock_ensure) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        stage._model = _mock_model(4.0)

        batch = AudioBatch(data=[{"some_key": "value"}], task_id="test", dataset_name="test")
        result = stage.process(batch)

        assert len(result.data) == 0

    def test_model_not_loaded(self) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        stage._model = None

        with patch.object(stage, "_ensure_model"):
            batch = AudioBatch(data=[_make_item()], task_id="test", dataset_name="test")
            result = stage.process(batch)

        assert isinstance(result, AudioBatch)
        assert len(result.data) == 0

    def test_teardown_clears_model(self) -> None:
        stage = UTMOSFilterStage()
        stage._model = MagicMock()
        stage.teardown()
        assert stage._model is None
