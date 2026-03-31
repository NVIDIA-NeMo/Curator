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
from nemo_curator.tasks import AudioTask


def _make_task(duration_s: float = 1.0, sample_rate: int = 16000) -> AudioTask:
    num_samples = int(duration_s * sample_rate)
    return AudioTask(
        data={"waveform": torch.randn(1, num_samples), "sample_rate": sample_rate},
        task_id="test",
        dataset_name="test",
    )


def _mock_model(score: float) -> MagicMock:
    model = MagicMock()
    model.return_value = torch.tensor([score])
    model.parameters = lambda: iter([torch.tensor([0.0])])
    return model


class TestUTMOSFilterStage:
    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_process_passes_above_threshold(self, mock_ensure: MagicMock) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        stage._model = _mock_model(4.5)

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert abs(result.data["utmos_mos"] - 4.5) < 1e-3

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_process_filters_below_threshold(self, mock_ensure: MagicMock) -> None:
        stage = UTMOSFilterStage(mos_threshold=4.0)
        stage._model = _mock_model(2.5)

        result = stage.process(_make_task())

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_none_threshold_passes_all(self, mock_ensure: MagicMock) -> None:
        stage = UTMOSFilterStage(mos_threshold=None)
        stage._model = _mock_model(1.0)

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert abs(result.data["utmos_mos"] - 1.0) < 1e-3

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_prediction_error_skips(self, mock_ensure: MagicMock) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        model = MagicMock(side_effect=RuntimeError("CUDA error"))
        model.parameters = lambda: iter([torch.tensor([0.0])])
        stage._model = model

        result = stage.process(_make_task())

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_no_waveform_no_filepath_skipped(self, mock_ensure: MagicMock) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        stage._model = _mock_model(4.0)

        task = AudioTask(data={"some_key": "value"}, task_id="test", dataset_name="test")
        result = stage.process(task)

        assert result == []

    def test_model_not_loaded(self) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        stage._model = None

        with patch.object(stage, "_ensure_model"):
            result = stage.process(_make_task())

        assert result == []

    def test_teardown_clears_model(self) -> None:
        stage = UTMOSFilterStage()
        stage._model = MagicMock()
        stage.teardown()
        assert stage._model is None

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_process_nested_segments_filters(self, mock_ensure: MagicMock) -> None:
        """Nested segments: only segments above threshold survive."""
        stage = UTMOSFilterStage(mos_threshold=3.0)
        call_count = {"n": 0}

        def model_side_effect(_waveform: torch.Tensor, sr: int = 16000) -> torch.Tensor:  # noqa: ARG001
            call_count["n"] += 1
            return torch.tensor([4.0 if call_count["n"] % 2 == 1 else 2.0])

        model = MagicMock(side_effect=model_side_effect)
        model.parameters = lambda: iter([torch.tensor([0.0])])
        stage._model = model

        sr = 16000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(4)]
        task = AudioTask(
            data={"segments": segments, "original_file": "test.wav"},
            task_id="test",
            dataset_name="test",
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert len(result.data["segments"]) == 2
        for seg in result.data["segments"]:
            assert "utmos_mos" in seg

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_process_nested_all_filtered_returns_empty(self, mock_ensure: MagicMock) -> None:
        """Nested segments: when all fail threshold, return []."""
        stage = UTMOSFilterStage(mos_threshold=4.0)
        stage._model = _mock_model(2.0)

        sr = 16000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(3)]
        task = AudioTask(
            data={"segments": segments},
            task_id="test",
            dataset_name="test",
        )

        result = stage.process(task)

        assert result == []
