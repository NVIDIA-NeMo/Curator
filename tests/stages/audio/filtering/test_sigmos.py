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

from nemo_curator.stages.audio.filtering.sigmos import SIGMOSFilterStage
from nemo_curator.tasks import AudioTask

_GOOD_SCORES = {
    "MOS_NOISE": 4.5,
    "MOS_OVRL": 4.0,
    "MOS_SIG": 4.2,
    "MOS_COL": 4.1,
    "MOS_DISC": 4.3,
    "MOS_LOUD": 3.8,
    "MOS_REVERB": 4.0,
}

_BAD_SCORES = {
    "MOS_NOISE": 2.0,
    "MOS_OVRL": 2.0,
    "MOS_SIG": 2.0,
    "MOS_COL": 2.0,
    "MOS_DISC": 2.0,
    "MOS_LOUD": 2.0,
    "MOS_REVERB": 2.0,
}


def _make_task(duration_s: float = 1.0, sample_rate: int = 48000) -> AudioTask:
    num_samples = int(duration_s * sample_rate)
    return AudioTask(
        data={"waveform": torch.randn(1, num_samples), "sample_rate": sample_rate},
        task_id="test",
        dataset_name="test",
    )


class TestSIGMOSFilterStage:

    @patch("nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage._ensure_predict")
    def test_process_passes_good_scores(self, mock_ensure) -> None:
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)

        def fake_predict(audio_data, sample_rate, model_path=None):
            return _GOOD_SCORES

        stage._predict_audio_mos = fake_predict

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert result.data["sigmos_noise"] == 4.5
        assert result.data["sigmos_ovrl"] == 4.0

    @patch("nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage._ensure_predict")
    def test_process_rejects_bad_scores(self, mock_ensure) -> None:
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)

        def fake_predict(audio_data, sample_rate, model_path=None):
            return _BAD_SCORES

        stage._predict_audio_mos = fake_predict

        result = stage.process(_make_task())

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage._ensure_predict")
    def test_none_thresholds_disable_checks(self, mock_ensure) -> None:
        stage = SIGMOSFilterStage(
            noise_threshold=None, ovrl_threshold=None,
            sig_threshold=None, col_threshold=None,
            disc_threshold=None, loud_threshold=None,
            reverb_threshold=None,
        )

        def fake_predict(audio_data, sample_rate, model_path=None):
            return _BAD_SCORES

        stage._predict_audio_mos = fake_predict

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)

    @patch("nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage._ensure_predict")
    def test_partial_threshold_fail(self, mock_ensure) -> None:
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=None)

        def fake_predict(audio_data, sample_rate, model_path=None):
            return {
                "MOS_NOISE": 3.0, "MOS_OVRL": 5.0,
                "MOS_SIG": 5.0, "MOS_COL": 5.0,
                "MOS_DISC": 5.0, "MOS_LOUD": 5.0,
                "MOS_REVERB": 5.0,
            }

        stage._predict_audio_mos = fake_predict

        result = stage.process(_make_task())

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage._ensure_predict")
    def test_sigmos_output_keys(self, mock_ensure) -> None:
        stage = SIGMOSFilterStage(noise_threshold=1.0, ovrl_threshold=1.0)

        def fake_predict(audio_data, sample_rate, model_path=None):
            return _GOOD_SCORES

        stage._predict_audio_mos = fake_predict

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        for key in ["sigmos_noise", "sigmos_ovrl", "sigmos_sig", "sigmos_col",
                     "sigmos_disc", "sigmos_loud", "sigmos_reverb"]:
            assert key in result.data

    @patch("nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage._ensure_predict")
    def test_no_audio_no_filepath_skipped(self, mock_ensure) -> None:
        stage = SIGMOSFilterStage()
        stage._predict_audio_mos = MagicMock()

        task = AudioTask(data={"some_key": "value"}, task_id="test", dataset_name="test")
        result = stage.process(task)

        assert result == []

    def test_predictor_not_available(self) -> None:
        stage = SIGMOSFilterStage()
        stage._predict_audio_mos = None

        with patch.object(stage, "_ensure_predict"):
            result = stage.process(_make_task())

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage._ensure_predict")
    def test_process_nested_segments_filters(self, mock_ensure) -> None:
        """Nested segments: only segments passing thresholds survive."""
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)
        call_count = {"n": 0}

        def fake_predict(audio_data, sample_rate, model_path=None):
            call_count["n"] += 1
            if call_count["n"] % 2 == 1:
                return _GOOD_SCORES
            return _BAD_SCORES

        stage._predict_audio_mos = fake_predict

        sr = 48000
        segments = [
            {"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i}
            for i in range(4)
        ]
        task = AudioTask(
            data={"segments": segments, "original_file": "test.wav"},
            task_id="test",
            dataset_name="test",
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert len(result.data["segments"]) == 2
        for seg in result.data["segments"]:
            assert "sigmos_noise" in seg

    @patch("nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage._ensure_predict")
    def test_process_nested_all_filtered_returns_empty(self, mock_ensure) -> None:
        """Nested segments: when all fail thresholds, return []."""
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)

        def fake_predict(audio_data, sample_rate, model_path=None):
            return _BAD_SCORES

        stage._predict_audio_mos = fake_predict

        sr = 48000
        segments = [
            {"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i}
            for i in range(3)
        ]
        task = AudioTask(
            data={"segments": segments},
            task_id="test",
            dataset_name="test",
        )

        result = stage.process(task)

        assert result == []
