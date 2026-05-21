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

from __future__ import annotations

import math
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile

from nemo_curator.stages.audio.metrics.utmosv2_score import GetUtmosv2ScoreStage
from nemo_curator.tasks import AudioTask


def _write_sine_wav(
    path: Path,
    duration_s: float = 1.0,
    sr: int = 16000,
    freq: float = 440.0,
) -> None:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    signal = 0.5 * np.sin(2 * np.pi * freq * t)
    soundfile.write(str(path), signal, sr)


class TestGetUtmosv2ScoreStage:
    """Unit tests for GetUtmosv2ScoreStage (model mocked)."""

    def test_scores_single_audiotask(self, tmp_path: Path) -> None:
        wav = tmp_path / "test.wav"
        _write_sine_wav(wav)

        stage = GetUtmosv2ScoreStage(score_key="mos")
        task = AudioTask(
            task_id="t1",
            dataset_name="test",
            data={"audio_filepath": str(wav)},
        )

        with mock.patch.object(stage, "_score_dir", return_value=[4.2]):
            result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["mos"] == 4.2

    def test_missing_file_gets_nan(self, tmp_path: Path) -> None:
        good = tmp_path / "good.wav"
        _write_sine_wav(good)

        stage = GetUtmosv2ScoreStage()
        task = AudioTask(
            task_id="t1",
            dataset_name="test",
            data={"audio_filepath": str(tmp_path / "missing.wav")},
        )

        result = stage.process(task)

        assert math.isnan(result.data["utmosv2_score"])

    def test_in_memory_waveform(self, tmp_path: Path) -> None:
        stage = GetUtmosv2ScoreStage()
        waveform = np.random.randn(16000).astype(np.float32)
        task = AudioTask(
            task_id="t1",
            dataset_name="test",
            data={"waveform": waveform, "sample_rate": 16000},
        )

        with mock.patch.object(stage, "_score_dir", return_value=[3.8]):
            result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["utmosv2_score"] == 3.8
        assert "waveform" not in result.data

    def test_custom_keys(self, tmp_path: Path) -> None:
        wav = tmp_path / "test.wav"
        _write_sine_wav(wav)

        stage = GetUtmosv2ScoreStage(audio_filepath_key="path", score_key="quality")
        task = AudioTask(
            task_id="t1",
            dataset_name="test",
            data={"path": str(wav)},
        )

        with mock.patch.object(stage, "_score_dir", return_value=[4.0]):
            result = stage.process(task)

        assert result.data["quality"] == 4.0

    def test_preserves_existing_fields(self, tmp_path: Path) -> None:
        wav = tmp_path / "test.wav"
        _write_sine_wav(wav)

        stage = GetUtmosv2ScoreStage()
        task = AudioTask(
            task_id="t1",
            dataset_name="test",
            data={"audio_filepath": str(wav), "text": "hello", "speaker_id": 42},
        )

        with mock.patch.object(stage, "_score_dir", return_value=[3.0]):
            result = stage.process(task)

        assert result.data["text"] == "hello"
        assert result.data["speaker_id"] == 42
        assert result.data["utmosv2_score"] == 3.0

    def test_setup_calls_create_model(self) -> None:
        fake_model = mock.MagicMock()
        mock_utmosv2 = mock.MagicMock()
        mock_utmosv2.create_model.return_value = fake_model

        with mock.patch.dict("sys.modules", {"utmosv2": mock_utmosv2}):
            stage = GetUtmosv2ScoreStage()
            stage.setup()
            mock_utmosv2.create_model.assert_called_once_with(pretrained=True)
            assert stage._model is fake_model
