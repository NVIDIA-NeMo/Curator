# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_curator.tasks import AudioBatch


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

    def test_scores_single_file(self, tmp_path: Path) -> None:
        wav = tmp_path / "test.wav"
        _write_sine_wav(wav)

        stage = GetUtmosv2ScoreStage(score_key="mos")
        with mock.patch.object(stage, "_score_files", return_value=[4.2]):
            result = stage.process(AudioBatch(data=[{"audio_filepath": str(wav)}]))

        assert len(result.data) == 1
        assert result.data[0]["mos"] == 4.2

    def test_scores_multiple_files(self, tmp_path: Path) -> None:
        entries = []
        for i in range(5):
            wav = tmp_path / f"test_{i}.wav"
            _write_sine_wav(wav, duration_s=0.5 + i * 0.3)
            entries.append({"audio_filepath": str(wav)})

        scores = [3.0, 3.1, 3.2, 3.3, 3.4]
        stage = GetUtmosv2ScoreStage()

        with mock.patch.object(stage, "_score_files", return_value=scores):
            result = stage.process(AudioBatch(data=entries))

        assert len(result.data) == 5
        for i, entry in enumerate(result.data):
            assert entry["utmosv2_score"] == scores[i]

    def test_missing_file_gets_nan(self, tmp_path: Path) -> None:
        good = tmp_path / "good.wav"
        _write_sine_wav(good)

        stage = GetUtmosv2ScoreStage()
        with mock.patch.object(stage, "_score_files", return_value=[3.5]):
            result = stage.process(
                AudioBatch(data=[
                    {"audio_filepath": str(good)},
                    {"audio_filepath": str(tmp_path / "missing.wav")},
                ])
            )

        assert result.data[0]["utmosv2_score"] == 3.5
        assert math.isnan(result.data[1]["utmosv2_score"])

    def test_all_files_missing(self, tmp_path: Path) -> None:
        stage = GetUtmosv2ScoreStage()
        result = stage.process(
            AudioBatch(data=[
                {"audio_filepath": str(tmp_path / "a.wav")},
                {"audio_filepath": str(tmp_path / "b.wav")},
            ])
        )

        assert len(result.data) == 2
        assert all(math.isnan(e["utmosv2_score"]) for e in result.data)

    def test_custom_keys(self, tmp_path: Path) -> None:
        wav = tmp_path / "test.wav"
        _write_sine_wav(wav)

        stage = GetUtmosv2ScoreStage(audio_filepath_key="path", score_key="quality")
        with mock.patch.object(stage, "_score_files", return_value=[4.0]):
            result = stage.process(AudioBatch(data=[{"path": str(wav)}]))

        assert result.data[0]["quality"] == 4.0

    def test_preserves_existing_fields(self, tmp_path: Path) -> None:
        wav = tmp_path / "test.wav"
        _write_sine_wav(wav)

        stage = GetUtmosv2ScoreStage()
        entry = {"audio_filepath": str(wav), "text": "hello", "speaker_id": 42}

        with mock.patch.object(stage, "_score_files", return_value=[3.0]):
            result = stage.process(AudioBatch(data=[entry]))

        assert result.data[0]["text"] == "hello"
        assert result.data[0]["speaker_id"] == 42
        assert result.data[0]["utmosv2_score"] == 3.0

    def test_audio_root_resolves_relative_paths(self, tmp_path: Path) -> None:
        subdir = tmp_path / "audio" / "sub"
        subdir.mkdir(parents=True)
        wav = subdir / "file.wav"
        _write_sine_wav(wav)

        stage = GetUtmosv2ScoreStage(audio_root=str(tmp_path / "audio"))
        with mock.patch.object(stage, "_score_files", return_value=[3.9]):
            result = stage.process(
                AudioBatch(data=[{"audio_filepath": "sub/file.wav"}])
            )

        assert result.data[0]["utmosv2_score"] == 3.9

    def test_audio_root_ignores_absolute_paths(self, tmp_path: Path) -> None:
        wav = tmp_path / "abs.wav"
        _write_sine_wav(wav)

        stage = GetUtmosv2ScoreStage(audio_root="/some/other/root")
        with mock.patch.object(stage, "_score_files", return_value=[4.1]):
            result = stage.process(
                AudioBatch(data=[{"audio_filepath": str(wav)}])
            )

        assert result.data[0]["utmosv2_score"] == 4.1

    def test_setup_calls_create_model(self) -> None:
        fake_model = mock.MagicMock()
        mock_utmosv2 = mock.MagicMock()
        mock_utmosv2.create_model.return_value = fake_model

        with mock.patch.dict("sys.modules", {"utmosv2": mock_utmosv2}):
            stage = GetUtmosv2ScoreStage()
            stage.setup()
            mock_utmosv2.create_model.assert_called_once_with(pretrained=True)
            assert stage._model is fake_model
