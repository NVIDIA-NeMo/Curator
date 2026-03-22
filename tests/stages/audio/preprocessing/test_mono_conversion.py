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

from pathlib import Path
from unittest.mock import patch

import torch

from nemo_curator.stages.audio.preprocessing.mono_conversion import MonoConversionStage
from nemo_curator.tasks import AudioBatch

MOCK_TARGET = "nemo_curator.stages.audio.preprocessing.mono_conversion.load_audio_file"
MOCK_EXISTS = "nemo_curator.stages.audio.preprocessing.mono_conversion.os.path.exists"


class TestMonoConversionStage:
    """Tests for MonoConversionStage."""

    def test_process_stereo_to_mono(self, tmp_path: Path) -> None:
        wav = tmp_path / "stereo.wav"
        wav.touch()

        stereo = torch.randn(2, 48000)

        with patch(MOCK_TARGET, return_value=(stereo, 48000)):
            with patch(MOCK_EXISTS, return_value=True):
                stage = MonoConversionStage(output_sample_rate=48000)
                batch = AudioBatch(data=[{"audio_filepath": wav.as_posix()}])
                result = stage.process(batch)

        assert isinstance(result, AudioBatch)
        assert len(result.data) == 1
        item = result.data[0]
        assert item["is_mono"] is True
        assert item["sample_rate"] == 48000
        assert item["waveform"].shape[0] == 1
        assert item["num_samples"] == 48000
        assert abs(item["duration"] - 1.0) < 1e-3

    def test_process_mono_passthrough(self, tmp_path: Path) -> None:
        wav = tmp_path / "mono.wav"
        wav.touch()

        mono = torch.randn(1, 16000)

        with patch(MOCK_TARGET, return_value=(mono, 48000)):
            with patch(MOCK_EXISTS, return_value=True):
                stage = MonoConversionStage(output_sample_rate=48000)
                batch = AudioBatch(data=[{"audio_filepath": wav.as_posix()}])
                result = stage.process(batch)

        assert len(result.data) == 1
        assert result.data[0]["waveform"].shape[0] == 1
        assert result.data[0]["num_samples"] == 16000

    def test_strict_sample_rate_rejects_mismatch(self, tmp_path: Path) -> None:
        wav = tmp_path / "wrong_sr.wav"
        wav.touch()

        audio = torch.randn(1, 22050)

        with patch(MOCK_TARGET, return_value=(audio, 22050)):
            with patch(MOCK_EXISTS, return_value=True):
                stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=True)
                batch = AudioBatch(data=[{"audio_filepath": wav.as_posix()}])
                result = stage.process(batch)

        assert len(result.data) == 0

    def test_non_strict_sample_rate_accepts_any(self, tmp_path: Path) -> None:
        wav = tmp_path / "any_sr.wav"
        wav.touch()

        audio = torch.randn(1, 22050)

        with patch(MOCK_TARGET, return_value=(audio, 22050)):
            with patch(MOCK_EXISTS, return_value=True):
                stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=False)
                batch = AudioBatch(data=[{"audio_filepath": wav.as_posix()}])
                result = stage.process(batch)

        assert len(result.data) == 1
        assert result.data[0]["sample_rate"] == 22050

    def test_missing_file_skipped(self) -> None:
        stage = MonoConversionStage()
        batch = AudioBatch(data=[{"audio_filepath": "/nonexistent/path.wav"}])
        result = stage.process(batch)
        assert len(result.data) == 0

    def test_missing_filepath_key_skipped(self) -> None:
        stage = MonoConversionStage()
        batch = AudioBatch(data=[{"other_key": "value"}])
        result = stage.process(batch)
        assert len(result.data) == 0

    def test_read_exception_skipped(self, tmp_path: Path) -> None:
        wav = tmp_path / "corrupt.wav"
        wav.touch()

        with patch(MOCK_TARGET, side_effect=RuntimeError("bad file")):
            with patch(MOCK_EXISTS, return_value=True):
                stage = MonoConversionStage()
                batch = AudioBatch(data=[{"audio_filepath": wav.as_posix()}])
                result = stage.process(batch)

        assert len(result.data) == 0
