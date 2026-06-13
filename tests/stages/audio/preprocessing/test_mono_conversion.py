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

import numpy as np
import torch

from nemo_curator.stages.audio.preprocessing.mono_conversion import MonoConversionStage
from nemo_curator.tasks import AudioTask

MOCK_TARGET = "nemo_curator.stages.audio.preprocessing.mono_conversion.load_audio_file"
MOCK_EXISTS = "nemo_curator.stages.audio.preprocessing.mono_conversion.os.path.exists"


class TestMonoConversionStage:
    def test_process_stereo_to_mono(self, tmp_path: Path) -> None:
        wav = tmp_path / "stereo.wav"
        wav.touch()

        stereo = torch.randn(2, 48000)

        with patch(MOCK_TARGET, return_value=(stereo, 48000)), patch(MOCK_EXISTS, return_value=True):
            stage = MonoConversionStage(output_sample_rate=48000)
            task = AudioTask(data={"audio_filepath": wav.as_posix()}, task_id="t1")
            result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["is_mono"] is True
        assert result.data["sample_rate"] == 48000
        assert result.data["waveform"].shape[0] == 1
        assert result.data["num_samples"] == 48000
        assert abs(result.data["duration"] - 1.0) < 1e-3

    def test_process_mono_passthrough(self, tmp_path: Path) -> None:
        wav = tmp_path / "mono.wav"
        wav.touch()

        mono = torch.randn(1, 16000)

        with patch(MOCK_TARGET, return_value=(mono, 48000)), patch(MOCK_EXISTS, return_value=True):
            stage = MonoConversionStage(output_sample_rate=48000)
            task = AudioTask(data={"audio_filepath": wav.as_posix()}, task_id="t1")
            result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["waveform"].shape[0] == 1
        assert result.data["num_samples"] == 16000

    def test_process_can_emit_shard_compatible_numpy_1d_waveform(self, tmp_path: Path) -> None:
        wav = tmp_path / "stereo.wav"
        wav.touch()

        stereo = torch.stack([torch.ones(16000), torch.zeros(16000)])

        with patch(MOCK_TARGET, return_value=(stereo, 16000)), patch(MOCK_EXISTS, return_value=True):
            stage = MonoConversionStage(
                output_sample_rate=16000,
                output_waveform_format="numpy_1d",
            )
            task = AudioTask(data={"audio_filepath": wav.as_posix()}, task_id="t1")
            result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert isinstance(result.data["waveform"], np.ndarray)
        assert result.data["waveform"].shape == (16000,)
        assert result.data["waveform"].dtype == np.float32
        np.testing.assert_allclose(result.data["waveform"], np.full(16000, 0.5, dtype=np.float32))
        assert result.data["num_samples"] == 16000
        assert result.data["duration"] == 1.0

    def test_rejects_unknown_output_waveform_format(self) -> None:
        try:
            MonoConversionStage(output_waveform_format="torch_1d")
        except ValueError as exc:
            assert "output_waveform_format" in str(exc)
        else:
            raise AssertionError("Expected invalid output_waveform_format to raise")

    def test_strict_sample_rate_rejects_mismatch(self, tmp_path: Path) -> None:
        wav = tmp_path / "wrong_sr.wav"
        wav.touch()

        audio = torch.randn(1, 22050)

        with patch(MOCK_TARGET, return_value=(audio, 22050)), patch(MOCK_EXISTS, return_value=True):
            stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=True)
            task = AudioTask(data={"audio_filepath": wav.as_posix()}, task_id="t1")
            result = stage.process(task)

        assert result == []

    def test_non_strict_sample_rate_accepts_any(self, tmp_path: Path) -> None:
        wav = tmp_path / "any_sr.wav"
        wav.touch()

        audio = torch.randn(1, 22050)

        with patch(MOCK_TARGET, return_value=(audio, 22050)), patch(MOCK_EXISTS, return_value=True):
            stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=False)
            task = AudioTask(data={"audio_filepath": wav.as_posix()}, task_id="t1")
            result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["sample_rate"] == 22050

    def test_missing_file_skipped(self) -> None:
        stage = MonoConversionStage()
        task = AudioTask(data={"audio_filepath": "/nonexistent/path.wav"}, task_id="t1")
        result = stage.process(task)
        assert result == []

    def test_missing_filepath_key_skipped(self) -> None:
        stage = MonoConversionStage()
        task = AudioTask(data={"other_key": "value"}, task_id="t1")
        result = stage.process(task)
        assert result == []

    def test_read_exception_skipped(self, tmp_path: Path) -> None:
        wav = tmp_path / "corrupt.wav"
        wav.touch()

        with patch(MOCK_TARGET, side_effect=RuntimeError("bad file")), patch(MOCK_EXISTS, return_value=True):
            stage = MonoConversionStage()
            task = AudioTask(data={"audio_filepath": wav.as_posix()}, task_id="t1")
            result = stage.process(task)

        assert result == []
