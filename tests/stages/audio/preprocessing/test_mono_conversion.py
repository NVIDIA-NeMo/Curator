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

from nemo_curator.stages.audio.configs.mono_conversion import MonoConversionConfig
from nemo_curator.stages.audio.preprocessing.mono_conversion import MonoConversionStage
from nemo_curator.tasks import AudioBatch


class TestMonoConversionConfig:
    """Tests for MonoConversionConfig."""

    def test_defaults(self) -> None:
        cfg = MonoConversionConfig()
        assert cfg.output_sample_rate == 48000
        assert cfg.audio_filepath_key == "audio_filepath"
        assert cfg.strict_sample_rate is True

    def test_from_dict(self) -> None:
        cfg = MonoConversionConfig.from_dict(
            {"output_sample_rate": 16000, "strict_sample_rate": False}
        )
        assert cfg.output_sample_rate == 16000
        assert cfg.strict_sample_rate is False
        assert cfg.audio_filepath_key == "audio_filepath"

    def test_from_dict_ignores_unknown_keys(self) -> None:
        cfg = MonoConversionConfig.from_dict({"unknown_key": 42})
        assert cfg.output_sample_rate == 48000

    def test_to_dict(self) -> None:
        cfg = MonoConversionConfig(output_sample_rate=22050)
        d = cfg.to_dict()
        assert d["output_sample_rate"] == 22050
        assert d["audio_filepath_key"] == "audio_filepath"
        assert d["strict_sample_rate"] is True

    def test_roundtrip(self) -> None:
        original = MonoConversionConfig(output_sample_rate=16000, strict_sample_rate=False)
        restored = MonoConversionConfig.from_dict(original.to_dict())
        assert restored.output_sample_rate == original.output_sample_rate
        assert restored.strict_sample_rate == original.strict_sample_rate


class TestMonoConversionStage:
    """Tests for MonoConversionStage."""

    def test_stage_properties(self) -> None:
        stage = MonoConversionStage()
        assert stage.name == "MonoConversion"
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == ([], ["waveform", "sample_rate", "is_mono", "duration", "num_samples"])

    def test_config_overrides_params(self) -> None:
        cfg = MonoConversionConfig(output_sample_rate=16000, strict_sample_rate=False)
        stage = MonoConversionStage(config=cfg)
        assert stage.output_sample_rate == 16000
        assert stage.strict_sample_rate is False

    def test_process_stereo_to_mono(self, tmp_path: Path) -> None:
        wav = tmp_path / "stereo.wav"
        wav.touch()

        stereo = np.random.randn(48000, 2).astype(np.float32)

        with patch("nemo_curator.stages.audio.preprocessing.mono_conversion.sf.read", return_value=(stereo, 48000)):
            with patch("os.path.exists", return_value=True):
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

        mono = np.random.randn(16000).astype(np.float32)

        with patch("nemo_curator.stages.audio.preprocessing.mono_conversion.sf.read", return_value=(mono, 48000)):
            with patch("os.path.exists", return_value=True):
                stage = MonoConversionStage(output_sample_rate=48000)
                batch = AudioBatch(data=[{"audio_filepath": wav.as_posix()}])
                result = stage.process(batch)

        assert len(result.data) == 1
        assert result.data[0]["waveform"].shape[0] == 1
        assert result.data[0]["num_samples"] == 16000

    def test_strict_sample_rate_rejects_mismatch(self, tmp_path: Path) -> None:
        wav = tmp_path / "wrong_sr.wav"
        wav.touch()

        audio = np.random.randn(22050).astype(np.float32)

        with patch("nemo_curator.stages.audio.preprocessing.mono_conversion.sf.read", return_value=(audio, 22050)):
            with patch("os.path.exists", return_value=True):
                stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=True)
                batch = AudioBatch(data=[{"audio_filepath": wav.as_posix()}])
                result = stage.process(batch)

        assert len(result.data) == 0

    def test_non_strict_sample_rate_accepts_any(self, tmp_path: Path) -> None:
        wav = tmp_path / "any_sr.wav"
        wav.touch()

        audio = np.random.randn(22050).astype(np.float32)

        with patch("nemo_curator.stages.audio.preprocessing.mono_conversion.sf.read", return_value=(audio, 22050)):
            with patch("os.path.exists", return_value=True):
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

        with patch("nemo_curator.stages.audio.preprocessing.mono_conversion.sf.read", side_effect=RuntimeError("bad file")):
            with patch("os.path.exists", return_value=True):
                stage = MonoConversionStage()
                batch = AudioBatch(data=[{"audio_filepath": wav.as_posix()}])
                result = stage.process(batch)

        assert len(result.data) == 0

    def test_preserves_task_metadata(self, tmp_path: Path) -> None:
        wav = tmp_path / "meta.wav"
        wav.touch()

        mono = np.random.randn(48000).astype(np.float32)

        with patch("nemo_curator.stages.audio.preprocessing.mono_conversion.sf.read", return_value=(mono, 48000)):
            with patch("os.path.exists", return_value=True):
                stage = MonoConversionStage()
                batch = AudioBatch(
                    data=[{"audio_filepath": wav.as_posix()}],
                    task_id="test_task",
                    dataset_name="test_ds",
                )
                result = stage.process(batch)

        assert result.task_id == "test_task"
        assert result.dataset_name == "test_ds"
