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

import pytest
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
            task = AudioTask(data={"audio_filepath": wav.as_posix()})
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
            task = AudioTask(data={"audio_filepath": wav.as_posix()})
            result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["waveform"].shape[0] == 1
        assert result.data["num_samples"] == 16000

    def test_strict_sample_rate_rejects_mismatch(self, tmp_path: Path) -> None:
        wav = tmp_path / "wrong_sr.wav"
        wav.touch()

        audio = torch.randn(1, 22050)

        with patch(MOCK_TARGET, return_value=(audio, 22050)), patch(MOCK_EXISTS, return_value=True):
            stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=True)
            task = AudioTask(data={"audio_filepath": wav.as_posix()})
            result = stage.process(task)

        assert result == []

    def test_non_strict_sample_rate_accepts_any(self, tmp_path: Path) -> None:
        wav = tmp_path / "any_sr.wav"
        wav.touch()

        audio = torch.randn(1, 22050)

        with patch(MOCK_TARGET, return_value=(audio, 22050)), patch(MOCK_EXISTS, return_value=True):
            stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=False)
            task = AudioTask(data={"audio_filepath": wav.as_posix()})
            result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["sample_rate"] == 22050

    def test_missing_file_skipped(self) -> None:
        stage = MonoConversionStage()
        task = AudioTask(data={"audio_filepath": "/nonexistent/path.wav"})
        result = stage.process(task)
        assert result == []

    def test_missing_filepath_key_skipped(self) -> None:
        stage = MonoConversionStage()
        task = AudioTask(data={"other_key": "value"})
        result = stage.process(task)
        assert result == []

    def test_read_exception_skipped(self, tmp_path: Path) -> None:
        wav = tmp_path / "corrupt.wav"
        wav.touch()

        with patch(MOCK_TARGET, side_effect=RuntimeError("bad file")), patch(MOCK_EXISTS, return_value=True):
            stage = MonoConversionStage()
            task = AudioTask(data={"audio_filepath": wav.as_posix()})
            result = stage.process(task)

        assert result == []


class TestMonoOutputGatingAndResidency:
    """Which destination keys appear, and which input ``auto`` residency picks.

    Lifted from tests/stages/audio/test_agent_simulation_pipelines.py: it drives only this
    stage, so it belongs beside the rest of MonoConversionStage's behaviour rather than in an
    agent-simulation file, where it was the sole coverage of these two knobs.
    """

    def test_auto_residency_uses_the_tensor_and_never_reads_disk(self, tmp_path: Path) -> None:
        """A resident waveform wins under ``auto``, even pointing at a path that cannot be read."""
        stage = MonoConversionStage(
            audio_filepath_key="agent_audio_path",
            waveform_key="agent_waveform",
            sample_rate_key="agent_sr",
            output_audio_filepath_key="agent_mono_path",
            output_sample_rate=16000,
            input_residency="auto",
            keep_waveform_in_task=True,
            write_to_disk=False,
        )
        task = AudioTask(
            dataset_name="agent",
            data={
                "agent_audio_path": str(tmp_path / "does_not_exist.wav"),
                "agent_waveform": torch.stack([torch.linspace(-0.25, 0.25, 8000)] * 2),
                "agent_sr": 16000,
            },
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["agent_waveform"].shape[0] == 1, "the stereo tensor was mixed down in memory"
        assert "agent_mono_path" not in result.data, "disk-path key must be absent when write_to_disk=False"

    @pytest.mark.parametrize("sample_rate", [True, 0, -1, 16000.5, torch.tensor([16000])])
    def test_auto_residency_uses_the_file_when_the_resident_rate_is_invalid(
        self, tmp_path: Path, sample_rate: object
    ) -> None:
        path = tmp_path / "valid.wav"
        path.touch()
        loaded = torch.ones(1, 16000)
        stage = MonoConversionStage(output_sample_rate=16000, input_residency="auto")
        task = AudioTask(
            data={
                "audio_filepath": str(path),
                "waveform": torch.zeros(1, 8000),
                "sample_rate": sample_rate,
            }
        )

        with patch(MOCK_TARGET, return_value=(loaded, 16000)) as loader:
            result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert loader.call_count == 1
        assert result.data["waveform"] is loaded
        assert result.data["sample_rate"] == 16000

    def test_disk_only_output_omits_the_waveform_key(self, tmp_path: Path) -> None:
        """``keep_waveform_in_task=False`` must drop the tensor rather than leave it stale."""
        stage = MonoConversionStage(
            audio_filepath_key="agent_audio_path",
            waveform_key="agent_waveform",
            sample_rate_key="agent_sr",
            output_audio_filepath_key="agent_mono_path",
            output_sample_rate=16000,
            input_residency="file",
            keep_waveform_in_task=False,
            write_to_disk=True,
            output_dir=str(tmp_path / "mono_out"),
        )
        task = AudioTask(dataset_name="agent", data={"agent_audio_path": str(tmp_path / "src.wav")})

        with patch(MOCK_TARGET, return_value=(torch.randn(1, 16000), 16000)), patch(MOCK_EXISTS, return_value=True):
            result = stage.process(task)

        assert "agent_mono_path" in result.data, "disk-path key must be present when write_to_disk=True"
        assert "agent_waveform" not in result.data, "tensor must be omitted when keep_waveform_in_task=False"

    @pytest.mark.parametrize("residency", ["disk", "wavefrom", ""])
    def test_unknown_residency_is_rejected_at_construction(self, residency: str) -> None:
        with pytest.raises(ValueError, match="input_residency"):
            MonoConversionStage(input_residency=residency)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"duration_key": "waveform"},
            {"is_mono_key": "sample_rate"},
            {"num_samples_key": "audio_filepath"},
            {"write_to_disk": True, "output_audio_filepath_key": "audio_filepath"},
        ],
    )
    def test_metadata_and_path_outputs_cannot_overwrite_audio_inputs(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="must not collide"):
            MonoConversionStage(**kwargs)


class TestMonoConversionPositionalCompatibility:
    def test_legacy_positional_call_keeps_strict_sample_rate_third(self) -> None:
        """Pre-agent order was (output_sample_rate, audio_filepath_key, strict_sample_rate)."""
        stage = MonoConversionStage(16000, "path_col", False)
        assert stage.output_sample_rate == 16000
        assert stage.audio_filepath_key == "path_col"
        assert stage.strict_sample_rate is False
        # The agent-added fields stayed keyword-only, so they keep their defaults here.
        assert stage.input_residency == "file"
        assert stage.waveform_key == "waveform"

    def test_agent_added_fields_are_keyword_only(self) -> None:
        """A former agent field cannot be reached positionally past the legacy slots."""
        from nemo_curator.stages.resources import Resources

        # The six legacy positional slots still accept positionals in their original order.
        stage = MonoConversionStage(16000, "path_col", True, "Custom", 2, Resources(cpus=1.0))
        assert (stage.name, stage.batch_size) == ("Custom", 2)
        # A 7th positional would be a keyword-only agent field -> TypeError.
        with pytest.raises(TypeError):
            MonoConversionStage(16000, "path_col", True, "Custom", 2, Resources(cpus=1.0), "wf")
