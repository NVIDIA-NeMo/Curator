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
from unittest import mock

import numpy as np
import pytest
import torch

from nemo_curator.stages.audio.common import (
    GetAudioDurationStage,
)
from nemo_curator.tasks import AudioTask
from tests import FIXTURES_DIR


@pytest.mark.parametrize("residency", ["file", "waveform", "auto"])
def test_duration_keeps_subclass_input_validation(residency: str) -> None:
    class DurationWithTenant(GetAudioDurationStage):
        def inputs(self) -> tuple[list[str], list[str]]:
            attributes, keys = super().inputs()
            return attributes, [*keys, "tenant_id"]

    stage = DurationWithTenant(input_residency=residency)
    task = AudioTask(data={"audio_filepath": "unused.wav", "waveform": torch.ones(1, 8), "sample_rate": 8000})
    assert not stage.validate_input(task)
    with pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([task])
    task.data["tenant_id"] = "tenant"
    assert stage.validate_input(task)
    if residency == "auto":
        del task.data["audio_filepath"]
        assert stage.validate_input(task)


@pytest.mark.parametrize(
    ("path_key", "duration_key"),
    [("waveform", "duration"), ("", "duration"), ("audio_filepath", "audio_filepath"), ("audio_filepath", "")],
)
def test_duration_legacy_key_mappings_process_a_real_file(path_key: str, duration_key: str) -> None:
    path = str(FIXTURES_DIR / "audio" / "qwen_omni" / "audio_1_5s_16khz_mono.wav")
    stage = GetAudioDurationStage(audio_filepath_key=path_key, duration_key=duration_key)
    result = stage.process_batch([AudioTask(data={path_key: path})])[0]
    assert result.data[duration_key] == pytest.approx(5.0)


@pytest.mark.parametrize("duration_key", ["waveform", "sample_rate"])
def test_duration_file_mode_can_write_an_inactive_residency_key(
    duration_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nemo_curator.stages.audio.common.get_audio_duration", lambda _path: 1.0)
    stage = GetAudioDurationStage(duration_key=duration_key)
    result = stage.process(AudioTask(data={"audio_filepath": "unused.wav"}))
    assert result.data[duration_key] == 1.0


def test_get_audio_duration_validate_input_valid() -> None:
    stage = GetAudioDurationStage()
    assert stage.validate_input(AudioTask(data={"audio_filepath": "/a.wav"})) is True


def test_get_audio_duration_validate_input_missing_column() -> None:
    stage = GetAudioDurationStage()
    assert stage.validate_input(AudioTask(data={"text": "hello"})) is False


@pytest.mark.parametrize("residency", ["disk", "wavefrom", ""])
def test_get_audio_duration_rejects_unknown_residency(residency: str) -> None:
    with pytest.raises(ValueError, match="input_residency"):
        GetAudioDurationStage(input_residency=residency)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("input_residency", "duration_key"),
    [
        ("waveform", "waveform"),
        ("waveform", "sample_rate"),
        ("auto", "audio_filepath"),
        ("auto", "waveform"),
        ("auto", "sample_rate"),
    ],
)
def test_get_audio_duration_output_cannot_overwrite_an_audio_input(input_residency: str, duration_key: str) -> None:
    with pytest.raises(ValueError, match="must not collide"):
        GetAudioDurationStage(duration_key=duration_key, input_residency=input_residency)


def test_get_audio_duration_process_batch_raises_on_missing_column() -> None:
    stage = GetAudioDurationStage()
    stage.setup()
    with pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([AudioTask(data={"text": "hello"})])


def test_get_audio_duration_success(tmp_path: Path) -> None:
    class FakeInfo:
        def __init__(self, frames: int, samplerate: int):
            self.frames = frames
            self.samplerate = samplerate

    fake_info = FakeInfo(frames=16000 * 2, samplerate=16000)
    with mock.patch("soundfile.info", return_value=fake_info):
        stage = GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration")
        stage.setup()
        entry = AudioTask(data={"audio_filepath": (tmp_path / "fake.wav").as_posix()})
        result = stage.process(entry)
        assert isinstance(result, AudioTask)
        assert result.data["duration"] == 2.0


def test_get_audio_duration_error_sets_minus_one(tmp_path: Path) -> None:
    with mock.patch("soundfile.info", side_effect=RuntimeError("bad file")):
        stage = GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration")
        stage.setup()
        entry = AudioTask(data={"audio_filepath": (tmp_path / "missing.wav").as_posix()})
        result = stage.process(entry)
        assert result.data["duration"] == -1.0


def test_get_audio_duration_waveform_residency() -> None:
    """input_residency='waveform' computes duration from samples/sample_rate (no file)."""
    stage = GetAudioDurationStage(input_residency="waveform")
    stage.setup()
    result = stage.process(AudioTask(data={"waveform": torch.zeros(1, 16000 * 3), "sample_rate": 16000}))
    assert result.data["duration"] == 3.0


@pytest.mark.parametrize("sample_rate", [True, 0, -1, 16000.5, torch.tensor([16000])])
def test_get_audio_duration_rejects_invalid_resident_rates(sample_rate: object) -> None:
    stage = GetAudioDurationStage(input_residency="waveform")
    task = AudioTask(data={"waveform": torch.zeros(1, 16000), "sample_rate": sample_rate})

    assert not stage.validate_input(task)
    with pytest.raises(ValueError, match="invalid resident sample rate"):
        stage.process(task)


def test_get_audio_duration_auto_falls_back_from_invalid_resident_rate(tmp_path: Path) -> None:
    path = tmp_path / "valid.wav"
    with mock.patch("soundfile.info", return_value=mock.Mock(frames=32000, samplerate=16000)):
        stage = GetAudioDurationStage(input_residency="auto")
        stage.setup()
        task = AudioTask(
            data={
                "audio_filepath": str(path),
                "waveform": torch.zeros(1, 8000),
                "sample_rate": True,
            }
        )

        assert stage.validate_input(task)
        assert stage.process(task).data["duration"] == 2.0


@pytest.mark.parametrize("sample_rate", [16000, np.int64(16000), 16000.0, "16000", torch.tensor(16000)])
def test_get_audio_duration_accepts_lossless_resident_rates(sample_rate: object) -> None:
    stage = GetAudioDurationStage(input_residency="waveform")
    task = AudioTask(data={"waveform": torch.zeros(1, 16000), "sample_rate": sample_rate})

    assert stage.validate_input(task)
    assert stage.process(task).data["duration"] == 1.0


def test_get_audio_duration_auto_prefers_waveform() -> None:
    stage = GetAudioDurationStage(input_residency="auto")
    stage.setup()
    result = stage.process(AudioTask(data={"waveform": torch.zeros(1, 16000), "sample_rate": 16000}))
    assert result.data["duration"] == 1.0


def test_get_audio_duration_default_rejects_waveform_only() -> None:
    """Regression: default residency is 'file'; a waveform-only task is not valid input."""
    import torch

    stage = GetAudioDurationStage()
    assert stage.input_residency == "file"
    assert stage.validate_input(AudioTask(data={"waveform": torch.zeros(1, 16000), "sample_rate": 16000})) is False
    assert stage.validate_input(AudioTask(data={"audio_filepath": "/a.wav"})) is True


def test_get_audio_duration_waveform_validate() -> None:
    import torch

    stage = GetAudioDurationStage(input_residency="waveform")
    assert stage.validate_input(AudioTask(data={"waveform": torch.zeros(1, 16000), "sample_rate": 16000})) is True
    assert stage.validate_input(AudioTask(data={"audio_filepath": "/a.wav"})) is False


def test_duration_legacy_path_alias_can_be_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nemo_curator.stages.audio.common.get_audio_duration", lambda _path: 1.0)
    stage = GetAudioDurationStage(audio_filepath_key="waveform")
    result = stage.process(AudioTask(data={"waveform": "unused.wav"}))
    assert result.data["duration"] == 1.0
