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

import pytest

from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.tasks import AudioTask


def test_preserve_by_value_validate_input_valid() -> None:
    stage = PreserveByValueStage(input_value_key="wer", target_value=50, operator="le")
    assert stage.validate_input(AudioTask(data={"wer": 30})) is True


def test_preserve_by_value_validate_input_missing_column() -> None:
    stage = PreserveByValueStage(input_value_key="wer", target_value=50, operator="le")
    assert stage.validate_input(AudioTask(data={"text": "hello"})) is False


def test_get_audio_duration_validate_input_valid() -> None:
    stage = GetAudioDurationStage()
    assert stage.validate_input(AudioTask(data={"audio_filepath": "/a.wav"})) is True


def test_get_audio_duration_validate_input_missing_column() -> None:
    stage = GetAudioDurationStage()
    assert stage.validate_input(AudioTask(data={"text": "hello"})) is False


def test_get_audio_duration_process_batch_raises_on_missing_column() -> None:
    stage = GetAudioDurationStage()
    stage.setup()
    with pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([AudioTask(data={"text": "hello"})])


def test_preserve_by_value_process_raises_not_implemented() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=3, operator="eq")
    with pytest.raises(NotImplementedError, match="only supports process_batch"):
        stage.process(AudioTask(data={"v": 3}))


def test_preserve_by_value_process_batch_raises_on_missing_column() -> None:
    stage = PreserveByValueStage(input_value_key="wer", target_value=50, operator="le")
    with pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([AudioTask(data={"text": "hello"})])


def test_preserve_by_value_eq_keeps_match() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=3, operator="eq")
    result = stage.process_batch([AudioTask(data={"v": 3})])
    assert len(result) == 1
    assert isinstance(result[0], AudioTask)
    assert result[0].data["v"] == 3


def test_preserve_by_value_eq_filters_non_match() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=3, operator="eq")
    result = stage.process_batch([AudioTask(data={"v": 1})])
    assert len(result) == 0


def test_preserve_by_value_lt() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=5, operator="lt")
    assert len(stage.process_batch([AudioTask(data={"v": 2})])) == 1
    assert len(stage.process_batch([AudioTask(data={"v": 7})])) == 0


def test_preserve_by_value_ge() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=10, operator="ge")
    assert len(stage.process_batch([AudioTask(data={"v": 9})])) == 0
    assert len(stage.process_batch([AudioTask(data={"v": 10})])) == 1
    assert len(stage.process_batch([AudioTask(data={"v": 11})])) == 1


def test_get_audio_duration_success(tmp_path: Path) -> None:
    class FakeInfo:
        frames = 32000
        samplerate = 16000

    with mock.patch("soundfile.info", return_value=FakeInfo()):
        stage = GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration")
        stage.setup()
        entry = AudioTask(data={"audio_filepath": (tmp_path / "fake.wav").as_posix()})
        result = stage.process(entry)
        assert isinstance(result, AudioTask)
        assert result.data["duration"] == 2.0


def test_get_audio_duration_error_sets_minus_one(tmp_path: Path) -> None:
    class FakeError(Exception):
        pass

    with (
        mock.patch("soundfile.read", side_effect=FakeError()),
        mock.patch("soundfile.SoundFileError", FakeError),
    ):
        stage = GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration")
        stage.setup()
        entry = AudioTask(data={"audio_filepath": (tmp_path / "missing.wav").as_posix()})
        result = stage.process(entry)
        assert result.data["duration"] == -1.0
