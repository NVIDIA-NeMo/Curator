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

from pathlib import Path
from unittest import mock

from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.tasks import AudioEntry


def test_preserve_by_value_eq_keeps_match() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=3, operator="eq")
    result = stage.process(AudioEntry(data={"v": 3}))
    assert isinstance(result, AudioEntry)
    assert result.data["v"] == 3


def test_preserve_by_value_eq_filters_non_match() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=3, operator="eq")
    result = stage.process(AudioEntry(data={"v": 1}))
    assert result == []


def test_preserve_by_value_lt() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=5, operator="lt")
    assert isinstance(stage.process(AudioEntry(data={"v": 2})), AudioEntry)
    assert stage.process(AudioEntry(data={"v": 7})) == []


def test_preserve_by_value_ge() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=10, operator="ge")
    assert stage.process(AudioEntry(data={"v": 9})) == []
    assert isinstance(stage.process(AudioEntry(data={"v": 10})), AudioEntry)
    assert isinstance(stage.process(AudioEntry(data={"v": 11})), AudioEntry)


def test_get_audio_duration_success(tmp_path: Path) -> None:
    class FakeArray:
        def __init__(self, length: int):
            self.shape = (length,)

    fake_sr = 16000
    fake_samples = FakeArray(fake_sr * 2)
    with mock.patch("soundfile.read", return_value=(fake_samples, fake_sr)):
        stage = GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration")
        stage.setup()
        entry = AudioEntry(data={"audio_filepath": (tmp_path / "fake.wav").as_posix()})
        result = stage.process(entry)
        assert isinstance(result, AudioEntry)
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
        entry = AudioEntry(data={"audio_filepath": (tmp_path / "missing.wav").as_posix()})
        result = stage.process(entry)
        assert result.data["duration"] == -1.0
