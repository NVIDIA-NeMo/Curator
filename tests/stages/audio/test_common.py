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
from nemo_curator.tasks import AudioBatch


def test_preserve_by_value_eq_operator() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=3, operator="eq")
    batch = AudioBatch(data=[{"v": 1}, {"v": 3}, {"v": 5}])
    out = stage.process(batch)
    assert len(out) == 1
    assert isinstance(out[0], AudioBatch)
    assert out[0].data[0]["v"] == 3


def test_preserve_by_value_comparators() -> None:
    # lt
    stage = PreserveByValueStage(input_value_key="v", target_value=5, operator="lt")
    out = stage.process(AudioBatch(data=[{"v": 2}, {"v": 7}]))
    assert len(out) == 1
    assert out[0].data[0]["v"] == 2

    # ge
    stage = PreserveByValueStage(input_value_key="v", target_value=10, operator="ge")
    out = stage.process(AudioBatch(data=[{"v": 9}, {"v": 10}, {"v": 11}]))
    kept = [o.data[0]["v"] for o in out]
    assert kept == [10, 11]


def test_get_audio_duration_success(tmp_path: Path) -> None:
    # Mock soundfile.read to return object with shape[0] like numpy array
    class FakeArray:
        def __init__(self, length: int):
            self.shape = (length,)

    fake_sr = 16000
    fake_samples = FakeArray(fake_sr * 2)
    with mock.patch("soundfile.read", return_value=(fake_samples, fake_sr)):
        stage = GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration")
        entry = {"audio_filepath": (tmp_path / "fake.wav").as_posix()}
        out = stage.process(AudioBatch(data=[entry]))
        assert len(out) == 1
        assert out[0].data[0]["duration"] == 2.0


def test_get_audio_duration_error_sets_minus_one(tmp_path: Path) -> None:
    class FakeError(Exception):
        pass

    with (
        mock.patch("soundfile.read", side_effect=FakeError()),
        mock.patch("nemo_curator.stages.audio.common.soundfile.SoundFileError", FakeError),
    ):
        stage = GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration")
        entry = {"audio_filepath": (tmp_path / "missing.wav").as_posix()}
        out = stage.process(AudioBatch(data=[entry]))
        assert out[0].data[0]["duration"] == -1.0
