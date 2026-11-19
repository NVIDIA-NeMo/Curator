# modality: audio

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

import fsspec
import pytest

from nemo_curator.tasks import AudioTask


def test_audio_task_stores_dict() -> None:
    entry = AudioTask(data={"audio_filepath": "/x.wav"})
    assert isinstance(entry.data, dict)
    assert entry.data["audio_filepath"] == "/x.wav"
    assert entry.num_items == 1


def test_audio_task_default_empty_dict() -> None:
    entry = AudioTask()
    assert entry.data == {}
    assert entry.num_items == 1


def test_audio_task_validation_existing_file(tmp_path: Path) -> None:
    existing = tmp_path / "ok.wav"
    existing.write_bytes(b"fake")

    entry = AudioTask(data={"audio_filepath": existing.as_posix()}, filepath_key="audio_filepath")
    assert entry.validate() is True


def test_audio_task_validation_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.wav"
    entry = AudioTask(data={"audio_filepath": missing.as_posix()}, filepath_key="audio_filepath")
    assert entry.validate() is False


def test_audio_task_validation_no_filepath_key() -> None:
    entry = AudioTask(data={"text": "hello"})
    assert entry.validate() is True


def test_audio_task_validation_uses_fsspec() -> None:
    fs = fsspec.filesystem("memory")
    fs.pipe("memory://audio/ok.wav", b"fake")
    assert AudioTask(data={"audio_filepath": "memory://audio/ok.wav"}, filepath_key="audio_filepath").validate()
    assert not AudioTask(data={"audio_filepath": "memory://audio/no.wav"}, filepath_key="audio_filepath").validate()


def test_audio_task_validates_on_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(AudioTask, "validate", lambda self: calls.append(self) or True)
    task = AudioTask(data={"audio_filepath": "/missing.wav"}, filepath_key="audio_filepath")
    assert calls == [task]


def test_audio_task_validation_rejects_non_path_value() -> None:
    assert AudioTask(data={"audio_filepath": None}, filepath_key="audio_filepath").validate() is False
