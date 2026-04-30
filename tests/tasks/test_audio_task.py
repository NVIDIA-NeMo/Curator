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

from nemo_curator.tasks import AudioTask
from nemo_curator.tasks.audio_task import build_audio_sample_key


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


def test_audio_task_propagates_explicit_sample_key_from_data() -> None:
    entry = AudioTask(data={"audio_filepath": "/x.wav", "sample_key": "sample-123"})
    assert entry.sample_key == "sample-123"


def test_audio_task_persists_constructor_sample_key_back_to_data() -> None:
    entry = AudioTask(data={"audio_filepath": "/x.wav"}, sample_key="sample-456")
    assert entry.sample_key == "sample-456"
    assert entry.data["sample_key"] == "sample-456"


def test_build_audio_sample_key_is_stable_for_same_identity() -> None:
    entry = {
        "audio_filepath": "/a.wav",
        "offset": 0.25,
        "duration": 1.5,
    }

    first = build_audio_sample_key(entry, dataset_name="dataset")
    second = build_audio_sample_key(dict(entry), dataset_name="dataset")

    assert first == second
    assert first
