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
from nemo_curator.tasks.audio_task import (
    build_checkpoint_shard_id,
    build_audio_sample_key,
    carry_sample_key,
    derive_child_sample_key,
    ensure_sample_key,
)


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


def test_build_checkpoint_shard_id_strips_compound_extensions() -> None:
    assert build_checkpoint_shard_id(source_files=["/tmp/manifest_0001.jsonl.gz"]) == "manifest_0001"
    assert build_checkpoint_shard_id(source_files=["/tmp/audio_0001.tar.gz"]) == "audio_0001"


def test_ensure_sample_key_derives_and_caches_key() -> None:
    task = AudioTask(dataset_name="dataset", data={"audio_filepath": "/a.wav"})

    first = ensure_sample_key(task)
    second = ensure_sample_key(task)

    assert first == second
    assert task.sample_key == first


def test_carry_sample_key_prefers_parent_key() -> None:
    task = AudioTask(data={"audio_filepath": "/a.wav"}, sample_key="parent-key")

    assert carry_sample_key(task) == "parent-key"


def test_derive_child_sample_key_is_stable_and_unique() -> None:
    task = AudioTask(dataset_name="dataset", data={"audio_filepath": "/a.wav"})

    first = derive_child_sample_key(
        task,
        child_kind="segment",
        child_identity={"segment_index": 0, "offset": 0.0, "duration": 1.0},
    )
    second = derive_child_sample_key(
        task,
        child_kind="segment",
        child_identity={"duration": 1.0, "offset": 0.0, "segment_index": 0},
    )
    third = derive_child_sample_key(
        task,
        child_kind="segment",
        child_identity={"segment_index": 1, "offset": 1.0, "duration": 1.0},
    )

    assert first == second
    assert first != third
