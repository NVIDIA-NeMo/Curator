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
from typing import Any

import pytest

from nemo_curator.stages.audio.asr.io import TarredAudioDatasetWriterStage as ExportedTarredAudioDatasetWriterStage
from nemo_curator.stages.audio.asr.io import tarred_dataset_writer as writer_module
from nemo_curator.stages.audio.asr.io.tarred_dataset_writer import TarredAudioDatasetWriterStage
from nemo_curator.tasks import _EmptyTask


def test_tarred_dataset_writer_is_exported() -> None:
    assert ExportedTarredAudioDatasetWriterStage is TarredAudioDatasetWriterStage


def test_tarred_dataset_writer_rejects_mismatched_manifest_and_target_dirs() -> None:
    with pytest.raises(ValueError, match="same length"):
        TarredAudioDatasetWriterStage(
            manifest_paths=["train.jsonl", "dev.jsonl"],
            target_dirs=["tarred/train"],
            num_shards=2,
            max_duration=20.0,
        )


def test_tarred_dataset_writer_accepts_single_manifest_and_target_dir_strings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_create_tar_datasets(**kwargs: Any) -> None:  # noqa: ANN401
        calls.append(kwargs)

    monkeypatch.setattr(writer_module, "create_tar_datasets", fake_create_tar_datasets)
    manifest_path = str(tmp_path / "train.jsonl")
    target_dir = str(tmp_path / "tarred_train")
    stage = TarredAudioDatasetWriterStage(
        manifest_paths=manifest_path,
        target_dirs=target_dir,
        num_shards=2,
        max_duration=20.0,
        dry_run=True,
    )

    output = stage.process(_EmptyTask(dataset_name="test", data=None))
    metrics = stage._consume_custom_metrics()

    assert output == []
    assert [call["manifest_path"] for call in calls] == [manifest_path]
    assert [call["target_dir"] for call in calls] == [target_dir]
    assert metrics["input_manifests"] == 1
    assert metrics["emitted_tasks"] == 0


def test_tarred_dataset_writer_runs_converter_once_per_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_create_tar_datasets(**kwargs: Any) -> None:  # noqa: ANN401
        calls.append(kwargs)

    monkeypatch.setattr(writer_module, "create_tar_datasets", fake_create_tar_datasets)
    manifest_paths = [str(tmp_path / "train.jsonl"), str(tmp_path / "dev.jsonl")]
    target_dirs = [str(tmp_path / "tarred_train"), str(tmp_path / "tarred_dev")]
    stage = TarredAudioDatasetWriterStage(
        manifest_paths=manifest_paths,
        target_dirs=target_dirs,
        num_shards=8,
        max_duration=25.0,
        min_duration=0.1,
        shuffle=True,
        keep_files_together=True,
        sort_in_shards=True,
        buckets_num=1,
        dynamic_buckets_num=16,
        shuffle_seed=123,
        no_shard_manifests=True,
        force_codec="flac",
        workers=4,
        slice_with_offset=True,
        only_manifests=True,
        dry_run=True,
    )

    output = stage.process(_EmptyTask(dataset_name="test", data=None))
    metrics = stage._consume_custom_metrics()

    assert output == []
    assert [call["manifest_path"] for call in calls] == manifest_paths
    assert [call["target_dir"] for call in calls] == target_dirs
    assert all(call["num_shards"] == 8 for call in calls)
    assert all(call["max_duration"] == 25.0 for call in calls)
    assert all(call["min_duration"] == 0.1 for call in calls)
    assert all(call["shuffle"] is True for call in calls)
    assert all(call["keep_files_together"] is True for call in calls)
    assert all(call["sort_in_shards"] is True for call in calls)
    assert all(call["buckets_num"] == 1 for call in calls)
    assert all(call["dynamic_buckets_num"] == 16 for call in calls)
    assert all(call["shuffle_seed"] == 123 for call in calls)
    assert all(call["no_shard_manifests"] is True for call in calls)
    assert all(call["force_codec"] == "flac" for call in calls)
    assert all(call["workers"] == 4 for call in calls)
    assert all(call["slice_with_offset"] is True for call in calls)
    assert all(call["only_manifests"] is True for call in calls)
    assert all(call["dry_run"] is True for call in calls)
    assert metrics["input_manifests"] == 2
    assert metrics["emitted_tasks"] == 0
