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

from __future__ import annotations

import json
import re
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.io.materialize import BaseAudioMaterializeStage
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.tasks import AudioTask, FileGroupTask, _EmptyTask
from nemo_curator.tasks.audio_task import build_audio_sample_key
from nemo_curator.utils.file_utils import infer_dataset_name_from_path
from nemo_curator.utils.remote_io import (
    PipeStream,
    expand_sharded_paths,
    iter_tar_member_names,
    open_binary_stream,
    open_text_stream,
)

_MANIFEST_SHARD_PATTERN = re.compile(r"manifest[^/\s]*_(\d+)[^/\s]*\.(?:json|jsonl)(?:\.[^/\s]+)?")
_TAR_SHARD_PATTERN = re.compile(r"audio[^/\s]*_(\d+)[^/\s]*\.tar(?:\.[^/\s]+)?")
_OFFSET_MEMBER_PATTERN = re.compile(r"^(?P<stem>.+?)(?P<offset>-sub\d+)(?P<ext>\.[^.\\/]+)?$")

_PipeStream = PipeStream
_open_binary_stream = open_binary_stream
_open_text_stream = open_text_stream
_iter_tar_member_names = iter_tar_member_names


def _extract_shard_id(path: str, kind: Literal["manifest", "tar"]) -> int:
    pattern = _MANIFEST_SHARD_PATTERN if kind == "manifest" else _TAR_SHARD_PATTERN
    match = pattern.search(path)
    if match is None:
        msg = f"Cannot determine shard id from {kind} path/specifier: {path}"
        raise ValueError(msg)
    return int(match.group(1))


def _normalize_tar_member(audio_filepath: str) -> str:
    match = _OFFSET_MEMBER_PATTERN.match(audio_filepath)
    if match is None:
        return audio_filepath
    stem = match.group("stem")
    ext = match.group("ext") or ""
    return f"{stem}{ext}"


def _partition_paths(paths: list[str], files_per_partition: int) -> list[list[str]]:
    return [paths[i : i + files_per_partition] for i in range(0, len(paths), files_per_partition)]


def _dataset_name_from_path(path: str) -> str:
    if path.startswith("pipe:"):
        return "dataset"
    try:
        return infer_dataset_name_from_path(path)
    except Exception:  # noqa: BLE001
        return "dataset"


@dataclass
class TarredAudioManifestPartitionStage(ProcessingStage[_EmptyTask, FileGroupTask]):
    manifest_paths: str | list[str]
    files_per_partition: int = 1
    limit: int | None = None
    name: str = "tarred_audio_manifest_partitioning"

    def __post_init__(self) -> None:
        if self.files_per_partition <= 0:
            msg = "files_per_partition must be positive"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def process(self, _: _EmptyTask) -> list[FileGroupTask]:
        manifest_files = expand_sharded_paths(self.manifest_paths)
        if not manifest_files:
            return []

        partitions = _partition_paths(manifest_files, self.files_per_partition)
        dataset_name = _dataset_name_from_path(manifest_files[0])
        tasks: list[FileGroupTask] = []
        for i, file_group in enumerate(partitions):
            if self.limit is not None and len(tasks) >= self.limit:
                break
            tasks.append(
                FileGroupTask(
                    task_id=f"manifest_group_{i}",
                    dataset_name=dataset_name,
                    data=file_group,
                    _metadata={
                        "partition_index": i,
                        "total_partitions": len(partitions),
                        "source_files": file_group,
                    },
                )
            )
        return tasks


@dataclass
class TarredAudioManifestReaderStage(ProcessingStage[FileGroupTask, AudioTask]):
    tar_paths: str | list[str]
    storage_options: dict[str, Any] | None = None
    transport: Literal["auto", "fsspec", "pipe"] = "auto"
    audio_filepath_key: str = "audio_filepath"
    tar_path_key: str = "_tar_path"
    tar_member_key: str = "_tar_member"
    shard_id_key: str = "_shard_id"
    manifest_path_key: str = "_manifest_path"
    source_type_key: str = "_audio_source_type"
    skip_missing_entries: bool = False
    name: str = "tarred_audio_manifest_reader"

    def __post_init__(self) -> None:
        expanded_tar_paths = expand_sharded_paths(self.tar_paths)
        self._shard_id_to_tar_path: dict[int, str] = {}
        for tar_path in expanded_tar_paths:
            shard_id = _extract_shard_id(tar_path, "tar")
            if shard_id in self._shard_id_to_tar_path:
                msg = f"Duplicate tar shard id {shard_id} for paths: {tar_path} and {self._shard_id_to_tar_path[shard_id]}"
                raise ValueError(msg)
            self._shard_id_to_tar_path[shard_id] = tar_path

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            self.audio_filepath_key,
            self.tar_path_key,
            self.tar_member_key,
            self.shard_id_key,
            self.manifest_path_key,
            self.source_type_key,
        ]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        results: list[AudioTask] = []
        for manifest_index, manifest_path in enumerate(task.data):
            shard_id = _extract_shard_id(manifest_path, "manifest")
            if shard_id not in self._shard_id_to_tar_path:
                msg = f"No tar shard found for manifest shard {shard_id}: {manifest_path}"
                raise RuntimeError(msg)
            tar_path = self._shard_id_to_tar_path[shard_id]
            tar_members = set(
                _iter_tar_member_names(
                    tar_path,
                    storage_options=self.storage_options,
                    transport=self.transport,
                )
            )
            with _open_text_stream(
                manifest_path,
                storage_options=self.storage_options,
                transport=self.transport,
            ) as fin:
                for entry_index, line in enumerate(fin):
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    manifest_audio_path = entry[self.audio_filepath_key]
                    tar_member = _normalize_tar_member(manifest_audio_path)
                    if tar_member not in tar_members:
                        msg = (
                            f"Mismatched entry between JSON manifest ('{manifest_path}') and tar file ('{tar_path}'). "
                            f"Cannot locate tar member '{tar_member}' referenced by '{manifest_audio_path}'"
                        )
                        if self.skip_missing_entries:
                            logger.warning(msg)
                            continue
                        raise RuntimeError(msg)

                    item = dict(entry)
                    item[self.tar_path_key] = tar_path
                    item[self.tar_member_key] = tar_member
                    item[self.shard_id_key] = shard_id
                    item[self.manifest_path_key] = manifest_path
                    item[self.source_type_key] = "tarred"
                    results.append(
                        AudioTask(
                            task_id=f"{task.task_id}_{manifest_index}_{entry_index}",
                            dataset_name=task.dataset_name,
                            data=item,
                            sample_key=build_audio_sample_key(item, dataset_name=task.dataset_name),
                            _metadata=task._metadata,
                            _stage_perf=list(task._stage_perf),
                        )
                    )
        return results


@dataclass
class TarredAudioManifestReader(CompositeStage[_EmptyTask, AudioTask]):
    manifest_paths: str | list[str]
    tar_paths: str | list[str]
    files_per_partition: int = 1
    limit: int | None = None
    storage_options: dict[str, Any] | None = None
    transport: Literal["auto", "fsspec", "pipe"] = "auto"
    skip_missing_entries: bool = False
    name: str = "tarred_audio_manifest_reader"

    def __post_init__(self) -> None:
        super().__init__()
        if not self.manifest_paths:
            msg = "manifest_paths is required for TarredAudioManifestReader"
            raise ValueError(msg)
        if not self.tar_paths:
            msg = "tar_paths is required for TarredAudioManifestReader"
            raise ValueError(msg)

    def decompose(self) -> list[ProcessingStage]:
        return [
            TarredAudioManifestPartitionStage(
                manifest_paths=self.manifest_paths,
                files_per_partition=self.files_per_partition,
                limit=self.limit,
            ),
            TarredAudioManifestReaderStage(
                tar_paths=self.tar_paths,
                storage_options=self.storage_options,
                transport=self.transport,
                skip_missing_entries=self.skip_missing_entries,
            ),
        ]

    def get_description(self) -> str:
        return (
            f"Read tarred audio manifests from {self.manifest_paths} and match shards against {self.tar_paths}"
        )


@dataclass
class MaterializeTarredAudioStage(BaseAudioMaterializeStage):
    tar_path_key: str = "_tar_path"
    tar_member_key: str = "_tar_member"
    transport: Literal["auto", "fsspec", "pipe"] = "auto"
    storage_options: dict[str, Any] | None = None
    name: str = "materialize_tarred_audio"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.tar_path_key, self.tar_member_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            self.audio_filepath_key,
            self.manifest_audio_filepath_key,
            self.temporary_audio_key,
            self.materialization_mode_key,
            self.materialized_field_name_key,
        ]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []

        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task!s} failed validation for stage {self}"
                raise ValueError(msg)

        grouped_tasks: dict[str, dict[str, list[AudioTask]]] = defaultdict(lambda: defaultdict(list))
        for task in tasks:
            grouped_tasks[task.data[self.tar_path_key]][task.data[self.tar_member_key]].append(task)

        for tar_path, member_tasks in grouped_tasks.items():
            self._materialize_from_tar(tar_path, member_tasks)

        return tasks

    def _materialize_from_tar(self, tar_path: str, member_tasks: dict[str, list[AudioTask]]) -> None:
        remaining = set(member_tasks)
        with (
            _open_binary_stream(
                tar_path,
                storage_options=self.storage_options,
                transport=self.transport,
                allow_sigpipe=True,
            ) as stream,
            tarfile.open(fileobj=stream, mode="r|*") as tar,
        ):
            for tar_info in tar:
                if not tar_info.isfile() or tar_info.name not in member_tasks:
                    continue
                extracted = tar.extractfile(tar_info)
                if extracted is None:
                    continue
                raw_audio = extracted.read()
                self._materialize_tasks_from_bytes(
                    member_tasks[tar_info.name],
                    raw_audio,
                    tar_info.name,
                    reference_field=self.audio_filepath_key,
                    output_field=self.audio_filepath_key,
                )
                remaining.discard(tar_info.name)
                if not remaining:
                    break

        if remaining:
            msg = f"Failed to materialize tar members {sorted(remaining)} from tar shard '{tar_path}'"
            raise RuntimeError(msg)
