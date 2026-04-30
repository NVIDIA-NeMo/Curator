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

import hashlib
import json
import os
import posixpath
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import FileGroupTask, Task
from nemo_curator.utils.remote_io import (
    basename_from_path,
    build_remote_uri,
    copy_path,
    read_binary,
    remove_path,
)

DictTask = Task[dict[str, Any]]


def _resolve_task_storage_options(
    task: Task[Any],
    *,
    metadata_key: str,
    stage_options: dict[str, Any] | None,
) -> dict[str, Any]:
    task_options = task._metadata.get(metadata_key)
    if isinstance(task_options, dict) and task_options:
        return task_options
    return stage_options or {}


def _split_field_path(field_path: str) -> list[str]:
    parts = [part for part in field_path.split(".") if part]
    if not parts:
        msg = "field path must not be empty"
        raise ValueError(msg)
    return parts


def _task_data_as_dict(task: DictTask) -> dict[str, Any]:
    if not isinstance(task.data, dict):
        msg = f"{type(task).__name__} must have dict-backed data for file field stages"
        raise TypeError(msg)
    return task.data


def _resolve_field_path(data: dict[str, Any], field_path: str) -> Any:  # noqa: ANN401
    current: Any = data
    traversed: list[str] = []
    for part in _split_field_path(field_path):
        traversed.append(part)
        if not isinstance(current, dict):
            msg = f"Field path '{field_path}' is not addressable past '{'.'.join(traversed[:-1])}'"
            raise TypeError(msg)
        if part not in current:
            msg = f"Field path '{field_path}' is missing key '{part}'"
            raise KeyError(msg)
        current = current[part]
    return current


def _set_field_path(data: dict[str, Any], field_path: str, value: object) -> None:
    parts = _split_field_path(field_path)
    current: dict[str, Any] = data
    traversed: list[str] = []
    for part in parts[:-1]:
        traversed.append(part)
        next_value = current.get(part)
        if next_value is None:
            next_value = {}
            current[part] = next_value
        elif not isinstance(next_value, dict):
            msg = f"Field path '{field_path}' cannot be created past '{'.'.join(traversed)}'"
            raise TypeError(msg)
        current = next_value
    current[parts[-1]] = value


def _delete_field_path(data: dict[str, Any], field_path: str) -> None:
    parts = _split_field_path(field_path)
    current: dict[str, Any] = data
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            msg = f"Field path '{field_path}' is not addressable past '{part}'"
            raise TypeError(msg)
        current = next_value
    if parts[-1] not in current:
        msg = f"Field path '{field_path}' is missing key '{parts[-1]}'"
        raise KeyError(msg)
    del current[parts[-1]]


def _resolved_path_value(data: dict[str, Any], field_path: str) -> str:
    value = _resolve_field_path(data, field_path)
    if not isinstance(value, str):
        msg = f"Field path '{field_path}' must resolve to a string path or URI"
        raise TypeError(msg)
    normalized = value.strip()
    if not normalized:
        msg = f"Field path '{field_path}' resolved to an empty string"
        raise ValueError(msg)
    return normalized


def _path_suffix(path: str, fallback: str = ".bin") -> str:
    suffix = posixpath.splitext(basename_from_path(path))[1]
    return suffix if suffix else fallback


@dataclass
class MaterializeFilesStage(ProcessingStage[DictTask, DictTask]):
    source_field_path: str
    output_field_path: str
    temp_dir: str | None = None
    materialization_dir: str | None = None
    storage_options: dict[str, Any] | None = None
    transport: Literal["auto", "fsspec", "pipe"] = "auto"
    source_storage_metadata_key: str = "source_storage_options"
    name: str = "materialize_files"

    def __post_init__(self) -> None:
        _split_field_path(self.source_field_path)
        _split_field_path(self.output_field_path)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: DictTask) -> DictTask:
        msg = "MaterializeFilesStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[DictTask]) -> list[DictTask]:
        if len(tasks) == 0:
            return []

        grouped_tasks: dict[tuple[str, str], list[DictTask]] = defaultdict(list)
        source_options_by_group: dict[tuple[str, str], dict[str, Any]] = {}
        for task in tasks:
            data = _task_data_as_dict(task)
            source_path = _resolved_path_value(data, self.source_field_path)
            source_storage_options = _resolve_task_storage_options(
                task,
                metadata_key=self.source_storage_metadata_key,
                stage_options=self.storage_options,
            )
            group_key = (source_path, json.dumps(source_storage_options, sort_keys=True, default=str))
            grouped_tasks[group_key].append(task)
            source_options_by_group[group_key] = source_storage_options

        for group_key, tasks_for_source in grouped_tasks.items():
            source_path, _storage_key = group_key
            raw_bytes = read_binary(
                source_path,
                storage_options=source_options_by_group[group_key],
                transport=self.transport,
                allow_sigpipe=True,
            )
            for task in tasks_for_source:
                output_path = self._create_output_path(source_path)
                output_path.write_bytes(raw_bytes)
                _set_field_path(_task_data_as_dict(task), self.output_field_path, output_path.as_posix())

        return tasks

    def _create_output_path(self, source_path: str) -> Path:
        suffix = _path_suffix(source_path)
        if self.materialization_dir is None:
            return self._create_temp_path(suffix=suffix)

        materialization_identity = {
            "source_path": source_path,
            "output_field_path": self.output_field_path,
        }
        identity_json = json.dumps(materialization_identity, sort_keys=True, separators=(",", ":"))
        source_hash = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
        target_dir = Path(self.materialization_dir) / source_hash[:2]
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{source_hash}{suffix}"

    def _create_temp_path(self, *, suffix: str) -> Path:
        target_dir = Path(self.temp_dir) if self.temp_dir is not None else Path(tempfile.gettempdir())
        target_dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="nemo_curator_materialized_file_", suffix=suffix, dir=target_dir)
        os.close(fd)
        return Path(path)


@dataclass
class UploadFilesStage(ProcessingStage[DictTask, DictTask]):
    source_field_path: str
    output_field_path: str
    bucket: str
    protocol: str = "s3"
    key_prefix: str = ""
    key_field_path: str | None = None
    storage_options: dict[str, Any] | None = None
    source_storage_options: dict[str, Any] | None = None
    source_transport: Literal["auto", "fsspec", "pipe"] = "auto"
    source_storage_metadata_key: str = "source_storage_options"
    name: str = "upload_files"

    def __post_init__(self) -> None:
        _split_field_path(self.source_field_path)
        _split_field_path(self.output_field_path)
        if not self.bucket.strip("/"):
            msg = "bucket is required for UploadFilesStage"
            raise ValueError(msg)
        if self.key_field_path is not None:
            _split_field_path(self.key_field_path)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: DictTask) -> DictTask:
        data = _task_data_as_dict(task)
        source_path = _resolved_path_value(data, self.source_field_path)
        destination_uri = build_remote_uri(
            protocol=self.protocol,
            bucket=self.bucket,
            key=self._build_object_key(data, source_path),
        )
        copy_path(
            source_path,
            destination_uri,
            source_storage_options=_resolve_task_storage_options(
                task,
                metadata_key=self.source_storage_metadata_key,
                stage_options=self.source_storage_options,
            ),
            destination_storage_options=self.storage_options,
            source_transport=self.source_transport,
        )
        _set_field_path(data, self.output_field_path, destination_uri)
        return task

    def _build_object_key(self, data: dict[str, Any], source_path: str) -> str:
        if self.key_field_path is not None:
            key_name = _resolved_path_value(data, self.key_field_path).strip("/")
        else:
            key_name = basename_from_path(source_path)
        if self.key_prefix:
            return posixpath.join(self.key_prefix.strip("/"), key_name)
        return key_name


@dataclass
class DeleteFilesStage(ProcessingStage[DictTask, DictTask]):
    source_field_path: str
    storage_options: dict[str, Any] | None = None
    ignore_missing: bool = True
    source_storage_metadata_key: str = "source_storage_options"
    name: str = "delete_files"

    def __post_init__(self) -> None:
        _split_field_path(self.source_field_path)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: DictTask) -> DictTask:
        data = _task_data_as_dict(task)
        remove_path(
            _resolved_path_value(data, self.source_field_path),
            storage_options=_resolve_task_storage_options(
                task,
                metadata_key=self.source_storage_metadata_key,
                stage_options=self.storage_options,
            ),
            ignore_missing=self.ignore_missing,
        )
        _delete_field_path(data, self.source_field_path)
        return task


@dataclass
class UploadManifestStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    bucket: str
    protocol: str = "s3"
    key_prefix: str = ""
    storage_options: dict[str, Any] | None = None
    source_storage_options: dict[str, Any] | None = None
    source_transport: Literal["auto", "fsspec", "pipe"] = "auto"
    name: str = "upload_manifest"

    def __post_init__(self) -> None:
        if not self.bucket.strip("/"):
            msg = "bucket is required for UploadManifestStage"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: FileGroupTask) -> FileGroupTask:
        uploaded_paths: list[str] = []
        for source_path in task.data:
            key_name = basename_from_path(source_path)
            if self.key_prefix:
                key_name = posixpath.join(self.key_prefix.strip("/"), key_name)
            destination_uri = build_remote_uri(protocol=self.protocol, bucket=self.bucket, key=key_name)
            copy_path(
                source_path,
                destination_uri,
                source_storage_options=self.source_storage_options,
                destination_storage_options=self.storage_options,
                source_transport=self.source_transport,
            )
            uploaded_paths.append(destination_uri)

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=uploaded_paths,
            _metadata={
                **task._metadata,
                "uploaded_files": uploaded_paths,
                "local_source_files": list(task.data),
            },
            _stage_perf=task._stage_perf,
            reader_config=task.reader_config,
        )
