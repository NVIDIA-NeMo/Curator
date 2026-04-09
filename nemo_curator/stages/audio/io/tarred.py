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

import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal

import fsspec
import soundfile
from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.tasks import AudioTask, FileGroupTask, _EmptyTask
from nemo_curator.utils.file_utils import infer_dataset_name_from_path

_OP_CL_PATTERN = re.compile(r"_OP_(\d+)\.\.(\d+)_CL_")
_BRACE_RANGE_PATTERN = re.compile(r"\{(\d+)\.\.(\d+)\}")
_MANIFEST_SHARD_PATTERN = re.compile(r"manifest[^/\s]*_(\d+)[^/\s]*\.(?:json|jsonl)(?:\.[^/\s]+)?")
_TAR_SHARD_PATTERN = re.compile(r"audio[^/\s]*_(\d+)[^/\s]*\.tar(?:\.[^/\s]+)?")
_OFFSET_MEMBER_PATTERN = re.compile(r"^(?P<stem>.+?)(?P<offset>-sub\d+)(?P<ext>\.[^.\\/]+)?$")


class _PipeStream:
    def __init__(self, command: str):
        self.command = command
        self.process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> IO[bytes]:
        self.process = subprocess.Popen(  # noqa: S602
            self.command,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if self.process.stdout is None:
            msg = f"Failed to open pipe command stdout: {self.command}"
            raise RuntimeError(msg)
        return self.process.stdout

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        if self.process is None:
            return False
        if self.process.stdout is not None and not self.process.stdout.closed:
            self.process.stdout.close()
        stderr = b""
        if self.process.stderr is not None:
            stderr = self.process.stderr.read()
            self.process.stderr.close()
        return_code = self.process.wait()
        if exc_type is None and return_code != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            msg = f"Pipe command failed with exit code {return_code}: {self.command}"
            if detail:
                msg += f"\n{detail}"
            raise RuntimeError(msg)
        return False


def _expand_spec_string(spec: str) -> list[str]:
    for pattern in (_OP_CL_PATTERN, _BRACE_RANGE_PATTERN):
        match = pattern.search(spec)
        if match is None:
            continue
        start_str, end_str = match.groups()
        start = int(start_str)
        end = int(end_str)
        if end < start:
            msg = f"Invalid shard range: start={start}, end={end}, spec={spec}"
            raise ValueError(msg)
        width = max(len(start_str), len(end_str))
        prefix = spec[: match.start()]
        suffix = spec[match.end() :]
        expanded = [
            f"{prefix}{value:0{width}d}{suffix}"
            for value in range(start, end + 1)
        ]
        results: list[str] = []
        for item in expanded:
            results.extend(_expand_spec_string(item))
        return results
    return [spec]


def expand_sharded_paths(paths: str | list[str]) -> list[str]:
    if isinstance(paths, list):
        results: list[str] = []
        for path in paths:
            results.extend(_expand_spec_string(path))
        return results
    return _expand_spec_string(paths)


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


def _resolve_transport(path: str, transport: Literal["auto", "fsspec", "pipe"]) -> Literal["fsspec", "pipe"]:
    if transport == "auto":
        return "pipe" if path.startswith("pipe:") else "fsspec"
    return transport


@contextmanager
def _open_binary_stream(
    path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    transport: Literal["auto", "fsspec", "pipe"] = "auto",
) -> Iterator[IO[bytes]]:
    resolved_transport = _resolve_transport(path, transport)
    if resolved_transport == "pipe":
        command = path[len("pipe:") :].strip() if path.startswith("pipe:") else path
        with _PipeStream(command) as stream:
            yield stream
    else:
        with fsspec.open(path, mode="rb", **(storage_options or {})) as stream:
            yield stream


@contextmanager
def _open_text_stream(
    path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    transport: Literal["auto", "fsspec", "pipe"] = "auto",
    encoding: str = "utf-8",
) -> Iterator[IO[str]]:
    resolved_transport = _resolve_transport(path, transport)
    if resolved_transport == "pipe":
        with _open_binary_stream(path, storage_options=storage_options, transport=transport) as stream:
            text_stream = io.TextIOWrapper(stream, encoding=encoding)
            try:
                yield text_stream
            finally:
                text_stream.detach()
    else:
        with fsspec.open(path, mode="rt", encoding=encoding, **(storage_options or {})) as stream:
            yield stream


def _iter_tar_member_names(
    tar_path: str,
    *,
    storage_options: dict[str, Any] | None,
    transport: Literal["auto", "fsspec", "pipe"],
) -> Iterator[str]:
    with (
        _open_binary_stream(tar_path, storage_options=storage_options, transport=transport) as stream,
        tarfile.open(fileobj=stream, mode="r|*") as tar,
    ):
        for member in tar:
            if member.isfile():
                yield member.name


def _partition_paths(paths: list[str], files_per_partition: int) -> list[list[str]]:
    return [paths[i : i + files_per_partition] for i in range(0, len(paths), files_per_partition)]


def _dataset_name_from_path(path: str) -> str:
    if path.startswith("pipe:"):
        return "dataset"
    try:
        return infer_dataset_name_from_path(path)
    except Exception:  # noqa: BLE001
        return "dataset"


def _tar_member_suffix(member_name: str, fallback: str = ".bin") -> str:
    suffix = Path(member_name).suffix
    return suffix if suffix else fallback


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
class MaterializeTarredAudioStage(ProcessingStage[AudioTask, AudioTask]):
    tar_path_key: str = "_tar_path"
    tar_member_key: str = "_tar_member"
    audio_filepath_key: str = "audio_filepath"
    manifest_audio_filepath_key: str = "_manifest_audio_filepath"
    temporary_audio_key: str = "_temporary_audio_path"
    materialization_mode_key: str = "_materialization_mode"
    offset_key: str = "offset"
    duration_key: str = "duration"
    temp_dir: str | None = None
    transport: Literal["auto", "fsspec", "pipe"] = "auto"
    storage_options: dict[str, Any] | None = None
    segment_if_offset_present: bool = True
    name: str = "materialize_tarred_audio"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.tar_path_key, self.tar_member_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            self.audio_filepath_key,
            self.manifest_audio_filepath_key,
            self.temporary_audio_key,
            self.materialization_mode_key,
        ]

    def process(self, task: AudioTask) -> AudioTask:
        msg = "MaterializeTarredAudioStage only supports process_batch"
        raise NotImplementedError(msg)

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
                for task in member_tasks[tar_info.name]:
                    self._materialize_task(task, raw_audio, tar_info.name)
                remaining.discard(tar_info.name)
                if not remaining:
                    break

        if remaining:
            msg = f"Failed to materialize tar members {sorted(remaining)} from tar shard '{tar_path}'"
            raise RuntimeError(msg)

    def _materialize_task(self, task: AudioTask, raw_audio: bytes, member_name: str) -> None:
        should_segment = self._should_segment(task, member_name)
        temp_path = self._create_temp_path(
            suffix=".wav" if should_segment else _tar_member_suffix(member_name),
        )
        if should_segment:
            self._write_segmented_audio(task, raw_audio, temp_path)
            materialization_mode = "segment"
        else:
            temp_path.write_bytes(raw_audio)
            materialization_mode = "member"

        task.data.setdefault(self.manifest_audio_filepath_key, task.data.get(self.audio_filepath_key))
        task.data[self.audio_filepath_key] = temp_path.as_posix()
        task.data[self.temporary_audio_key] = temp_path.as_posix()
        task.data[self.materialization_mode_key] = materialization_mode

    def _should_segment(self, task: AudioTask, member_name: str) -> bool:
        if not self.segment_if_offset_present:
            return False
        original_path = task.data.get(self.audio_filepath_key, "")
        offset = float(task.data.get(self.offset_key, 0.0) or 0.0)
        return member_name != original_path or offset > 0.0

    def _create_temp_path(self, *, suffix: str) -> Path:
        target_dir = Path(self.temp_dir) if self.temp_dir is not None else Path(tempfile.gettempdir())
        target_dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="nemo_curator_tarred_audio_", suffix=suffix, dir=target_dir)
        os.close(fd)
        return Path(path)

    def _write_segmented_audio(self, task: AudioTask, raw_audio: bytes, output_path: Path) -> None:
        offset = float(task.data.get(self.offset_key, 0.0) or 0.0)
        duration = task.data.get(self.duration_key)
        waveform, sample_rate = soundfile.read(io.BytesIO(raw_audio), dtype="float32")
        start = max(round(offset * sample_rate), 0)
        end = waveform.shape[0]
        if duration is not None:
            end = min(start + round(float(duration) * sample_rate), waveform.shape[0])
        soundfile.write(output_path.as_posix(), waveform[start:end], sample_rate)


@dataclass
class CleanupTemporaryAudioStage(ProcessingStage[AudioTask, AudioTask]):
    temporary_audio_key: str = "_temporary_audio_path"
    audio_filepath_key: str = "audio_filepath"
    manifest_audio_filepath_key: str = "_manifest_audio_filepath"
    materialization_mode_key: str = "_materialization_mode"
    restore_manifest_audio_filepath: bool = True
    drop_temporary_metadata: bool = True
    ignore_missing: bool = True
    name: str = "cleanup_temporary_audio"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def process(self, task: AudioTask) -> AudioTask:
        temp_path = task.data.get(self.temporary_audio_key)
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                if not self.ignore_missing:
                    raise

        if self.restore_manifest_audio_filepath and self.manifest_audio_filepath_key in task.data:
            task.data[self.audio_filepath_key] = task.data[self.manifest_audio_filepath_key]

        if self.drop_temporary_metadata:
            task.data.pop(self.temporary_audio_key, None)
            task.data.pop(self.materialization_mode_key, None)

        return task
