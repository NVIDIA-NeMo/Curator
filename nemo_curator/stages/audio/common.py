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

import json
from dataclasses import dataclass, field
from operator import eq, ge, gt, le, lt, ne
from typing import Any

import soundfile
from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import AudioTask, FileGroupTask, _EmptyTask


def get_audio_duration(audio_filepath: str) -> float:
    """Get the duration of the audio file in seconds."""
    try:
        raw, samplerate = soundfile.read(audio_filepath)
        return raw.shape[0] / samplerate
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to get duration for audio file {audio_filepath}: {e}")
        return -1.0


@dataclass
class GetAudioDurationStage(ProcessingStage[AudioTask, AudioTask]):
    """Compute audio duration from the file at *audio_filepath_key* and
    store the result under *duration_key*.

    Args:
        audio_filepath_key: Key to get path to wav file.
        duration_key: Key to put audio duration.
    """

    name: str = "GetAudioDurationStage"
    audio_filepath_key: str = "audio_filepath"
    duration_key: str = "duration"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.duration_key]

    def process(self, task: AudioTask) -> AudioTask:
        audio_filepath = task.data[self.audio_filepath_key]
        duration = get_audio_duration(audio_filepath)
        task.data[self.duration_key] = duration
        return task


class PreserveByValueStage(ProcessingStage[AudioTask, AudioTask]):
    """Filter entries by comparing *input_value_key* against *target_value*.

    Returns ``None`` from ``process()`` to drop entries that fail the
    comparison, matching the text-modality filter convention.

    Args:
        input_value_key: The field in the dataset entries to evaluate.
        target_value: The value to compare with.
        operator: Comparison operator (lt, le, eq, ne, ge, gt).
    """

    name: str = "PreserveByValueStage"

    def __init__(
        self,
        input_value_key: str,
        target_value: int | str,
        operator: str = "eq",
    ):
        self.input_value_key = input_value_key
        self.target_value = target_value
        ops = {"lt": lt, "le": le, "eq": eq, "ne": ne, "ge": ge, "gt": gt}
        if operator not in ops:
            msg = f"Operator must be one of: {', '.join(ops)}"
            raise ValueError(msg)
        self.operator = ops[operator]

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.input_value_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.input_value_key]

    def process(self, task: AudioTask) -> AudioTask | None:
        msg = "PreserveByValueStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        results = []
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task!s} failed validation for stage {self}"
                raise ValueError(msg)
            if self.operator(task.data[self.input_value_key], self.target_value):
                results.append(task)
        return results


@dataclass
class ManifestReaderStage(ProcessingStage[FileGroupTask, AudioTask]):
    """Read JSONL manifest files from a FileGroupTask and emit one AudioTask per line.

    Uses line-by-line streaming via fsspec (no Pandas) to keep memory at ~1x file size.
    Supports local and cloud paths (S3, GCS).
    """

    name: str = "manifest_reader_stage"

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        paths = task.data
        results: list[AudioTask] = []
        for manifest in paths:
            count = 0
            fs, resolved = url_to_fs(manifest)
            with fs.open(resolved, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        results.append(
                            AudioTask(
                                task_id=task.task_id,
                                dataset_name=task.dataset_name,
                                data=json.loads(line.strip()),
                                _metadata=task._metadata,
                                _stage_perf=list(task._stage_perf),
                            )
                        )
                        count += 1
            logger.info(f"ManifestReaderStage: loaded {count} entries from {manifest}")
        return results

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_fanout_stage": True}


@dataclass
class ManifestReader(CompositeStage[_EmptyTask, AudioTask]):
    """Composite stage for reading ALM JSONL manifests.

    Decomposes into:
    1. FilePartitioningStage — discovers and partitions manifest files
    2. ALMManifestReaderStage — reads each partition line-by-line (no Pandas)

    Args:
        manifest_path: Path or list of paths to JSONL manifests (local or cloud).
        files_per_partition: Number of manifest files per partition. Defaults to 1.
        blocksize: Target size per partition (e.g., "100MB"). Ignored if files_per_partition is set.
        file_extensions: File extensions to filter. Defaults to [".jsonl", ".json"].
        storage_options: Storage options for cloud paths (S3, GCS credentials, endpoints).
    """

    name: str = "manifest_reader"
    manifest_path: str | list[str] = ""
    files_per_partition: int | None = 1
    blocksize: int | str | None = None
    file_extensions: list[str] = field(default_factory=lambda: [".jsonl", ".json"])
    storage_options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        super().__init__()
        if not self.manifest_path:
            msg = "manifest_path is required for ManifestReader"
            raise ValueError(msg)

    def decompose(self) -> list[ProcessingStage]:
        return [
            FilePartitioningStage(
                file_paths=self.manifest_path,
                files_per_partition=self.files_per_partition,
                blocksize=self.blocksize,
                file_extensions=self.file_extensions,
                storage_options=self.storage_options,
            ),
            ManifestReaderStage(),
        ]

    def get_description(self) -> str:
        parts = [f"Read JSONL manifests from {self.manifest_path}"]
        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")
        return ", ".join(parts)


@dataclass
class ManifestWriterStage(ProcessingStage[AudioTask, AudioTask]):
    """Append a single AudioTask to a JSONL manifest file.

    The output file is truncated once in ``setup()`` (called on the driver)
    so repeated pipeline runs produce a clean output.  ``setup_on_node()``
    only creates the parent directory -- it never truncates, so multi-node
    deployments do not erase each other's data.

    .. note::
       Because all nodes append to the same path, callers in multi-node
       setups should either use a shared filesystem or provide a
       node-unique ``output_path``.

    Supports local and cloud paths via fsspec.

    Args:
        output_path: Destination JSONL path (local or cloud).
    """

    name: str = "manifest_writer"
    output_path: str = ""

    def __post_init__(self) -> None:
        if not self.output_path:
            msg = "output_path is required for ManifestWriterStage"
            raise ValueError(msg)

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002, ANN401
        """Truncate the output file once on the driver before processing starts."""
        fs, path = url_to_fs(self.output_path)
        parent_dir = "/".join(path.split("/")[:-1])
        if parent_dir:
            fs.makedirs(parent_dir, exist_ok=True)
        with fs.open(path, "w", encoding="utf-8"):
            pass
        logger.info(f"ManifestWriterStage: writing to {self.output_path}")

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        """Ensure parent directory exists on each node (no truncation)."""
        fs, path = url_to_fs(self.output_path)
        parent_dir = "/".join(path.split("/")[:-1])
        if parent_dir:
            fs.makedirs(parent_dir, exist_ok=True)

    def process(self, task: AudioTask) -> AudioTask:
        fs, path = url_to_fs(self.output_path)
        with fs.open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(task.data, ensure_ascii=False) + "\n")
        return AudioTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=task.data,
            _metadata=task._metadata,
            _stage_perf=list(task._stage_perf),
        )

    def num_workers(self) -> int | None:
        return 1

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}
