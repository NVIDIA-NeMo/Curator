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

"""Common audio stages: base classes, manifest I/O, and simple filters."""

import json
from abc import abstractmethod
from dataclasses import dataclass, field
from operator import eq, ge, gt, le, lt, ne
from typing import Any

from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import AudioBatch, FileGroupTask, Task, _EmptyTask


class LegacySpeechStage(ProcessingStage[Task, Task]):
    """
    LegacySpeechStage for SDP processors inherited from BaseParallelProcessor

    """

    def process(self, task: AudioBatch) -> list[Task]:
        result = []
        for entry in task.data:
            entries = self.process_dataset_entry(entry)
            for r in entries:
                if r is not task and not r._stage_perf:
                    r._stage_perf = list(task._stage_perf)
                if r is not task and not r._metadata:
                    r._metadata = task._metadata.copy()
            result.extend(entries)
        return result

    @abstractmethod
    def process_dataset_entry(self, data_entry: AudioBatch) -> list[AudioBatch]:
        return [data_entry]


@dataclass
class GetAudioDurationStage(LegacySpeechStage):
    """
    Stage that computes the duration of the file in ``audio_filepath_key`` (using soundfile)
    and saves the duration in ``duration_key``. If there is an error computing the duration,
    the value at ``duration_key`` will be updated with the value -1.0.

    Args:
        audio_filepath_key (str): Key to get path to wav file.
        duration_key (str): Key to put to audio duration.
    Returns:
        All the same fields as in the input manifest plus duration_key
    """

    name = "GetAudioDurationStage"
    audio_filepath_key: str
    duration_key: str

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002, ANN401
        import soundfile

        self._soundfile = soundfile

    def process_dataset_entry(self, data_entry: dict) -> list[AudioBatch]:
        audio_filepath = data_entry[self.audio_filepath_key]
        try:
            data, samplerate = self._soundfile.read(audio_filepath)
            data_entry[self.duration_key] = data.shape[0] / samplerate
        except self._soundfile.SoundFileError as e:
            logger.warning(str(e) + " file: " + audio_filepath)
            data_entry[self.duration_key] = -1.0
        return [AudioBatch(data=data_entry)]


class PreserveByValueStage(LegacySpeechStage):
    """
    Processor for preserving dataset entries based on a specified condition involving a target value and an input field.

    Args:
        input_value_key (str): The field in the dataset entries to be evaluated.
        target_value (Union[int, str]): The value to compare with the input field.
        operator (str): (Optional) The operator to apply for comparison. Options: "lt" (less than), "le" (less than or equal to), "eq" (equal to), "ne" (not equal to), "ge" (greater than or equal to), "gt" (greater than). Defaults to "eq".
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    """

    name = "PreserveByValueStage"

    def __init__(
        self,
        input_value_key: str,
        target_value: int | str,
        operator: str = "eq",
    ):
        self.input_value_key = input_value_key
        self.target_value = target_value
        if operator == "lt":
            self.operator = lt
        elif operator == "le":
            self.operator = le
        elif operator == "eq":
            self.operator = eq
        elif operator == "ne":
            self.operator = ne
        elif operator == "ge":
            self.operator = ge
        elif operator == "gt":
            self.operator = gt
        else:
            msg = 'Operator must be one from the list: "lt" (less than), "le" (less than or equal to), "eq" (equal to), "ne" (not equal to), "ge" (greater than or equal to), "gt" (greater than)'
            raise ValueError(msg)

    def process_dataset_entry(self, data_entry: AudioBatch) -> list[AudioBatch]:
        input_value = data_entry[self.input_value_key]
        target = self.target_value
        if self.operator(input_value, target):
            return [AudioBatch(data=data_entry)]
        else:
            return []


# ---------------------------------------------------------------------------
# Manifest I/O — generic stages shared by tagging, ALM, and other pipelines
# ---------------------------------------------------------------------------


@dataclass
class ManifestReaderStage(ProcessingStage[FileGroupTask, AudioBatch]):
    """Read JSONL manifest files from a FileGroupTask and emit one AudioBatch per entry.

    Uses line-by-line streaming via fsspec (no Pandas) to keep memory at ~1x file size.
    Supports local and cloud paths (S3, GCS).
    """

    name: str = "ManifestReaderStage"

    def process(self, task: FileGroupTask) -> list[AudioBatch]:
        paths = task.data
        entries: list[dict[str, Any]] = []
        for manifest in paths:
            manifest_entries: list[dict[str, Any]] = []
            fs, resolved = url_to_fs(manifest)
            with fs.open(resolved, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        manifest_entries.append(json.loads(line.strip()))
            entries.extend(manifest_entries)
            logger.info(
                f"ManifestReaderStage: loaded {len(manifest_entries)} entries from {manifest}"
            )

        return [
            AudioBatch(
                data=[entry],
                _metadata=task._metadata,
                _stage_perf=list(task._stage_perf),
            )
            for entry in entries
        ]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_fanout_stage": True}

    def num_workers(self) -> int | None:
        return 1

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}


@dataclass
class ManifestReader(CompositeStage[_EmptyTask, AudioBatch]):
    """Composite stage for reading JSONL manifests.

    Decomposes into:
    1. FilePartitioningStage — discovers and partitions manifest files
    2. ManifestReaderStage — reads each partition line-by-line (no Pandas)

    Args:
        manifest_path: Path or list of paths to JSONL manifests (local or cloud).
        files_per_partition: Number of manifest files per partition. Defaults to 1.
        blocksize: Target size per partition (e.g., "100MB"). Ignored if files_per_partition is set.
        file_extensions: File extensions to filter. Defaults to [".jsonl", ".json"].
        storage_options: Storage options for cloud paths (S3, GCS credentials, endpoints).
    """

    manifest_path: str | list[str]
    files_per_partition: int | None = 1
    blocksize: int | str | None = None
    file_extensions: list[str] = field(default_factory=lambda: [".jsonl", ".json"])
    storage_options: dict[str, Any] | None = None
    name: str = "ManifestReader"

    def __post_init__(self) -> None:
        super().__init__()

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
class ManifestWriterStage(ProcessingStage[AudioBatch, AudioBatch]):
    """Append AudioBatch entries to a JSONL manifest file.

    Each processed AudioBatch has its data entries appended to the output
    file. The file is truncated on ``setup()`` so repeated pipeline runs
    produce a clean output. Supports local and cloud paths via fsspec.

    Args:
        output_path: Destination JSONL path (local or cloud).
    """

    output_path: str
    name: str = "ManifestWriter"

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002, ANN401
        fs, path = url_to_fs(self.output_path)
        parent_dir = "/".join(path.split("/")[:-1])
        if parent_dir:
            fs.makedirs(parent_dir, exist_ok=True)
        with fs.open(path, "w", encoding="utf-8"):
            pass
        logger.info(f"ManifestWriterStage: writing to {self.output_path}")

    def process(self, task: AudioBatch) -> AudioBatch:
        fs, path = url_to_fs(self.output_path)
        with fs.open(path, "a", encoding="utf-8") as f:
            for entry in task.data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return AudioBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=task.data,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

    def num_workers(self) -> int | None:
        return 1

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}
