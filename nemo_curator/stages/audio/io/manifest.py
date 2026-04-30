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
from dataclasses import dataclass, field
from typing import Any, Literal

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import AudioTask, FileGroupTask, _EmptyTask
from nemo_curator.tasks.audio_task import build_audio_sample_key
from nemo_curator.utils.remote_io import open_text_stream


@dataclass
class AudioManifestReaderStage(ProcessingStage[FileGroupTask, AudioTask]):
    fields: list[str] | None = None
    storage_options: dict[str, Any] | None = None
    transport: Literal["auto", "fsspec", "pipe"] = "auto"
    audio_filepath_key: str = "audio_filepath"
    manifest_path_key: str = "_manifest_path"
    source_type_key: str = "_audio_source_type"
    source_type_value: str = "manifest"
    name: str = "audio_manifest_reader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        output_fields = list(self.fields or [])
        if self.audio_filepath_key not in output_fields:
            output_fields.append(self.audio_filepath_key)
        output_fields.extend([self.manifest_path_key, self.source_type_key])
        return [], output_fields

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        results: list[AudioTask] = []
        for manifest_index, manifest_path in enumerate(task.data):
            with open_text_stream(
                manifest_path,
                storage_options=self.storage_options,
                transport=self.transport,
            ) as fin:
                for entry_index, line in enumerate(fin):
                    if not line.strip():
                        continue
                    raw_entry = json.loads(line)
                    sample_key = build_audio_sample_key(raw_entry, dataset_name=task.dataset_name)
                    entry = dict(raw_entry)
                    if self.fields is not None:
                        entry = {field: entry[field] for field in self.fields if field in entry}
                    if self.audio_filepath_key in raw_entry and self.audio_filepath_key not in entry:
                        entry[self.audio_filepath_key] = raw_entry[self.audio_filepath_key]
                    entry[self.manifest_path_key] = manifest_path
                    entry[self.source_type_key] = self.source_type_value
                    results.append(
                        AudioTask(
                            task_id=f"{task.task_id}_{manifest_index}_{entry_index}",
                            dataset_name=task.dataset_name,
                            data=entry,
                            sample_key=sample_key,
                            _metadata=task._metadata,
                            _stage_perf=list(task._stage_perf),
                        )
                    )
        return results


@dataclass
class AudioManifestReader(CompositeStage[_EmptyTask, AudioTask]):
    manifest_paths: str | list[str]
    files_per_partition: int | None = 1
    blocksize: int | str | None = None
    file_extensions: list[str] = field(default_factory=lambda: [".jsonl", ".json"])
    fields: list[str] | None = None
    storage_options: dict[str, Any] | None = None
    transport: Literal["auto", "fsspec", "pipe"] = "auto"
    limit: int | None = None
    name: str = "audio_manifest_reader"

    def __post_init__(self) -> None:
        super().__init__()
        if not self.manifest_paths:
            msg = "manifest_paths is required for AudioManifestReader"
            raise ValueError(msg)

    def decompose(self) -> list[ProcessingStage]:
        return [
            FilePartitioningStage(
                file_paths=self.manifest_paths,
                files_per_partition=self.files_per_partition,
                blocksize=self.blocksize,
                file_extensions=self.file_extensions,
                storage_options=self.storage_options,
                limit=self.limit,
            ),
            AudioManifestReaderStage(
                fields=self.fields,
                storage_options=self.storage_options,
                transport=self.transport,
            ),
        ]

    def get_description(self) -> str:
        parts = [f"Read audio manifests from {self.manifest_paths}"]
        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")
        return ", ".join(parts)
