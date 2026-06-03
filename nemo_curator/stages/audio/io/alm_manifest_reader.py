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

"""ALM Manifest Reader — CompositeStage using FilePartitioningStage + line-by-line JSONL reading.

Avoids Pandas to handle large manifests with deeply nested audio metadata
(word timestamps, segments, metrics) that would cause 3-5x memory blow-up
with pd.read_json.
"""

from dataclasses import dataclass, field
from typing import Any

from nemo_curator.stages.audio.common import read_jsonl_manifests
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import AudioTask, FileGroupTask, _EmptyTask


@dataclass
class ALMManifestReaderStage(ProcessingStage[FileGroupTask, AudioTask]):
    """Read JSONL manifest files from a FileGroupTask and emit one AudioTask per line.

    Delegates to :func:`~nemo_curator.stages.audio.common.read_jsonl_manifests`
    for the core JSONL streaming logic (shared with ``ManifestReaderStage``).
    """

    name: str = "alm_manifest_reader_stage"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        return read_jsonl_manifests(
            task.data,
            parent_metadata=task._metadata,
            parent_stage_perf=task._stage_perf,
        )

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_fanout_stage": True}


@dataclass
class ALMManifestReader(CompositeStage[_EmptyTask, AudioTask]):
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

    name: str = "alm_manifest_reader"
    manifest_path: str | list[str] = ""
    files_per_partition: int | None = 1
    blocksize: int | str | None = None
    file_extensions: list[str] = field(default_factory=lambda: [".jsonl", ".json"])
    storage_options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        super().__init__()
        if not self.manifest_path:
            msg = "manifest_path is required for ALMManifestReader"
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
            ALMManifestReaderStage(),
        ]

    def get_description(self) -> str:
        parts = [f"Read ALM JSONL manifests from {self.manifest_path}"]
        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")
        return ", ".join(parts)
