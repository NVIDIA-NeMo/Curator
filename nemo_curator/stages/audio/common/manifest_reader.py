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
import os
import time
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, ClassVar

from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, Role, StageContract, StaticHints
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import AudioTask, EmptyTask, FileGroupTask


def _row_names_file(row: dict[str, Any], key: str, wanted: set[str]) -> bool:
    """Whether a manifest row's audio path is one of ``wanted`` (absolute-path comparison)."""
    value = row.get(key)
    return isinstance(value, str) and os.path.abspath(os.path.expanduser(value)) in wanted


@dataclass
class ManifestReaderStage(AgentReady, ProcessingStage[FileGroupTask, AudioTask]):
    """Read JSONL manifest files from a FileGroupTask and emit one AudioTask per line.

    Uses line-by-line streaming via fsspec (no Pandas) to keep memory at ~1x file size.
    Supports local and cloud paths (S3, GCS).

    Args:
        include_files: Emit only rows whose audio path (under ``include_files_key``) is one of
            these files, comparing absolute paths. ``None`` -- the default -- reads every row.
            It restricts the same reader over the same manifest rather than pointing a delta
            run at a filtered copy, so the rows a partial run emits are the rows a full run
            would have emitted.
        include_files_key: Which row column holds that path.
    """

    name: str = "manifest_reader_stage"
    _: KW_ONLY
    include_files: list[str] | None = None
    include_files_key: str = "audio_filepath"
    # Declared statically as well as in describe(): a narrowable source has to be readable as
    # safe-to-narrow without constructing it, which is how the instance-free conformance sweep
    # sees it.
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(lifecycle_side_effects=True, per_row_independent=True)
    )
    # It points at the column holding a row's audio path, which is an existing role rather
    # than a new one -- a filter comparing something else would not be filtering by file.
    KEY_ROLE_OVERRIDES: ClassVar[Mapping[str, Role]] = {"include_files_key": "audio_filepath"}

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        t0 = time.perf_counter()
        paths = task.data
        results: list[AudioTask] = []
        count = 0
        wanted = (
            None
            if self.include_files is None
            else {os.path.abspath(os.path.expanduser(p)) for p in self.include_files}
        )
        for manifest in paths:
            fs, resolved = url_to_fs(manifest)
            with fs.open(resolved, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line.strip())
                        if wanted is not None and not _row_names_file(row, self.include_files_key, wanted):
                            continue
                        results.append(
                            AudioTask(
                                dataset_name=task.dataset_name,
                                data=row,
                                _metadata=task._metadata,
                                _stage_perf=list(task._stage_perf),
                            )
                        )
                        count += 1
            logger.info(f"ManifestReaderStage: loaded {count} entries from {manifest}")
        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "manifests_read": len(paths),
                "entries_read": len(results),
            }
        )
        return results

    def num_workers(self) -> int | None:
        return 1

    def describe(self) -> StageContract:
        return StageContract(
            # The row is emitted verbatim, so the path column this reader was pointed at IS
            # the column downstream sees. Hard-coding ``audio_filepath`` claimed a key the
            # rows do not carry whenever ``include_files_key`` is configured: a manifest of
            # ``recording_path`` validated clean against a default consumer, which then found
            # no path at runtime.
            writes=IOSpec(data_keys=[self.include_files_key]),
            cardinality="1:N fan-out",
            gates=Gates(lifecycle_side_effects=True, per_row_independent=True),
        )


@dataclass
class ManifestReader(AgentReady, CompositeStage[EmptyTask, AudioTask]):
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
        include_files: Read only the rows naming these audio files (see ``ManifestReaderStage``).
        include_files_key: Which row column holds the audio path.
    """

    manifest_path: str | list[str]
    name: str = "manifest_reader"
    files_per_partition: int | None = 1
    blocksize: int | str | None = None
    file_extensions: list[str] = field(default_factory=lambda: [".jsonl", ".json"])
    storage_options: dict[str, Any] | None = None
    _: KW_ONLY
    include_files: list[str] | None = None
    include_files_key: str = "audio_filepath"
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(gates=Gates(per_row_independent=True))
    KEY_ROLE_OVERRIDES: ClassVar[Mapping[str, Role]] = {"include_files_key": "audio_filepath"}

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
            ManifestReaderStage(
                include_files=self.include_files,
                include_files_key=self.include_files_key,
            ),
        ]

    def get_description(self) -> str:
        parts = [f"Read JSONL manifests from {self.manifest_path}"]
        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")
        return ", ".join(parts)

    def describe(self) -> StageContract:
        return StageContract(
            cardinality="1:N fan-out",
            wrappable=False,
            gates=Gates(per_row_independent=True),
        )
