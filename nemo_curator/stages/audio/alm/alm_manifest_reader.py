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

import json
from dataclasses import dataclass, field
from typing import Any

from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import AudioTask, FileGroupTask, _EmptyTask


@dataclass
class ALMManifestReaderStage(ProcessingStage[FileGroupTask, AudioTask]):
    """Read JSONL manifest files from a FileGroupTask and emit one AudioTask per line.

    Uses line-by-line streaming via fsspec (no Pandas) to keep memory at ~1x file size.
    Supports local and cloud paths (S3, GCS).
    """

    name: str = "alm_manifest_reader_stage"

    @staticmethod
    def _derive_shard_key(manifest_path: str, corpus: str = "") -> str:
        """Derive a shard key from the manifest path starting at the corpus directory.

        Looks for ``corpus`` in the path components (from the task data)
        and returns everything from that point onward, stripping the
        file extension.  Falls back to using the ``corpus`` field from
        the first entry in the manifest if not provided.
        """
        import os
        parts = manifest_path.replace("\\", "/").rstrip("/").split("/")
        basename = parts[-1]
        for ext in (".jsonl", ".json", ".jsonl.gz"):
            if basename.endswith(ext):
                basename = basename[: -len(ext)]
                break
        parts[-1] = basename

        if corpus:
            parts_lower = [p.lower() for p in parts]
            corpus_lower = corpus.lower()
            matches = [i for i, p in enumerate(parts_lower) if p == corpus_lower]
            if matches:
                return "/".join(parts[matches[0]:])

        return "/".join(parts[-4:]) if len(parts) >= 4 else "/".join(parts)

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        paths = task.data
        results: list[AudioTask] = []
        for manifest in paths:
            entries = []
            fs, resolved = url_to_fs(manifest)
            with fs.open(resolved, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line.strip()))

            corpus = entries[0].get("corpus", "") if entries else ""
            shard_key = self._derive_shard_key(manifest, corpus)
            metadata = {**task._metadata, "_shard_key": shard_key}

            for entry in entries:
                results.append(
                    AudioTask(
                        data=entry,
                        _metadata=metadata,
                        _stage_perf=list(task._stage_perf),
                    )
                )
            logger.info(f"ALMManifestReaderStage: loaded {len(entries)} entries from {manifest} (shard_key={shard_key})")

        shard_total = len(results)
        for r in results:
            r._metadata["_shard_total"] = shard_total

        return results

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
