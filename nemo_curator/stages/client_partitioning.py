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

import json
import posixpath
from dataclasses import dataclass, field
from typing import Any

import fsspec
from fsspec.core import url_to_fs

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import FileGroupTask, _EmptyTask
from nemo_curator.utils.client_utils import FSPath


@dataclass
class ClientPartitioningStage(FilePartitioningStage):
    """Stage that partitions input file paths from a client into FileGroupTasks.

    This stage runs as a dedicated processing stage (not on the driver)
    and creates file groups based on the partitioning strategy.
    """

    input_list_json_path: str | None = None
    _name: str = "client_partitioning"
    # internal
    _fs: fsspec.AbstractFileSystem | None = field(default=None, init=False, repr=False)
    _root: str | None = field(default=None, init=False, repr=False)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        self._fs, self._root = url_to_fs(self.file_paths, **self.storage_options or {})

    def process(self, _: _EmptyTask) -> list[FileGroupTask]:
        if not self._fs or not self._root:
            msg = "Stage not initialized. Call setup() before process()."
            raise RuntimeError(msg)

        # Discover relative file list
        rel_paths = self._list_relative()

        # FILTER BY EXTENSIONS
        if self.file_extensions is not None:
            rel_paths = [p for p in rel_paths if any(p.lower().endswith(ext.lower()) for ext in self.file_extensions)]

        # FILTER BY LIMIT
        if self.limit is not None and self.limit > 0:
            rel_paths = rel_paths[: self.limit]

        # Convert relative paths to FSPath objects that embed the filesystem
        rel_paths = [FSPath(self._fs, posixpath.join(self._root, p)) for p in rel_paths]
        # Always work with List[List[str]] partitions
        if self.files_per_partition:
            partitions: list[list[str]] = self._partition_by_count(rel_paths, self.files_per_partition)
        else:
            partitions = [[p] for p in rel_paths]

        # Create FileGroupTasks for each partition
        tasks = []
        total = len(partitions)
        dataset_name = self._get_dataset_name(self.file_paths)
        for i, group in enumerate(partitions):
            tasks.append(
                FileGroupTask(
                    task_id=f"file_group_{i}",
                    dataset_name=dataset_name,
                    data=group,
                    _metadata={
                        "partition_index": i,
                        "total_partitions": total,
                        "storage_options": self.storage_options,
                        "source_files": group,  # always a list for deterministic downstream naming
                    },
                    reader_config={},
                )
            )

        return tasks

    def _list_relative(self) -> list[str]:
        """Return sorted, de-duplicated list of paths relative to root."""
        fs, root = self._fs, self._root
        if not fs or not root:
            msg = "Filesystem not initialized."
            raise RuntimeError(msg)

        if self.input_list_json_path:
            return _read_list_json_rel(root, self.input_list_json_path, self.storage_options or {})

        # Prefer pushdown via glob when extensions are known; fallback to full recursive find.
        base = root.rstrip("/")
        if self.file_extensions:
            abs_paths: list[str] = []
            for ext in self.file_extensions:
                # Recursive glob: backend may optimize server-side (e.g., s3fs)
                abs_paths.extend(fs.glob(f"{base}/**/*{ext}"))
        else:
            abs_paths = fs.find(root)

        # Relativize and normalize; de-dup while preserving order
        rels = [posixpath.relpath(p, root) for p in abs_paths]
        rels = list(dict.fromkeys(rels))  # stable de-dup
        rels.sort()
        return rels


def _read_list_json_rel(root: str, json_url: str, storage_options: dict[str, Any]) -> list[str]:
    """
    Read JSON list (via fsspec) and return entries relative to `root`.
    Validates each entry is under `root`.
    """
    with fsspec.open(json_url, "rb", **storage_options) as f:
        data = json.load(f)

    if not isinstance(data, list):
        msg = f"List JSON at {json_url} must be an array."
        raise TypeError(msg)

    listed = [str(x) for x in data]
    prefix = root.rstrip("/") + "/"

    rels: list[str] = []
    for p in listed:
        if not p.startswith(prefix):
            msg = f"Input path {p} is not under root {prefix}"
            raise ValueError(msg)
        rels.append(p[len(prefix) :])

    # stable de-dup then sort for determinism
    rels = list(dict.fromkeys(rels))
    rels.sort()
    return rels
