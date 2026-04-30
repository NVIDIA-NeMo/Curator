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
from typing import Any, Literal

import pandas as pd
import ray
from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import AudioTask, DocumentBatch, FileGroupTask, _EmptyTask
from nemo_curator.tasks.audio_task import build_audio_sample_key
from nemo_curator.utils.file_utils import FILETYPE_TO_DEFAULT_EXTENSIONS, pandas_select_columns

from .base import BaseReader


@dataclass
class JsonlReaderStage(BaseReader):
    """
    Stage that processes a group of JSONL files into a DocumentBatch.
    This stage accepts FileGroupTasks created by FilePartitioningStage
    and reads the actual file contents into DocumentBatches.

    Args:
        fields (list[str], optional): If specified, only read these fields (columns). Defaults to None.
        read_kwargs (dict[str, Any], optional): Keyword arguments for the reader. Defaults to {}.
        _generate_ids (bool): Whether to generate monotonically increasing IDs across all files.
            This uses IdGenerator actor, which needs to be instantiated before using this stage.
            This can be slow, so it is recommended to use AddId stage instead, unless monotonically increasing IDs
            are required.
        _assign_ids (bool): Whether to assign monotonically increasing IDs from an IdGenerator.
            This uses IdGenerator actor, which needs to be instantiated before using this stage.
            This can be slow, so it is recommended to use AddId stage instead, unless monotonically increasing IDs
            are required.
    """

    name: str = "jsonl_reader"

    def read_data(
        self,
        paths: list[str],
        read_kwargs: dict[str, Any] | None = None,
        fields: list[str] | None = None,
    ) -> pd.DataFrame | None:
        """Read JSONL files using Pandas."""

        # Normalize read_kwargs to a dict to avoid TypeError when None
        # Work on a copy to avoid mutating caller's dict
        read_kwargs = {} if read_kwargs is None else dict(read_kwargs)
        # Default to lines=True if not specified
        if "lines" in read_kwargs and read_kwargs["lines"] is False:
            msg = "lines=False is not supported for JSONL reader"
            raise ValueError(msg)
        else:
            read_kwargs["lines"] = True

        dfs = []
        for file_path in paths:
            df = pd.read_json(file_path, **read_kwargs)
            if fields is not None:
                df = pandas_select_columns(df, fields, file_path)
            dfs.append(df)
        # Concatenate all dataframes
        if not dfs:
            msg = f"No data read from files in task {paths} with read_kwargs {read_kwargs} in JSONL reader"
            logger.error(msg)
            raise ValueError(msg)
        return pd.concat(dfs, ignore_index=True)


@dataclass
class JsonlAudioReaderStage(ProcessingStage[FileGroupTask, AudioTask]):
    """Stage that streams JSONL manifests and emits one ``AudioTask`` per line.

    Unlike ``JsonlReaderStage``, this stage avoids Pandas and fans out each JSONL
    entry into an ``AudioTask``. This keeps audio manifests compatible with
    downstream audio stages and avoids materializing nested metadata as a
    ``DocumentBatch``.
    """

    fields: list[str] | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    _generate_ids: bool = False
    _assign_ids: bool = False
    name: str = "jsonl_audio_reader"

    def __post_init__(self) -> None:
        if self._generate_ids and self._assign_ids:
            msg = "Cannot generate and assign IDs at the same time"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        output_fields = list(self.fields or [])
        if self._generate_ids or self._assign_ids:
            from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

            output_fields.append(CURATOR_DEDUP_ID_STR)
        return ["sample_key"], output_fields

    def setup(self, _: Any = None) -> None:  # noqa: ANN401
        if self._generate_ids or self._assign_ids:
            from nemo_curator.stages.deduplication.id_generator import get_id_generator_actor

            try:
                self.id_generator = get_id_generator_actor()
            except ValueError:
                msg = (
                    "ID generator is required when self._generate_ids or self._assign_ids is True, "
                    "and the actor 'id_generator' does not exist. Please start the id_generator actor."
                )
                raise RuntimeError(msg) from None

    def _apply_generated_ids(self, file_paths: list[str], tasks: list[AudioTask]) -> None:
        from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

        if any(CURATOR_DEDUP_ID_STR in task.data for task in tasks):
            logger.warning(f"Column {CURATOR_DEDUP_ID_STR} already exists in {file_paths}, not generating new IDs")
            return

        min_id = ray.get(self.id_generator.register_batch.remote(file_paths, len(tasks)))
        for offset, task in enumerate(tasks):
            task.data[CURATOR_DEDUP_ID_STR] = min_id + offset

    def _apply_assigned_ids(self, file_paths: list[str], tasks: list[AudioTask]) -> None:
        from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

        if any(CURATOR_DEDUP_ID_STR in task.data for task in tasks):
            logger.warning(f"Column {CURATOR_DEDUP_ID_STR} already exists in {file_paths}, not re-assigning IDs")
            return

        min_id, max_id = ray.get(self.id_generator.get_batch_range.remote(file_paths, None))
        assigned_count = max_id - min_id + 1
        task_count = len(tasks)
        if assigned_count < task_count:
            msg = (
                f"Assigned ID range for {file_paths} contains {assigned_count} IDs, but the audio JSONL reader "
                f"produced {task_count} tasks. Ensure the batch was pre-registered with the number of non-blank "
                "JSONL entries."
            )
            raise RuntimeError(msg)
        if assigned_count > task_count:
            logger.warning(
                "Assigned ID range for {} contains {} IDs, but the audio JSONL reader produced {} tasks "
                "after skipping blank lines. Assigning the first {} IDs from the registered range.",
                file_paths,
                assigned_count,
                task_count,
                task_count,
            )

        for next_id, task in zip(range(min_id, max_id + 1), tasks, strict=False):
            task.data[CURATOR_DEDUP_ID_STR] = next_id

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        """Read JSONL files line-by-line and return one ``AudioTask`` per entry."""
        read_kwargs = {} if self.read_kwargs is None else dict(self.read_kwargs)
        if "lines" in read_kwargs and read_kwargs["lines"] is False:
            msg = "lines=False is not supported for JSONL reader"
            raise ValueError(msg)
        read_kwargs.pop("lines", None)

        storage_options = read_kwargs.pop("storage_options", None) or {}
        open_kwargs = {
            key: read_kwargs.pop(key)
            for key in ("compression", "encoding", "errors", "newline")
            if key in read_kwargs
        }
        open_kwargs.setdefault("encoding", "utf-8")

        if read_kwargs:
            logger.warning(f"Ignoring unsupported read_kwargs for audio JSONL reader: {sorted(read_kwargs.keys())}")

        results: list[AudioTask] = []
        for file_path in task.data:
            fs, resolved = url_to_fs(file_path, **storage_options)
            with fs.open(resolved, "r", **open_kwargs) as f:
                for line in f:
                    if not line.strip():
                        continue
                    raw_entry = json.loads(line)
                    sample_key = build_audio_sample_key(raw_entry, dataset_name=task.dataset_name)
                    entry = raw_entry
                    if self.fields is not None:
                        entry = {field: entry[field] for field in self.fields if field in entry}
                    results.append(
                        AudioTask(
                            task_id=f"{task.task_id}_{len(results)}",
                            dataset_name=task.dataset_name,
                            data=entry,
                            sample_key=sample_key,
                            _metadata=task._metadata,
                            _stage_perf=list(task._stage_perf),
                        )
                    )

        if results:
            if self._generate_ids:
                self._apply_generated_ids(task.data, results)
            elif self._assign_ids:
                self._apply_assigned_ids(task.data, results)

        return results

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}


@dataclass
class JsonlReader(CompositeStage[_EmptyTask, DocumentBatch | AudioTask]):
    """Composite stage for reading JSONL files.

    The output type depends on ``task_type``:
    1. ``document`` -> ``FilePartitioningStage`` + ``JsonlReaderStage`` -> ``DocumentBatch``
    2. ``audio`` -> ``FilePartitioningStage`` + ``JsonlAudioReaderStage`` -> ``AudioTask``
    """

    file_paths: str | list[str]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    fields: list[str] | None = None  # If specified, only read these columns
    read_kwargs: dict[str, Any] | None = None
    task_type: Literal["document", "image", "video", "audio"] = "document"
    file_extensions: list[str] = field(default_factory=lambda: FILETYPE_TO_DEFAULT_EXTENSIONS["jsonl"])
    _generate_ids: bool = False
    _assign_ids: bool = False
    name: str = "jsonl_reader"

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()
        if self.read_kwargs is not None:
            self.storage_options = self.read_kwargs.get("storage_options", {})

    def decompose(self) -> list[ProcessingStage]:
        """Decompose into file partitioning and processing stages."""
        if self.task_type not in {"document", "audio"}:
            msg = f"Converting DocumentBatch to {self.task_type} is not supported yet."
            raise NotImplementedError(msg)

        stages: list[ProcessingStage] = [
            FilePartitioningStage(
                file_paths=self.file_paths,
                files_per_partition=self.files_per_partition,
                blocksize=self.blocksize,
                file_extensions=self.file_extensions,
                storage_options=self.read_kwargs.get("storage_options", None)
                if self.read_kwargs is not None
                else None,
            )
        ]

        if self.task_type == "audio":
            stages.append(
                JsonlAudioReaderStage(
                    fields=self.fields,
                    read_kwargs=(self.read_kwargs or {}),
                    _generate_ids=self._generate_ids,
                    _assign_ids=self._assign_ids,
                )
            )
        else:
            stages.append(
                JsonlReaderStage(
                    fields=self.fields,
                    read_kwargs=(self.read_kwargs or {}),
                    _generate_ids=self._generate_ids,
                    _assign_ids=self._assign_ids,
                )
            )

        return stages

    def get_description(self) -> str:
        """Get a description of this composite stage."""

        parts = [f"Read JSONL files from {self.file_paths}"]

        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")

        if self.fields:
            parts.append(f"reading columns: {self.fields}")

        return ", ".join(parts)
