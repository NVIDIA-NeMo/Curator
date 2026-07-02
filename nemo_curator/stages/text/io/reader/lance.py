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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pyarrow as pa

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.tasks import DocumentBatch, EmptyTask
from nemo_curator.tasks.tasks import Task
from nemo_curator.utils.hash_utils import get_deterministic_hash
from nemo_curator.utils.lance import (
    LANCE_FRAGID_COLUMN,
    LANCE_ROWADDR_COLUMN,
    LANCE_ROWID_COLUMN,
    add_lance_metadata_columns,
)

from .base import BaseReader, ReaderOutput


@dataclass
class LanceReadTask(Task[list[int]]):
    """Task containing Lance fragment ids assigned to one read partition.

    This is created by ``LancePartitioningStage`` and consumed by
    ``LanceReaderStage``.

    Args:
        data: Lance fragment ids to read.
    """

    data: list[int] = field(default_factory=list)

    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return bool(self.data)

    def get_deterministic_id(self) -> str:
        lance_metadata = self._metadata.get("lance") or {}
        parts = [
            str(lance_metadata.get("path", self.dataset_name)),
            str(lance_metadata.get("version", "")),
            *(str(fragment_id) for fragment_id in self.data),
        ]
        return get_deterministic_hash(parts)


@dataclass
class LancePartitioningStage(ProcessingStage[EmptyTask, LanceReadTask]):
    """Stage that partitions a Lance dataset into fragment-id read tasks.

    The stage opens the dataset once, records the resolved Lance version in
    each task, and emits fragment groups for ``LanceReaderStage``.

    Args:
        path: Path or URI of the Lance dataset.
        fragments_per_partition: Number of Lance fragments assigned to each read task.
        fragment_ids: Optional explicit fragment ids to read. Defaults to all fragments. Duplicates are ignored.
        read_kwargs: Keyword arguments for opening the Lance dataset.
    """

    path: str
    fragments_per_partition: int = 32
    fragment_ids: list[int] | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = "lance_partitioning"

    def __post_init__(self) -> None:
        if self.fragments_per_partition <= 0:
            msg = "fragments_per_partition must be greater than 0"
            raise ValueError(msg)
        self.read_kwargs = dict(self.read_kwargs or {})

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def _dataset_kwargs(self) -> dict[str, Any]:
        read_kwargs = dict(self.read_kwargs)
        dataset_kwargs = dict(read_kwargs.pop("dataset_options", {}) or {})
        version = dataset_kwargs.pop("version", None)
        version = read_kwargs.pop("version", version)
        if version is not None:
            dataset_kwargs["version"] = version
        storage_options = read_kwargs.pop("storage_options", None)
        if storage_options is not None:
            dataset_kwargs["storage_options"] = storage_options
        return dataset_kwargs

    def process(self, _: EmptyTask) -> list[LanceReadTask]:
        import lance

        dataset = lance.dataset(self.path, **self._dataset_kwargs())
        available_fragments = sorted(fragment.fragment_id for fragment in dataset.get_fragments())
        if self.fragment_ids is None:
            fragment_ids = available_fragments
        else:
            fragment_ids = sorted(set(self.fragment_ids))
            missing = sorted(set(fragment_ids) - set(available_fragments))
            if missing:
                msg = f"Lance dataset does not contain requested fragment ids: {missing[:10]}"
                raise ValueError(msg)

        tasks = []
        for start in range(0, len(fragment_ids), self.fragments_per_partition):
            owned_fragments = fragment_ids[start : start + self.fragments_per_partition]
            tasks.append(
                LanceReadTask(
                    dataset_name=self.path,
                    data=owned_fragments,
                    _metadata={
                        "source_files": [self.path],
                        "lance": {
                            "path": self.path,
                            "version": dataset.version,
                            "fragment_ids": owned_fragments,
                        },
                    },
                )
            )
        return tasks


@dataclass
class LanceReaderStage(BaseReader[LanceReadTask]):
    """Stage that reads Lance fragment groups into ``DocumentBatch`` objects.

    This stage consumes ``LanceReadTask`` objects from ``LancePartitioningStage``
    and reads the pinned dataset version stored in each task.

    Args:
        path: Path or URI of the Lance dataset.
        fields: Optional columns to read. Overrides ``columns`` in ``read_kwargs``.
        read_kwargs: Keyword arguments for Lance dataset and scanner construction.
        include_lance_metadata: Whether to include row-id, row-address, and fragment-id metadata columns.
        allow_empty: Whether filtered reads may return empty tables without raising.
    """

    path: str = ""
    fields: list[str] | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    include_lance_metadata: bool = True
    allow_empty: bool = True
    name: str = "lance_reader"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.path:
            msg = "path is required"
            raise ValueError(msg)
        self.read_kwargs = dict(self.read_kwargs or {})

    def outputs(self) -> tuple[list[str], list[str]]:
        scanner_options = self.read_kwargs.get("scanner_options") or {}
        columns = self.fields if self.fields is not None else self.read_kwargs.get("columns")
        if columns is None:
            columns = scanner_options.get("columns")
        output_fields = list(columns or [])
        if self.include_lance_metadata:
            output_fields.extend([LANCE_ROWID_COLUMN, LANCE_ROWADDR_COLUMN, LANCE_FRAGID_COLUMN])
        return ["data"], output_fields

    def _output_metadata(self, task: LanceReadTask, output: ReaderOutput) -> dict[str, Any]:
        return output.metadata if output.metadata is not None else task._metadata

    def _restore_blob_v2_columns(self, dataset: object, table: pa.Table, blob_columns: list[str]) -> pa.Table:
        import lance

        rowaddrs = [int(value) for value in table["_rowaddr"].combine_chunks().to_pylist()]
        for column in blob_columns:
            payloads = [
                payload
                for _, payload in dataset.read_blobs(column, addresses=rowaddrs, preserve_order=True)  # type: ignore[attr-defined]
            ]
            table = table.set_column(table.schema.get_field_index(column), column, lance.blob_array(payloads))
        return table

    def _task_version(self, task: LanceReadTask) -> int:
        version = (task._metadata.get("lance") or {}).get("version")
        if version is None:
            msg = f"Lance read task {task.task_id} is missing a pinned Lance version"
            raise ValueError(msg)
        return version

    def _dataset_kwargs(self, read_kwargs: dict[str, Any], version: int) -> dict[str, Any]:
        dataset_kwargs = dict(read_kwargs.pop("dataset_options", {}) or {})
        requested_version = dataset_kwargs.pop("version", None)
        requested_version = read_kwargs.pop("version", requested_version)
        if requested_version is not None and requested_version != version:
            msg = f"Lance read version mismatch: task version={version}, requested version={requested_version}"
            raise ValueError(msg)
        dataset_kwargs["version"] = version
        storage_options = read_kwargs.pop("storage_options", None)
        if storage_options is not None:
            dataset_kwargs["storage_options"] = storage_options
        return dataset_kwargs

    def _scanner_kwargs(self, read_kwargs: dict[str, Any], fields: list[str] | None) -> dict[str, Any]:
        scanner_kwargs = dict(read_kwargs.pop("scanner_options", {}) or {})
        scanner_kwargs.update(read_kwargs)
        if fields is not None:
            scanner_kwargs["columns"] = fields
        return scanner_kwargs

    def read_task(
        self,
        task: LanceReadTask,
        read_kwargs: dict[str, Any] | None,
        fields: list[str] | None,
    ) -> ReaderOutput:
        import lance
        from lance.schema import schema_to_json

        read_kwargs = dict(read_kwargs or {})
        dataset_kwargs = self._dataset_kwargs(read_kwargs, self._task_version(task))
        scanner_kwargs = self._scanner_kwargs(read_kwargs, fields)
        dataset = lance.dataset(self.path, **dataset_kwargs)
        fragments = [dataset.get_fragment(fragment_id) for fragment_id in task.data]
        requested_columns = scanner_kwargs.get("columns")
        blob_columns = [
            field.name
            for field in dataset.schema
            if getattr(field.type, "extension_name", None) == "lance.blob.v2"
            and (requested_columns is None or field.name in requested_columns)
        ]
        if self.include_lance_metadata or blob_columns:
            scanner_kwargs["with_row_address"] = True
        if self.include_lance_metadata:
            scanner_kwargs["with_row_id"] = True
        scanner_kwargs["fragments"] = fragments
        table = dataset.scanner(**scanner_kwargs).to_table()
        if blob_columns:
            table = self._restore_blob_v2_columns(dataset, table, blob_columns)
        if self.include_lance_metadata:
            table = add_lance_metadata_columns(table)
        elif blob_columns and "_rowaddr" in table.column_names:
            table = table.drop_columns(["_rowaddr"])

        metadata = dict(task._metadata)
        lance_metadata = dict(metadata.get("lance") or {})
        lance_metadata["schema"] = schema_to_json(dataset.schema)
        lance_metadata["has_stable_row_ids"] = dataset.has_stable_row_ids
        metadata["lance"] = lance_metadata
        return ReaderOutput(table, metadata)


@dataclass
class LanceReader(CompositeStage[EmptyTask, DocumentBatch]):
    """Composite stage for reading Lance datasets.

    This high-level stage decomposes into:
    1. ``LancePartitioningStage`` - partitions Lance fragments into read tasks.
    2. ``LanceReaderStage`` - reads fragment groups into ``DocumentBatch`` objects.

    Args:
        path: Path or URI of the Lance dataset.
        fragments_per_partition: Number of Lance fragments assigned to each read task.
        fields: Optional columns to read.
        read_kwargs: Keyword arguments for Lance dataset and scanner construction.
        include_lance_metadata: Whether to include row-id, row-address, and fragment-id metadata columns.
        fragment_ids: Optional explicit fragment ids to read. Defaults to all fragments. Duplicates are ignored.
        task_type: Output task type. Only ``"document"`` is currently supported.
    """

    path: str
    fragments_per_partition: int = 32
    fields: list[str] | None = None
    read_kwargs: dict[str, Any] | None = None
    include_lance_metadata: bool = True
    fragment_ids: list[int] | None = None
    task_type: Literal["document"] = "document"
    name: str = "lance_reader"

    def __post_init__(self) -> None:
        super().__init__()
        self.read_kwargs = {} if self.read_kwargs is None else dict(self.read_kwargs)

    def decompose(self) -> list[ProcessingStage]:
        if self.task_type != "document":
            msg = f"Converting DocumentBatch to {self.task_type} is not supported yet."
            raise NotImplementedError(msg)

        return [
            LancePartitioningStage(
                path=self.path,
                fragments_per_partition=self.fragments_per_partition,
                fragment_ids=self.fragment_ids,
                read_kwargs=self.read_kwargs,
            ),
            LanceReaderStage(
                path=self.path,
                fields=self.fields,
                read_kwargs=self.read_kwargs,
                include_lance_metadata=self.include_lance_metadata,
            ),
        ]
