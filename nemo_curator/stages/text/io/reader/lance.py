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
from typing import Any, Literal

import lance
from lance.schema import schema_to_json

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.tasks import DocumentBatch, EmptyTask, LanceReadTask
from nemo_curator.utils.file_utils import infer_dataset_name_from_path
from nemo_curator.utils.lance import (
    LANCE_FRAGID_COLUMN,
    LANCE_ROWADDR_COLUMN,
    LANCE_ROWID_COLUMN,
    add_lance_metadata_columns,
    materialize_lance_blob_columns,
)

from .base import BaseReader, ReaderOutput


def _pop_dataset_kwargs(read_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove and return options intended for ``lance.dataset``.

    ``dataset_options`` contains arbitrary dataset options. Top-level
    ``version`` and ``storage_options`` are convenience aliases that take
    precedence. All remaining options stay in ``read_kwargs`` for the scanner.
    """
    dataset_kwargs = dict(read_kwargs.pop("dataset_options", {}) or {})
    for name in ("version", "storage_options"):
        value = read_kwargs.pop(name, dataset_kwargs.get(name))
        if value is None:
            dataset_kwargs.pop(name, None)
        else:
            dataset_kwargs[name] = value
    return dataset_kwargs


@dataclass
class LancePartitioningStage(ProcessingStage[EmptyTask, LanceReadTask]):
    """Stage that partitions a Lance dataset into fragment-id read tasks.

    The stage opens the dataset once, records the resolved Lance version in
    each task, and emits fragment groups for ``LanceReaderStage``.

    Args:
        path: Path or URI of the Lance dataset.
        fragments_per_partition: Number of Lance fragments assigned to each read task.
        fragment_ids: Optional explicit fragment ids to read. Defaults to all fragments. Duplicates are ignored.
        read_kwargs: Options for opening the Lance dataset. Arbitrary dataset options belong under
            ``dataset_options``; top-level ``version`` and ``storage_options`` take precedence.
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

    def process(self, _: EmptyTask) -> list[LanceReadTask]:
        dataset = lance.dataset(self.path, **_pop_dataset_kwargs(dict(self.read_kwargs)))
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
        dataset_name = infer_dataset_name_from_path(self.path, path_kind="directory")
        for start in range(0, len(fragment_ids), self.fragments_per_partition):
            fragment_ids_for_task = fragment_ids[start : start + self.fragments_per_partition]
            tasks.append(
                LanceReadTask(
                    dataset_name=dataset_name,
                    path=self.path,
                    version=dataset.version,
                    data=fragment_ids_for_task,
                )
            )
        return tasks


@dataclass
class LanceReaderStage(BaseReader):
    """Stage that reads Lance fragment groups into ``DocumentBatch`` objects.

    This stage consumes ``LanceReadTask`` objects from ``LancePartitioningStage``
    and reads the dataset path and version stored in each task.

    Args:
        fields: Optional columns to read. Overrides ``columns`` in ``read_kwargs``.
        read_kwargs: Options for Lance dataset and scanner construction. See ``LanceReader`` for the
            parsing and precedence rules.
        include_lance_metadata: Whether to include row-id, row-address, and fragment-id metadata columns.
        allow_empty: Whether filtered reads may return empty tables without raising.
    """

    fields: list[str] | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    include_lance_metadata: bool = True
    allow_empty: bool = True
    name: str = "lance_reader"

    def __post_init__(self) -> None:
        super().__post_init__()
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

    def _scanner_kwargs(self, read_kwargs: dict[str, Any], fields: list[str] | None) -> dict[str, Any]:
        """Merge nested and top-level scanner options after dataset options are removed."""
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
        read_kwargs = dict(read_kwargs or {})
        dataset_kwargs = _pop_dataset_kwargs(read_kwargs)
        dataset_kwargs["version"] = task.version
        scanner_kwargs = self._scanner_kwargs(read_kwargs, fields)
        dataset = lance.dataset(task.path, **dataset_kwargs)
        fragments = [dataset.get_fragment(fragment_id) for fragment_id in task.data]
        requested_columns = scanner_kwargs.get("columns")
        # Blob v2 scans return storage descriptors instead of payload bytes.
        # Materialize requested blobs separately and align them by row address.
        has_requested_blobs = any(
            getattr(field.type, "extension_name", None) == "lance.blob.v2"
            and (requested_columns is None or field.name in requested_columns)
            for field in dataset.schema
        )
        if self.include_lance_metadata or has_requested_blobs:
            scanner_kwargs["with_row_address"] = True
        if self.include_lance_metadata:
            scanner_kwargs["with_row_id"] = True
        scanner_kwargs["fragments"] = fragments
        table = dataset.scanner(**scanner_kwargs).to_table()
        if has_requested_blobs:
            table = materialize_lance_blob_columns(dataset, table)
        if self.include_lance_metadata:
            table = add_lance_metadata_columns(table)
        elif has_requested_blobs and "_rowaddr" in table.column_names:
            table = table.drop_columns(["_rowaddr"])

        metadata = {
            "source_files": [task.path],
            "lance": {
                "version": task.version,
                "fragment_ids": list(task.data),
                "schema": schema_to_json(dataset.schema),
                "has_stable_row_ids": dataset.has_stable_row_ids,
            },
        }
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
        read_kwargs: Options for Lance dataset and scanner construction. Arbitrary dataset options
            belong under ``dataset_options``; top-level ``version`` and ``storage_options`` take
            precedence. Options under ``scanner_options`` are merged with remaining top-level options,
            which are forwarded to ``dataset.scanner``. ``fields`` overrides scanner ``columns``.
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
                fields=self.fields,
                read_kwargs=self.read_kwargs,
                include_lance_metadata=self.include_lance_metadata,
            ),
        ]
