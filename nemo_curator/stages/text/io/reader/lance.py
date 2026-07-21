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
import pyarrow as pa
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


def _validate_positive_optional(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        msg = f"{name} must be greater than 0 when set"
        raise ValueError(msg)


def _rows_per_split(table: pa.Table, *, max_batch_rows: int | None, max_batch_bytes: int | None) -> int:
    if table.num_rows == 0:
        return 0
    rows_per_split = table.num_rows
    if max_batch_rows is not None:
        rows_per_split = min(rows_per_split, max_batch_rows)
    if max_batch_bytes is not None and table.nbytes > max_batch_bytes:
        bytes_per_row = max(1, (table.nbytes + table.num_rows - 1) // table.num_rows)
        rows_per_split = min(rows_per_split, max(1, max_batch_bytes // bytes_per_row))
    return rows_per_split


def _split_table_by_limits(
    table: pa.Table,
    *,
    max_batch_rows: int | None,
    max_batch_bytes: int | None,
) -> list[pa.Table]:
    rows_per_split = _rows_per_split(table, max_batch_rows=max_batch_rows, max_batch_bytes=max_batch_bytes)
    if rows_per_split == 0:
        return [table]
    if rows_per_split >= table.num_rows:
        return [table]
    return [
        table.slice(start, min(rows_per_split, table.num_rows - start))
        for start in range(0, table.num_rows, rows_per_split)
    ]


def _would_exceed_limits(
    *,
    rows: int,
    bytes_: int,
    max_batch_rows: int | None,
    max_batch_bytes: int | None,
) -> bool:
    return bool(
        (max_batch_rows is not None and rows > max_batch_rows)
        or (max_batch_bytes is not None and bytes_ > max_batch_bytes)
    )


def _tables_from_record_batches(
    batches: object,
    *,
    max_batch_rows: int | None,
    max_batch_bytes: int | None,
) -> list[pa.Table]:
    chunks: list[pa.Table] = []
    pending: list[pa.Table] = []
    pending_rows = 0
    pending_bytes = 0

    def flush_pending() -> None:
        nonlocal pending_rows, pending_bytes
        if not pending:
            return
        chunks.append(pa.concat_tables(pending, promote_options="default") if len(pending) > 1 else pending[0])
        pending.clear()
        pending_rows = 0
        pending_bytes = 0

    for batch in batches:
        table = pa.Table.from_batches([batch]) if isinstance(batch, pa.RecordBatch) else batch
        if not isinstance(table, pa.Table) or table.num_rows == 0:
            continue
        for split in _split_table_by_limits(
            table,
            max_batch_rows=max_batch_rows,
            max_batch_bytes=max_batch_bytes,
        ):
            if max_batch_rows is None and max_batch_bytes is None:
                chunks.append(split)
                continue
            next_rows = pending_rows + split.num_rows
            next_bytes = pending_bytes + split.nbytes
            if pending and _would_exceed_limits(
                rows=next_rows,
                bytes_=next_bytes,
                max_batch_rows=max_batch_rows,
                max_batch_bytes=max_batch_bytes,
            ):
                flush_pending()
            pending.append(split)
            pending_rows += split.num_rows
            pending_bytes += split.nbytes

    flush_pending()
    return chunks


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
        fragments_per_partition: Number of Lance fragments assigned to each read task. This is a coarse
            partitioning knob: large multimodal fragments can still produce large read tasks, so tune this together
            with ``LanceReaderStage.max_batch_rows`` and ``LanceReaderStage.max_batch_bytes``.
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
        max_batch_rows: Optional maximum rows per emitted ``DocumentBatch``.
        max_batch_bytes: Optional approximate maximum Arrow bytes per emitted ``DocumentBatch``.
        streaming_read: Whether to read scanner record batches directly. This is enabled automatically when
            ``max_batch_rows`` or ``max_batch_bytes`` is set.
    """

    fields: list[str] | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    include_lance_metadata: bool = True
    allow_empty: bool = True
    max_batch_rows: int | None = None
    max_batch_bytes: int | None = None
    streaming_read: bool = False
    name: str = "lance_reader"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.read_kwargs = dict(self.read_kwargs or {})
        _validate_positive_optional("max_batch_rows", self.max_batch_rows)
        _validate_positive_optional("max_batch_bytes", self.max_batch_bytes)

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

    def _prepare_lance_scan(
        self,
        task: LanceReadTask,
        read_kwargs: dict[str, Any] | None,
        fields: list[str] | None,
    ) -> tuple[lance.LanceDataset, dict[str, Any], list[str]]:
        read_kwargs = dict(read_kwargs or {})
        dataset_kwargs = _pop_dataset_kwargs(read_kwargs)
        dataset_kwargs["version"] = task.version
        scanner_kwargs = self._scanner_kwargs(read_kwargs, fields)
        dataset = lance.dataset(task.path, **dataset_kwargs)
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
        return dataset, scanner_kwargs, blob_columns

    def _metadata_for_task(self, task: LanceReadTask, dataset: lance.LanceDataset) -> dict[str, Any]:
        return {
            "source_files": [task.path],
            "lance": {
                "version": task.version,
                "fragment_ids": list(task.data),
                "schema": schema_to_json(dataset.schema),
                "has_stable_row_ids": dataset.has_stable_row_ids,
            },
        }

    def _finalize_table(self, dataset: lance.LanceDataset, table: pa.Table, blob_columns: list[str]) -> pa.Table:
        if blob_columns:
            table = materialize_lance_blob_columns(dataset, table)
        if self.include_lance_metadata:
            return add_lance_metadata_columns(table)
        if blob_columns and "_rowaddr" in table.column_names:
            return table.drop_columns(["_rowaddr"])
        return table

    def _read_outputs(
        self,
        task: LanceReadTask,
        read_kwargs: dict[str, Any] | None,
        fields: list[str] | None,
    ) -> list[ReaderOutput]:
        dataset, scanner_kwargs, blob_columns = self._prepare_lance_scan(task, read_kwargs, fields)
        metadata = self._metadata_for_task(task, dataset)
        use_streaming = self.streaming_read or self.max_batch_rows is not None or self.max_batch_bytes is not None
        if not use_streaming:
            table = dataset.scanner(**scanner_kwargs).to_table()
            table = self._finalize_table(dataset, table, blob_columns)
            return [ReaderOutput(table, metadata)]

        scanner = dataset.scanner(**scanner_kwargs)
        tables = _tables_from_record_batches(
            scanner.to_batches(),
            max_batch_rows=self.max_batch_rows,
            max_batch_bytes=self.max_batch_bytes,
        )
        if not tables:
            tables = [dataset.scanner(**scanner_kwargs).to_table()]
        return [ReaderOutput(self._finalize_table(dataset, table, blob_columns), metadata) for table in tables]

    def process(self, task: LanceReadTask) -> DocumentBatch | list[DocumentBatch]:
        outputs = self._read_outputs(task, dict(self.read_kwargs or {}), self.fields)
        batches: list[DocumentBatch] = []
        for output in outputs:
            self._validate_result(task, output.data)
            batches.append(self._document_batch(task, output))
        if not batches:
            return []
        return batches if len(batches) > 1 else batches[0]

    def read_task(
        self,
        task: LanceReadTask,
        read_kwargs: dict[str, Any] | None,
        fields: list[str] | None,
    ) -> ReaderOutput:
        return self._read_outputs(task, read_kwargs, fields)[0]


@dataclass
class LanceReader(CompositeStage[EmptyTask, DocumentBatch]):
    """Composite stage for reading Lance datasets.

    This high-level stage decomposes into:
    1. ``LancePartitioningStage`` - partitions Lance fragments into read tasks.
    2. ``LanceReaderStage`` - reads fragment groups into ``DocumentBatch`` objects.

    Args:
        path: Path or URI of the Lance dataset.
        fragments_per_partition: Number of Lance fragments assigned to each read task. This is a coarse
            partitioning knob: large fragments can still produce large reader outputs, so tune it together with
            ``max_batch_rows`` and ``max_batch_bytes``.
        fields: Optional columns to read.
        read_kwargs: Options for Lance dataset and scanner construction. Arbitrary dataset options
            belong under ``dataset_options``; top-level ``version`` and ``storage_options`` take
            precedence. Options under ``scanner_options`` are merged with remaining top-level options,
            which are forwarded to ``dataset.scanner``. ``fields`` overrides scanner ``columns``.
        include_lance_metadata: Whether to include row-id, row-address, and fragment-id metadata columns.
        fragment_ids: Optional explicit fragment ids to read. Defaults to all fragments. Duplicates are ignored.
        max_batch_rows: Optional maximum rows per emitted ``DocumentBatch``.
        max_batch_bytes: Optional approximate maximum Arrow bytes per emitted ``DocumentBatch``.
        streaming_read: Whether to read scanner record batches directly. This is enabled automatically when
            ``max_batch_rows`` or ``max_batch_bytes`` is set.
        task_type: Output task type. Only ``"document"`` is currently supported.
    """

    path: str
    fragments_per_partition: int = 32
    fields: list[str] | None = None
    read_kwargs: dict[str, Any] | None = None
    include_lance_metadata: bool = True
    fragment_ids: list[int] | None = None
    max_batch_rows: int | None = None
    max_batch_bytes: int | None = None
    streaming_read: bool = False
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
                max_batch_rows=self.max_batch_rows,
                max_batch_bytes=self.max_batch_bytes,
                streaming_read=self.streaming_read,
            ),
        ]
