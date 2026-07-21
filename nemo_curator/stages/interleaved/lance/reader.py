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

"""Lance reader for row-wise interleaved multimodal datasets."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.io.reader.lance import LancePartitioningStage, LanceReaderStage, _pop_dataset_kwargs
from nemo_curator.tasks import EmptyTask, InterleavedBatch, LanceReadTask
from nemo_curator.utils.lance import add_lance_metadata_columns

if TYPE_CHECKING:
    from nemo_curator.stages.text.io.reader.base import ReaderOutput


_GROUP_COLUMN = "sample_id"


def _validate_positive_optional(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        msg = f"{name} must be > 0, got {value}"
        raise ValueError(msg)


def _split_table_by_consecutive_group(table: pa.Table, group_column: str) -> list[pa.Table]:
    """Split a table into consecutive group slices without reordering rows."""
    if table.num_rows == 0:
        return []
    if group_column not in table.column_names:
        msg = f"Group column '{group_column}' not found in table"
        raise ValueError(msg)
    col = table[group_column].combine_chunks()
    if pc.any(pc.is_null(col)).as_py():
        msg = f"Group column '{group_column}' contains null values"
        raise ValueError(msg)
    if table.num_rows == 1:
        return [table]

    group_change = pc.not_equal(col.slice(1), col.slice(0, table.num_rows - 1))
    group_change = pc.fill_null(group_change, False)
    split_points = pc.indices_nonzero(group_change).to_pylist()
    starts = [0, *(point + 1 for point in split_points)]
    ends = [*(point + 1 for point in split_points), table.num_rows]
    return [table.slice(start, end - start) for start, end in zip(starts, ends, strict=True)]


def _append_group_to_size_limited_chunks(  # noqa: PLR0913
    group: pa.Table,
    *,
    max_batch_bytes: int | None,
    max_batch_rows: int | None,
    chunks: list[pa.Table],
    pending_groups: list[pa.Table],
    pending_bytes: list[int],
    pending_rows: list[int],
) -> None:
    """Append one whole group, flushing before it if it would exceed limits."""
    group_bytes = group.nbytes
    group_rows = group.num_rows
    should_flush = bool(
        pending_groups
        and (
            (max_batch_bytes is not None and pending_bytes[0] + group_bytes > max_batch_bytes)
            or (max_batch_rows is not None and pending_rows[0] + group_rows > max_batch_rows)
        )
    )
    if should_flush:
        chunks.append(pa.concat_tables(pending_groups, promote_options="default"))
        pending_groups.clear()
        pending_bytes[0] = 0
        pending_rows[0] = 0

    pending_groups.append(group)
    pending_bytes[0] += group_bytes
    pending_rows[0] += group_rows


def _tables_from_record_batches(  # noqa: C901
    batches: object,
    *,
    group_column: str,
    max_batch_bytes: int | None,
    max_batch_rows: int | None,
    max_output_rows: int | None,
) -> tuple[list[pa.Table], bool]:
    """Build size-limited tables without splitting consecutive groups.

    The last group in each input scanner batch is carried forward because it
    may continue in the next scanner batch. If ``max_output_rows`` stops the
    stream early, the in-progress group is not emitted as a partial sample.
    """
    chunks: list[pa.Table] = []
    pending_groups: list[pa.Table] = []
    pending_bytes = [0]
    pending_rows = [0]
    carry: pa.Table | None = None
    emitted_rows = 0
    stopped_early = False

    def flush_pending() -> None:
        nonlocal emitted_rows
        if not pending_groups:
            return
        chunk = pa.concat_tables(pending_groups, promote_options="default")
        chunks.append(chunk)
        emitted_rows += chunk.num_rows
        pending_groups.clear()
        pending_bytes[0] = 0
        pending_rows[0] = 0

    def output_limit_reached(next_group_rows: int) -> bool:
        return max_output_rows is not None and emitted_rows + pending_rows[0] + next_group_rows > max_output_rows

    for batch in batches:
        table = pa.Table.from_batches([batch]) if isinstance(batch, pa.RecordBatch) else batch
        if not isinstance(table, pa.Table) or table.num_rows == 0:
            continue
        if carry is not None and carry.num_rows:
            table = pa.concat_tables([carry, table], promote_options="default")
            carry = None

        groups = _split_table_by_consecutive_group(table, group_column)
        if not groups:
            continue
        complete_groups = groups[:-1]
        carry = groups[-1]

        for group in complete_groups:
            if output_limit_reached(group.num_rows):
                flush_pending()
                stopped_early = True
                break
            _append_group_to_size_limited_chunks(
                group,
                max_batch_bytes=max_batch_bytes,
                max_batch_rows=max_batch_rows,
                chunks=chunks,
                pending_groups=pending_groups,
                pending_bytes=pending_bytes,
                pending_rows=pending_rows,
            )
        if stopped_early:
            break

    if not stopped_early and carry is not None and carry.num_rows:
        if output_limit_reached(carry.num_rows):
            stopped_early = True
        else:
            _append_group_to_size_limited_chunks(
                carry,
                max_batch_bytes=max_batch_bytes,
                max_batch_rows=max_batch_rows,
                chunks=chunks,
                pending_groups=pending_groups,
                pending_bytes=pending_bytes,
                pending_rows=pending_rows,
            )

    flush_pending()
    return chunks, stopped_early


def _reader_metadata(
    metadata: dict[str, Any],
    *,
    streaming_read: bool,
    stopped_early: bool,
) -> dict[str, Any]:
    metadata = copy.deepcopy(metadata)
    lance_metadata = dict(metadata.get("lance", {}))
    lance_metadata["streaming_read"] = streaming_read
    lance_metadata["stopped_early"] = stopped_early
    metadata["lance"] = lance_metadata
    return metadata


@dataclass
class InterleavedLanceReaderStage(LanceReaderStage):
    """Read Lance fragments into validated ``InterleavedBatch`` objects."""

    max_batch_bytes: int | None = None
    max_batch_rows: int | None = None
    streaming_read: bool = False
    max_output_rows: int | None = None
    name: str = "interleaved_lance_reader"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.fields is not None:
            missing = sorted(InterleavedBatch.REQUIRED_COLUMNS - set(self.fields))
            if missing:
                msg = f"Interleaved Lance fields omit required columns: {missing}"
                raise ValueError(msg)
        _validate_positive_optional("max_batch_bytes", self.max_batch_bytes)
        _validate_positive_optional("max_batch_rows", self.max_batch_rows)
        _validate_positive_optional("max_output_rows", self.max_output_rows)

    def process(self, task: LanceReadTask) -> InterleavedBatch | list[InterleavedBatch]:
        started = time.perf_counter()
        if self.streaming_read:
            splits, metadata = self._stream_read_task(task)
        else:
            output: ReaderOutput = self.read_task(task, dict(self.read_kwargs or {}), self.fields)
            self._validate_result(task, output.data)
            splits, stopped_early = _tables_from_record_batches(
                [output.data],
                group_column=_GROUP_COLUMN,
                max_batch_bytes=self.max_batch_bytes,
                max_batch_rows=self.max_batch_rows,
                max_output_rows=self.max_output_rows,
            )
            metadata = _reader_metadata(
                output.metadata if output.metadata is not None else task._metadata,
                streaming_read=False,
                stopped_early=stopped_early,
            )

        batches = [
            InterleavedBatch(
                dataset_name=task.dataset_name,
                data=split,
                _metadata=metadata,
                _stage_perf=task._stage_perf,
            )
            for split in splits
        ]
        for batch in batches:
            if batch.to_pyarrow().num_rows and not batch.validate():
                msg = f"Lance fragment task {task.task_id} is not a valid InterleavedBatch"
                raise ValueError(msg)

        rows = sum(split.num_rows for split in splits)
        bytes_ = sum(split.nbytes for split in splits)
        self._log_metrics(
            {
                "reader_process_seconds": time.perf_counter() - started,
                "reader_output_splits": float(len(splits)),
                "reader_output_rows": float(rows),
                "reader_output_bytes": float(bytes_),
            }
        )
        if not batches:
            return []
        return batches if len(batches) > 1 else batches[0]

    def _stream_read_task(self, task: LanceReadTask) -> tuple[list[pa.Table], dict[str, Any]]:
        import lance
        from lance.schema import schema_to_json

        read_kwargs = dict(self.read_kwargs or {})
        dataset_kwargs = _pop_dataset_kwargs(read_kwargs)
        dataset_kwargs["version"] = task.version

        scanner_kwargs = self._scanner_kwargs(read_kwargs, self.fields)
        dataset = lance.dataset(task.path, **dataset_kwargs)
        fragments = [dataset.get_fragment(fragment_id) for fragment_id in task.data]
        requested_columns = scanner_kwargs.get("columns")
        blob_columns = [
            field.name
            for field in dataset.schema
            if getattr(field.type, "extension_name", None) == "lance.blob.v2"
            and (requested_columns is None or field.name in requested_columns)
        ]
        if blob_columns:
            msg = "streaming InterleavedLanceReaderStage does not support lance.blob.v2 columns"
            raise NotImplementedError(msg)
        if self.include_lance_metadata:
            scanner_kwargs["with_row_address"] = True
            scanner_kwargs["with_row_id"] = True
        scanner_kwargs["fragments"] = fragments

        splits, stopped_early = _tables_from_record_batches(
            dataset.scanner(**scanner_kwargs).to_batches(),
            group_column=_GROUP_COLUMN,
            max_batch_bytes=self.max_batch_bytes,
            max_batch_rows=self.max_batch_rows,
            max_output_rows=self.max_output_rows,
        )
        if not splits:
            self._validate_result(task, pa.Table.from_pylist([]))
        if self.include_lance_metadata:
            splits = [add_lance_metadata_columns(split) for split in splits]

        metadata = {
            "source_files": [task.path],
            "lance": {
                "version": task.version,
                "fragment_ids": list(task.data),
                "schema": schema_to_json(dataset.schema),
                "has_stable_row_ids": dataset.has_stable_row_ids,
                "streaming_read": True,
                "stopped_early": stopped_early,
            },
        }
        return splits, metadata


@dataclass
class InterleavedLanceReader(CompositeStage[EmptyTask, InterleavedBatch]):
    """Partition and read a Lance dataset as row-wise interleaved batches."""

    path: str
    fragments_per_partition: int = 1
    fields: list[str] | None = None
    max_batch_bytes: int | None = None
    max_batch_rows: int | None = None
    streaming_read: bool = False
    max_output_rows: int | None = None
    read_kwargs: dict[str, Any] | None = None
    include_lance_metadata: bool = True
    fragment_ids: list[int] | None = None
    name: str = "interleaved_lance_reader"

    def __post_init__(self) -> None:
        super().__init__()
        self.read_kwargs = {} if self.read_kwargs is None else dict(self.read_kwargs)

    def decompose(self) -> list[ProcessingStage]:
        return [
            LancePartitioningStage(
                path=self.path,
                fragments_per_partition=self.fragments_per_partition,
                fragment_ids=self.fragment_ids,
                read_kwargs=self.read_kwargs,
            ),
            InterleavedLanceReaderStage(
                fields=self.fields,
                max_batch_bytes=self.max_batch_bytes,
                max_batch_rows=self.max_batch_rows,
                streaming_read=self.streaming_read,
                max_output_rows=self.max_output_rows,
                read_kwargs=self.read_kwargs,
                include_lance_metadata=self.include_lance_metadata,
            ),
        ]
