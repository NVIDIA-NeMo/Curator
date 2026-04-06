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

from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs
from pyarrow.fs import FSSpecHandler, PyFileSystem

from nemo_curator.core.utils import split_table_by_group_max_bytes
from nemo_curator.stages.interleaved.utils import resolve_storage_options
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA, RESERVED_COLUMNS

from .base import BaseInterleavedReader


@dataclass
class InterleavedParquetReaderStage(BaseInterleavedReader):
    """Read interleaved Parquet files into an ``InterleavedBatch``.

    *fields* lists extra (passthrough) column names to read beyond the reserved
    schema columns.  Any *fields* entry that is absent from a given file is
    null-filled, consistent with how the WebDataset reader handles ``fields``.
    Reserved columns are always read regardless of *fields*.

    When *max_batch_bytes* is set, the combined table is split into multiple
    batches so that no single batch exceeds the byte limit.  Each split's
    ``source_files`` metadata lists only the parquet files that contributed
    rows to that batch.
    """

    fields: tuple[str, ...] | None = None
    max_batch_bytes: int | None = None
    name: str = "interleaved_parquet_reader"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._storage_options = resolve_storage_options(io_kwargs=self.read_kwargs)

    def _columns_to_read(self, file_schema: pa.Schema) -> list[str] | None:
        """Return the column list to pass to ``pq.read_table``.

        When *fields* is ``None`` (the default) returns ``None``, which tells
        PyArrow to read all columns — non-lossy by default, consistent with the
        WebDataset reader.

        When *fields* is set, returns reserved columns plus those extra columns
        that exist in the file; missing declared fields are null-filled after
        the read.
        """
        if self.fields is None:
            return None
        file_col_set = set(file_schema.names)
        cols = [c for c in RESERVED_COLUMNS if c in file_col_set]
        for f in self.fields:
            if f in file_col_set and f not in RESERVED_COLUMNS:
                cols.append(f)
        return cols

    def _null_fill_missing_columns(self, table: pa.Table) -> pa.Table:
        """Null-fill any reserved or extra *fields* columns absent from *table*.

        Handles both reserved columns (typed from INTERLEAVED_SCHEMA) and user-requested
        passthrough fields (pa.null() typed, resolved later by _align_output).
        A single set() pass avoids duplicate schema introspection.
        """
        existing = set(table.schema.names)
        for schema_field in INTERLEAVED_SCHEMA:
            if schema_field.name not in existing:
                table = table.append_column(schema_field, pa.nulls(len(table), type=schema_field.type))
                existing.add(schema_field.name)
        if self.fields:
            for f in self.fields:
                if f not in existing:
                    table = table.append_column(pa.field(f, pa.null()), pa.nulls(len(table), type=pa.null()))
        return table

    def process(self, task: FileGroupTask) -> InterleavedBatch | list[InterleavedBatch]:
        tables: list[pa.Table] = []
        sample_id_to_path: dict[str, str] = {}

        for path in task.data:
            fs, _ = url_to_fs(path, **(self._storage_options or {}))
            pa_fs = PyFileSystem(FSSpecHandler(fs))
            file_schema = pq.read_schema(path, filesystem=pa_fs)
            columns = self._columns_to_read(file_schema)
            table = pq.read_table(path, columns=columns, filesystem=pa_fs)
            for sid in table["sample_id"].unique().to_pylist():
                sample_id_to_path.setdefault(sid, path)
            tables.append(table)

        if tables:
            combined = pa.concat_tables(tables, promote_options="default")
            combined = self._null_fill_missing_columns(combined)
            combined = self._align_output(combined)
        else:
            base = self.schema if self.schema is not None else INTERLEAVED_SCHEMA
            combined = pa.Table.from_pylist([], schema=base)

        splits = split_table_by_group_max_bytes(combined, "sample_id", self.max_batch_bytes)
        batches: list[InterleavedBatch] = []
        for idx, split in enumerate(splits):
            task_id = f"{task.task_id}_processed" if len(splits) == 1 else f"{task.task_id}_processed_{idx:05d}"
            metadata: dict[str, Any] = dict(task._metadata)
            if len(splits) == 1:
                metadata["source_files"] = list(task.data)
            else:
                metadata["source_files"] = self._source_files_for_split(split, idx, sample_id_to_path, task.data)
            if self._storage_options:
                metadata["source_storage_options"] = self._storage_options
            batches.append(
                InterleavedBatch(
                    task_id=task_id,
                    dataset_name=task.dataset_name,
                    data=split,
                    _metadata=metadata,
                    _stage_perf=task._stage_perf,
                )
            )
        return batches if len(batches) > 1 else batches[0]
