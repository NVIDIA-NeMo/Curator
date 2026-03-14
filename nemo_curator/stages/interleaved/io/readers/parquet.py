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

from nemo_curator.core.utils import split_table_by_group_max_bytes
from nemo_curator.stages.interleaved.utils import resolve_storage_options
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

from .base import BaseInterleavedReader


@dataclass
class InterleavedParquetReaderStage(BaseInterleavedReader):
    """Read parquet files in interleaved schema into InterleavedBatch.

    Uses native pyarrow reading. Columns listed in *fields* that are absent
    from a file are filled with typed nulls using the declared *schema*
    (defaults to ``INTERLEAVED_SCHEMA``). The output table is then aligned
    to the schema via :func:`align_table`.
    """

    fields: list[str] | None = None
    max_batch_bytes: int | None = None
    name: str = "interleaved_parquet_reader"

    def process(self, task: FileGroupTask) -> InterleavedBatch | list[InterleavedBatch]:
        storage_options = resolve_storage_options(io_kwargs=self.read_kwargs)
        effective_kwargs: dict[str, Any] = dict(self.read_kwargs) if self.read_kwargs else {}
        effective_kwargs.pop("storage_options", None)

        if "filesystem" not in effective_kwargs and storage_options:
            fs, _ = url_to_fs(task.data[0], **storage_options)
            effective_kwargs["filesystem"] = fs

        tables: list[pa.Table] = []
        all_missing: set[str] = set()
        _filesystem = effective_kwargs.get("filesystem")
        for path in task.data:
            file_columns = set(pq.read_schema(path, filesystem=_filesystem).names)
            existing_cols = [c for c in self.fields if c in file_columns] if self.fields is not None else None
            if self.fields is not None:
                all_missing.update(c for c in self.fields if c not in file_columns)
            tables.append(pq.read_table(path, columns=existing_cols, **effective_kwargs))

        if not tables:
            msg = f"No data read from parquet files in task {task.task_id}"
            raise ValueError(msg)

        table = pa.concat_tables(tables, promote_options="default")

        type_lookup = self.schema if self.schema is not None else INTERLEAVED_SCHEMA
        for col_name in sorted(all_missing):
            if col_name not in table.column_names:
                field_type = type_lookup.field(col_name).type if col_name in type_lookup.names else pa.null()
                table = table.append_column(col_name, pa.nulls(table.num_rows, type=field_type))

        table = self._align_output(table)
        splits = split_table_by_group_max_bytes(table, "sample_id", self.max_batch_bytes)

        metadata = dict(task._metadata)
        if storage_options:
            metadata["source_storage_options"] = storage_options

        batches: list[InterleavedBatch] = []
        for idx, split in enumerate(splits):
            task_id = f"{task.task_id}_processed" if len(splits) == 1 else f"{task.task_id}_processed_{idx:05d}"
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
