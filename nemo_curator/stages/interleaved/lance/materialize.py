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

"""Stable-row-id Lance image materialization for interleaved batches."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import TYPE_CHECKING, Literal

import pyarrow as pa

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import InterleavedBatch

from .fetch import (
    ImageFetchMode,
    _as_table,
    _LanceFetchTimeoutError,
    _LanceRowIdFetcher,
    _RowIdFetchResult,
    _slice_fetched_tables,
)

if TYPE_CHECKING:
    from .config import LanceTableConfig

ExistingColumnPolicy = Literal["error", "fill_null", "overwrite"]


def _is_lance_blob_type(data_type: pa.DataType) -> bool:
    return getattr(data_type, "extension_name", None) == "lance.blob.v2"


def _is_binary_like_type(data_type: pa.DataType) -> bool:
    return pa.types.is_binary(data_type) or pa.types.is_large_binary(data_type) or _is_lance_blob_type(data_type)


def _projected_type(source_type: pa.DataType, existing_type: pa.DataType | None) -> pa.DataType:
    if existing_type is not None:
        return existing_type
    if _is_lance_blob_type(source_type):
        return pa.large_binary()
    return source_type


@dataclass(frozen=True)
class _PreparedRowIdTask:
    task: InterleavedBatch
    table: pa.Table
    requested_indices: list[int]
    requested_row_ids: list[int]


@dataclass
class LanceRowIdImageMaterializationStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Materialize interleaved image rows from Lance stable row IDs.

    This stage performs no URL lookup. It expects upstream enrichment to add a
    stable Lance row-id column for rows whose image bytes should be fetched.
    """

    dataset: LanceTableConfig
    input_row_id_column: str = "lance_row_id"
    input_row_id_json_field: str | None = None
    columns: dict[str, str] = field(default_factory=lambda: {"image": "binary_content", "mime_type": "content_type"})
    presence_column: str | None = None
    existing_column_policy: ExistingColumnPolicy = "fill_null"
    fetch_batch_size: int = 512
    io_threads: int = 32
    metadata_cache_size_bytes: int = 1024**3
    sort_row_ids_for_fetch: bool = False
    fetch_mode: ImageFetchMode = "in_process"
    fetch_timeout_seconds: float = 600.0
    fetch_retries: int = 3
    fetcher_max_batches: int = 0
    name: str = "lance_rowid_image_materialization"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    _fetcher: _LanceRowIdFetcher | None = field(default=None, init=False, repr=False)
    _fetcher_batches: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.columns = dict(self.columns or {})
        self._validate_column_config()
        self._validate_batch_config()

    def _validate_column_config(self) -> None:
        if not self.input_row_id_column:
            msg = "input_row_id_column must not be empty"
            raise ValueError(msg)
        if self.input_row_id_json_field == "":
            msg = "input_row_id_json_field must be non-empty when provided"
            raise ValueError(msg)
        if not self.columns and not self.presence_column:
            msg = "columns may be empty only when presence_column is configured"
            raise ValueError(msg)
        if len(set(self.columns.values())) != len(self.columns):
            msg = "Each Lance source column must map to a distinct destination column"
            raise ValueError(msg)
        if self.presence_column in self.columns.values():
            msg = "presence_column must not also be a projected destination column"
            raise ValueError(msg)

    def _validate_batch_config(self) -> None:
        if self.existing_column_policy not in {"error", "fill_null", "overwrite"}:
            msg = f"Unsupported existing_column_policy: {self.existing_column_policy}"
            raise ValueError(msg)
        if self.fetch_mode not in {"in_process", "subprocess", "subprocess_on_timeout"}:
            msg = f"Unsupported fetch_mode: {self.fetch_mode}"
            raise ValueError(msg)
        for name, value in {
            "fetch_batch_size": self.fetch_batch_size,
            "io_threads": self.io_threads,
            "metadata_cache_size_bytes": self.metadata_cache_size_bytes,
        }.items():
            if value <= 0:
                msg = f"{name} must be greater than 0"
                raise ValueError(msg)
        if self.fetch_timeout_seconds < 0:
            msg = "fetch_timeout_seconds must be non-negative"
            raise ValueError(msg)
        if self.fetch_retries < 0:
            msg = "fetch_retries must be non-negative"
            raise ValueError(msg)
        if self.fetcher_max_batches < 0:
            msg = "fetcher_max_batches must be non-negative"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.input_row_id_column]

    def outputs(self) -> tuple[list[str], list[str]]:
        output_columns = list(self.columns.values())
        if self.presence_column:
            output_columns.append(self.presence_column)
        return ["data"], output_columns

    def teardown(self) -> None:
        self._close_fetcher(wait_for_fetches=True)

    def _make_fetcher(self, *, fetch_mode: ImageFetchMode) -> _LanceRowIdFetcher:
        return _LanceRowIdFetcher(
            table_config=self.dataset,
            columns=self.columns,
            fetch_batch_size=self.fetch_batch_size,
            io_threads=self.io_threads,
            metadata_cache_size_bytes=self.metadata_cache_size_bytes,
            sort_row_ids_for_fetch=self.sort_row_ids_for_fetch,
            fetch_timeout_seconds=self.fetch_timeout_seconds,
            fetch_mode=fetch_mode,
        )

    def _ensure_fetcher(self) -> _LanceRowIdFetcher:
        if self._fetcher is None:
            self._fetcher = self._make_fetcher(fetch_mode=self.fetch_mode)
            self._fetcher_batches = 0
        return self._fetcher

    def _close_fetcher(self, *, wait_for_fetches: bool) -> None:
        if self._fetcher is None:
            return
        self._fetcher.close(wait_for_fetches=wait_for_fetches)
        self._fetcher = None
        self._fetcher_batches = 0

    def _maybe_recycle_fetcher_after_success(self) -> None:
        if self._fetcher is None:
            return
        self._fetcher_batches += 1
        if self.fetcher_max_batches <= 0 or self._fetcher_batches < self.fetcher_max_batches:
            return
        self._close_fetcher(wait_for_fetches=True)

    def _fetch_requested_images_subprocess_fallback(self, requested_row_ids: list[int]) -> _RowIdFetchResult:
        fallback_fetcher = self._make_fetcher(fetch_mode="subprocess")
        try:
            return fallback_fetcher.fetch(requested_row_ids)
        finally:
            fallback_fetcher.close(wait_for_fetches=True)

    def _fetch_requested_images(self, requested_row_ids: list[int]) -> tuple[_RowIdFetchResult, int]:
        max_attempts = self.fetch_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                return self._ensure_fetcher().fetch(requested_row_ids), attempt
            except _LanceFetchTimeoutError as exc:
                self._close_fetcher(wait_for_fetches=False)
                if self.fetch_mode == "subprocess_on_timeout":
                    try:
                        return self._fetch_requested_images_subprocess_fallback(requested_row_ids), attempt + 1
                    except _LanceFetchTimeoutError as fallback_exc:
                        if attempt >= max_attempts:
                            msg = (
                                f"Lance image fetch timed out after {attempt} attempts and subprocess fallback "
                                f"(timeout={self.fetch_timeout_seconds:.1f}s)"
                            )
                            raise RuntimeError(msg) from fallback_exc
                if attempt >= max_attempts:
                    msg = (
                        f"Lance image fetch timed out after {attempt} attempts "
                        f"(timeout={self.fetch_timeout_seconds:.1f}s)"
                    )
                    raise RuntimeError(msg) from exc
        msg = "unreachable Lance fetch retry state"
        raise RuntimeError(msg)

    def _validate_input_table(self, table: pa.Table, source_types: dict[str, pa.DataType]) -> None:
        if self.input_row_id_column not in table.column_names:
            msg = f"Input row-id column {self.input_row_id_column!r} does not exist"
            raise ValueError(msg)
        collisions = sorted(set(self.columns.values()) & set(table.column_names))
        if collisions and self.existing_column_policy == "error":
            msg = f"Projected destination columns already exist: {collisions}"
            raise ValueError(msg)
        for source, destination in self.columns.items():
            if destination not in table.column_names:
                continue
            destination_type = table.schema.field(destination).type
            source_type = source_types[source]
            if destination_type == source_type:
                continue
            if _is_binary_like_type(destination_type) and _is_binary_like_type(source_type):
                continue
            msg = (
                f"Destination column {destination!r} has type {destination_type}; "
                f"Lance column {source!r} has type {source_type}"
            )
            raise TypeError(msg)
        if self.presence_column in table.column_names and not pa.types.is_boolean(
            table.schema.field(self.presence_column).type
        ):
            msg = f"Presence column {self.presence_column!r} must have boolean type"
            raise TypeError(msg)
        input_type = table.schema.field(self.input_row_id_column).type
        if self.input_row_id_json_field is not None and not pa.types.is_string(input_type):
            msg = "input_row_id_json_field requires a string input row-id column"
            raise TypeError(msg)
        if self.input_row_id_json_field is None and not (
            pa.types.is_integer(input_type) or pa.types.is_string(input_type)
        ):
            msg = f"Input row-id column has type {input_type}; expected an integer or string column"
            raise TypeError(msg)

    @staticmethod
    def _coerce_row_id(value: object) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            row_id = int(value)
        except (TypeError, ValueError):
            return None
        return row_id if row_id >= 0 else None

    def _extract_row_id(self, value: object) -> int | None:
        if self.input_row_id_json_field is None:
            return self._coerce_row_id(value)
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text) if text.startswith("{") else None
        except JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict):
            return None
        return self._coerce_row_id(parsed.get(self.input_row_id_json_field))

    def _table_row_ids(self, table: pa.Table) -> list[int | None]:
        return [self._extract_row_id(value) for value in table[self.input_row_id_column].combine_chunks().to_pylist()]

    def _requested_indices(
        self,
        table: pa.Table,
        row_ids: list[int | None],
        presence: list[bool | None] | None,
    ) -> tuple[list[int], list[int]]:
        destination_values = {
            destination: table[destination].combine_chunks().to_pylist()
            for destination in self.columns.values()
            if destination in table.column_names
        }
        indices: list[int] = []
        requested_row_ids: list[int] = []
        for index, row_id in enumerate(row_ids):
            if row_id is None or (presence is not None and presence[index] is False):
                continue
            if self.existing_column_policy == "fill_null" and self.columns:
                all_populated = all(
                    destination in destination_values and destination_values[destination][index] is not None
                    for destination in self.columns.values()
                )
                presence_populated = presence is None or presence[index] is not None
                if all_populated and presence_populated:
                    continue
            elif not self.columns and presence is not None and presence[index] is not None:
                continue
            indices.append(index)
            requested_row_ids.append(row_id)
        return indices, requested_row_ids

    def _prepare_task(
        self,
        task: InterleavedBatch,
        source_types: dict[str, pa.DataType],
    ) -> _PreparedRowIdTask:
        table = task.to_pyarrow()
        self._validate_input_table(table, source_types)
        row_ids = self._table_row_ids(table)
        presence = (
            table[self.presence_column].combine_chunks().to_pylist()
            if self.presence_column and self.presence_column in table.column_names
            else None
        )
        requested_indices, requested_row_ids = self._requested_indices(table, row_ids, presence)
        return _PreparedRowIdTask(
            task=task,
            table=table,
            requested_indices=requested_indices,
            requested_row_ids=requested_row_ids,
        )

    def _flatten_fetched_values(self, fetch_result: _RowIdFetchResult) -> dict[str, list[object]]:
        values = {source: [] for source in self.columns}
        for table in fetch_result.tables:
            for source in self.columns:
                values[source].extend(table[source].combine_chunks().to_pylist())
        return values

    def _can_replace_whole_columns(self, table: pa.Table, requested_indices: list[int]) -> bool:
        if requested_indices != list(range(table.num_rows)):
            return False
        if self.existing_column_policy != "fill_null":
            return True
        for destination in self.columns.values():
            if destination in table.column_names and table[destination].null_count != table.num_rows:
                return False
        return True

    @staticmethod
    def _write_column(table: pa.Table, destination: str, array: pa.ChunkedArray | pa.Array) -> pa.Table:
        idx = table.schema.get_field_index(destination)
        if idx >= 0:
            return table.set_column(idx, destination, array)
        return table.append_column(destination, array)

    @staticmethod
    def _concat_fetched_tables(fetch_result: _RowIdFetchResult) -> pa.Table:
        return _as_table(fetch_result.tables)

    @staticmethod
    def _coerce_fetched_column(
        column: pa.ChunkedArray,
        source_type: pa.DataType,
        existing_type: pa.DataType | None,
    ) -> pa.ChunkedArray | pa.Array:
        target_type = _projected_type(source_type, existing_type)
        if column.type == target_type:
            return column
        try:
            return column.cast(target_type)
        except (pa.ArrowInvalid, pa.ArrowTypeError, NotImplementedError):
            return pa.array(column.combine_chunks().to_pylist(), type=target_type, from_pandas=True)

    def _replace_whole_columns(
        self,
        table: pa.Table,
        requested_indices: list[int],
        fetch_result: _RowIdFetchResult,
        source_types: dict[str, pa.DataType],
    ) -> pa.Table:
        result = table
        fetched_table = self._concat_fetched_tables(fetch_result)
        if fetched_table.num_rows != len(requested_indices):
            msg = f"Lance returned {fetched_table.num_rows} rows for {len(requested_indices)} requested stable row IDs"
            raise RuntimeError(msg)
        for source, destination in self.columns.items():
            existing_type = result.schema.field(destination).type if destination in result.schema.names else None
            column = self._coerce_fetched_column(fetched_table[source], source_types[source], existing_type)
            result = self._write_column(result, destination, column)
        return result

    def _scatter_fetched_columns(
        self,
        table: pa.Table,
        requested_indices: list[int],
        fetch_result: _RowIdFetchResult,
        source_types: dict[str, pa.DataType],
    ) -> pa.Table:
        result = table
        fetched_values = self._flatten_fetched_values(fetch_result)
        fetched_rows = len(next(iter(fetched_values.values()), []))
        if fetched_rows != len(requested_indices):
            msg = f"Lance returned {fetched_rows} rows for {len(requested_indices)} requested stable row IDs"
            raise RuntimeError(msg)

        for source, destination in self.columns.items():
            existing_type = result.schema.field(destination).type if destination in result.schema.names else None
            if existing_type is not None:
                values = result[destination].combine_chunks().to_pylist()
            else:
                values = [None] * result.num_rows
            for fetched_index, row_index in enumerate(requested_indices):
                if self.existing_column_policy == "fill_null" and values[row_index] is not None:
                    continue
                values[row_index] = fetched_values[source][fetched_index]
            array = pa.array(values, type=_projected_type(source_types[source], existing_type), from_pandas=True)
            result = self._write_column(result, destination, array)
        return result

    def _apply_projection(
        self,
        table: pa.Table,
        requested_indices: list[int],
        fetch_result: _RowIdFetchResult,
        source_types: dict[str, pa.DataType],
    ) -> pa.Table:
        if not self.columns:
            return table
        if self._can_replace_whole_columns(table, requested_indices):
            return self._replace_whole_columns(table, requested_indices, fetch_result, source_types)
        return self._scatter_fetched_columns(table, requested_indices, fetch_result, source_types)

    def _apply_presence(self, table: pa.Table, requested_indices: list[int]) -> pa.Table:
        if not self.presence_column:
            return table
        if self.presence_column in table.column_names:
            values = table[self.presence_column].combine_chunks().to_pylist()
        else:
            values = [None] * table.num_rows
        for index in requested_indices:
            values[index] = True
        presence = pa.array(values, type=pa.bool_(), from_pandas=True)
        column_index = table.schema.get_field_index(self.presence_column)
        if column_index >= 0:
            return table.set_column(column_index, self.presence_column, presence)
        return table.append_column(self.presence_column, presence)

    def _process_tasks(self, tasks: list[InterleavedBatch]) -> list[InterleavedBatch]:
        if len(tasks) == 0:
            return []

        process_started = time.perf_counter()
        fetcher = self._ensure_fetcher()
        source_types = fetcher.source_types
        prepared = [self._prepare_task(task, source_types) for task in tasks]
        requested_row_ids = [row_id for prepared_task in prepared for row_id in prepared_task.requested_row_ids]
        fetch_result, fetch_attempts = self._fetch_requested_images(requested_row_ids)
        self._maybe_recycle_fetcher_after_success()

        outputs: list[InterleavedBatch] = []
        fetched_offset = 0
        for prepared_task in prepared:
            fetched_count = len(prepared_task.requested_row_ids)
            task_fetch_result = _RowIdFetchResult(
                tables=_slice_fetched_tables(fetch_result.tables, fetched_offset, fetched_count),
                fetch_seconds=fetch_result.fetch_seconds,
                fetched_bytes_by_column=fetch_result.fetched_bytes_by_column,
                read_bytes=fetch_result.read_bytes,
                read_iops=fetch_result.read_iops,
            )
            fetched_offset += fetched_count
            result = self._apply_projection(
                prepared_task.table,
                prepared_task.requested_indices,
                task_fetch_result,
                source_types,
            )
            result = self._apply_presence(result, prepared_task.requested_indices)
            outputs.append(
                InterleavedBatch(
                    dataset_name=prepared_task.task.dataset_name,
                    data=result,
                    _metadata=prepared_task.task._metadata,
                    _stage_perf=prepared_task.task._stage_perf,
                )
            )

        metrics = {
            "input_tasks": float(len(prepared)),
            "input_rows": float(sum(prepared_task.table.num_rows for prepared_task in prepared)),
            "requested_row_ids": float(len(requested_row_ids)),
            "materializer_seconds": time.perf_counter() - process_started,
            "lance_fetch_seconds": fetch_result.fetch_seconds,
            "lance_fetch_attempts": float(fetch_attempts),
            "lance_fetch_retries": float(max(fetch_attempts - 1, 0)),
            "lance_fetch_timeout_seconds": float(self.fetch_timeout_seconds),
            "lance_fetch_max_retries": float(self.fetch_retries),
            "lance_fetcher_max_batches": float(self.fetcher_max_batches),
            "lance_fetched_bytes": float(sum(fetch_result.fetched_bytes_by_column.values())),
            "lance_read_bytes": float(fetch_result.read_bytes),
            "lance_read_iops": float(fetch_result.read_iops),
            "fetch_mode.subprocess": float(self.fetch_mode == "subprocess"),
        }
        for source, value in fetch_result.fetched_bytes_by_column.items():
            metrics[f"lance_fetched_{source}_bytes"] = float(value)
        self._log_metrics(metrics)
        return outputs

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        return self._process_tasks([task])[0]

    def process_batch(self, tasks: list[InterleavedBatch]) -> list[InterleavedBatch]:
        return self._process_tasks(tasks)
