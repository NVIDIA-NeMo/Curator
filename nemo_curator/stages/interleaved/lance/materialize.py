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
from typing import Literal

import pyarrow as pa

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import InterleavedBatch

from .config import LanceTableConfig  # noqa: TC001
from .fetch import (
    _as_table,
    _LanceFetchTimeoutError,
    _LanceImageFetcherBase,
    _LanceRowAddress,
    _LanceRowAddressFetcher,
    _LanceRowIdFetcher,
    _RowIdFetchResult,
    _slice_fetched_tables,
)

ExistingColumnPolicy = Literal["error", "fill_null", "overwrite"]
LanceImageAddressMode = Literal["row_id", "row_address"]
LanceImageAddress = int | _LanceRowAddress


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
class _PreparedImageTask:
    task: InterleavedBatch
    table: pa.Table
    requested_indices: list[int]
    requested_addresses: list[LanceImageAddress]


@dataclass
class LanceRowIdImageMaterializationStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Materialize interleaved image rows from Lance image addresses.

    This stage performs no URL lookup. It expects upstream enrichment to add a
    stable Lance row-id column, or Lance fragment/row-offset columns, for rows
    whose image bytes should be fetched.
    """

    dataset: LanceTableConfig
    address_mode: LanceImageAddressMode = "row_id"
    input_row_id_column: str = "lance_row_id"
    input_row_id_json_field: str | None = None
    input_fragment_id_column: str = "lance_fragment_id"
    input_row_offset_column: str = "lance_row_offset"
    columns: dict[str, str] = field(default_factory=lambda: {"image": "binary_content", "mime_type": "content_type"})
    presence_column: str | None = None
    existing_column_policy: ExistingColumnPolicy = "fill_null"
    fetch_batch_size: int = 512
    io_threads: int = 32
    metadata_cache_size_bytes: int = 1024**3
    sort_row_ids_for_fetch: bool = False
    fetch_timeout_seconds: float = 600.0
    fetch_retries: int = 3
    fetcher_max_batches: int = 0
    name: str = "lance_rowid_image_materialization"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    _fetcher: _LanceImageFetcherBase | None = field(default=None, init=False, repr=False)
    _fetcher_batches: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.columns = dict(self.columns or {})
        self._validate_column_config()
        self._validate_batch_config()

    def _validate_column_config(self) -> None:
        self._validate_address_column_config()
        self._validate_projection_column_config()

    def _validate_address_column_config(self) -> None:
        if self.address_mode not in {"row_id", "row_address"}:
            msg = f"Unsupported address_mode: {self.address_mode}"
            raise ValueError(msg)
        if self.address_mode == "row_id" and not self.input_row_id_column:
            msg = "input_row_id_column must not be empty"
            raise ValueError(msg)
        if self.input_row_id_json_field == "":
            msg = "input_row_id_json_field must be non-empty when provided"
            raise ValueError(msg)
        if self.address_mode == "row_address":
            if self.input_row_id_json_field is not None:
                msg = "input_row_id_json_field is only supported for address_mode='row_id'"
                raise ValueError(msg)
            if not self.input_fragment_id_column:
                msg = "input_fragment_id_column must not be empty"
                raise ValueError(msg)
            if not self.input_row_offset_column:
                msg = "input_row_offset_column must not be empty"
                raise ValueError(msg)

    def _validate_projection_column_config(self) -> None:
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
        if self.address_mode == "row_address":
            return ["data"], [self.input_fragment_id_column, self.input_row_offset_column]
        return ["data"], [self.input_row_id_column]

    def outputs(self) -> tuple[list[str], list[str]]:
        output_columns = list(self.columns.values())
        if self.presence_column:
            output_columns.append(self.presence_column)
        return ["data"], output_columns

    def teardown(self) -> None:
        self._close_fetcher(wait_for_fetches=True)

    def _make_fetcher(self) -> _LanceImageFetcherBase:
        common_kwargs = {
            "table_config": self.dataset,
            "columns": self.columns,
            "fetch_batch_size": self.fetch_batch_size,
            "io_threads": self.io_threads,
            "metadata_cache_size_bytes": self.metadata_cache_size_bytes,
            "fetch_timeout_seconds": self.fetch_timeout_seconds,
        }
        if self.address_mode == "row_address":
            return _LanceRowAddressFetcher(**common_kwargs)
        return _LanceRowIdFetcher(
            **common_kwargs,
            sort_row_ids_for_fetch=self.sort_row_ids_for_fetch,
        )

    def _ensure_fetcher(self) -> _LanceImageFetcherBase:
        if self._fetcher is None:
            self._fetcher = self._make_fetcher()
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

    def _fetch_requested_images(self, requested_addresses: list[LanceImageAddress]) -> tuple[_RowIdFetchResult, int]:
        max_attempts = self.fetch_retries + 1
        attempt = 1
        while True:
            try:
                return self._ensure_fetcher().fetch(requested_addresses), attempt  # type: ignore[arg-type]
            except _LanceFetchTimeoutError as exc:
                self._close_fetcher(wait_for_fetches=False)
                if attempt >= max_attempts:
                    msg = (
                        f"Lance image fetch timed out after {attempt} attempts "
                        f"(timeout={self.fetch_timeout_seconds:.1f}s)"
                    )
                    raise RuntimeError(msg) from exc
                attempt += 1

    def _validate_input_address_columns(self, table: pa.Table) -> None:
        if self.address_mode == "row_address":
            missing_address_columns = [
                column
                for column in (self.input_fragment_id_column, self.input_row_offset_column)
                if column not in table.column_names
            ]
            if missing_address_columns:
                msg = f"Input row-address columns do not exist: {missing_address_columns}"
                raise ValueError(msg)
            for column in (self.input_fragment_id_column, self.input_row_offset_column):
                input_type = table.schema.field(column).type
                if not (pa.types.is_integer(input_type) or pa.types.is_string(input_type)):
                    msg = (
                        f"Input row-address column {column!r} has type {input_type}; "
                        "expected an integer or string column"
                    )
                    raise TypeError(msg)
            return

        if self.input_row_id_column not in table.column_names:
            msg = f"Input row-id column {self.input_row_id_column!r} does not exist"
            raise ValueError(msg)
        input_type = table.schema.field(self.input_row_id_column).type
        if self.input_row_id_json_field is not None and not pa.types.is_string(input_type):
            msg = "input_row_id_json_field requires a string input row-id column"
            raise TypeError(msg)
        if self.input_row_id_json_field is None and not (
            pa.types.is_integer(input_type) or pa.types.is_string(input_type)
        ):
            msg = f"Input row-id column has type {input_type}; expected an integer or string column"
            raise TypeError(msg)

    def _validate_destination_columns(self, table: pa.Table, source_types: dict[str, pa.DataType]) -> None:
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

    def _validate_presence_column(self, table: pa.Table) -> None:
        if self.presence_column in table.column_names and not pa.types.is_boolean(
            table.schema.field(self.presence_column).type
        ):
            msg = f"Presence column {self.presence_column!r} must have boolean type"
            raise TypeError(msg)

    def _validate_input_table(self, table: pa.Table, source_types: dict[str, pa.DataType]) -> None:
        self._validate_input_address_columns(table)
        self._validate_destination_columns(table, source_types)
        self._validate_presence_column(table)

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

    def _table_row_addresses(self, table: pa.Table) -> list[_LanceRowAddress | None]:
        fragment_ids = table[self.input_fragment_id_column].combine_chunks().to_pylist()
        row_offsets = table[self.input_row_offset_column].combine_chunks().to_pylist()
        addresses: list[_LanceRowAddress | None] = []
        for fragment_id_value, row_offset_value in zip(fragment_ids, row_offsets, strict=True):
            fragment_id = self._coerce_row_id(fragment_id_value)
            row_offset = self._coerce_row_id(row_offset_value)
            if fragment_id is None or row_offset is None:
                addresses.append(None)
            else:
                addresses.append(_LanceRowAddress(fragment_id=fragment_id, row_offset=row_offset))
        return addresses

    def _table_addresses(self, table: pa.Table) -> list[LanceImageAddress | None]:
        if self.address_mode == "row_address":
            return self._table_row_addresses(table)
        return self._table_row_ids(table)

    def _requested_indices(
        self,
        table: pa.Table,
        addresses: list[LanceImageAddress | None],
        presence: list[bool | None] | None,
    ) -> tuple[list[int], list[LanceImageAddress]]:
        destination_values = {
            destination: table[destination].combine_chunks().to_pylist()
            for destination in self.columns.values()
            if destination in table.column_names
        }
        indices: list[int] = []
        requested_addresses: list[LanceImageAddress] = []
        for index, address in enumerate(addresses):
            if address is None or (presence is not None and presence[index] is False):
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
            requested_addresses.append(address)
        return indices, requested_addresses

    def _prepare_task(
        self,
        task: InterleavedBatch,
        source_types: dict[str, pa.DataType],
    ) -> _PreparedImageTask:
        table = task.to_pyarrow()
        self._validate_input_table(table, source_types)
        addresses = self._table_addresses(table)
        presence = (
            table[self.presence_column].combine_chunks().to_pylist()
            if self.presence_column and self.presence_column in table.column_names
            else None
        )
        requested_indices, requested_addresses = self._requested_indices(table, addresses, presence)
        return _PreparedImageTask(
            task=task,
            table=table,
            requested_indices=requested_indices,
            requested_addresses=requested_addresses,
        )

    def _flatten_fetched_values(self, fetch_result: _RowIdFetchResult) -> dict[str, list[object]]:
        values = {source: [] for source in self.columns}
        for table in fetch_result.tables:
            for source in self.columns:
                values[source].extend(table[source].combine_chunks().to_pylist())
        return values

    def _can_replace_whole_columns(self, table: pa.Table, requested_indices: list[int]) -> bool:
        if len(requested_indices) != table.num_rows:
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
            msg = f"Lance returned {fetched_table.num_rows} rows for {len(requested_indices)} requested image rows"
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
            msg = f"Lance returned {fetched_rows} rows for {len(requested_indices)} requested image rows"
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
        return self._write_column(table, self.presence_column, presence)

    def _process_tasks(self, tasks: list[InterleavedBatch]) -> list[InterleavedBatch]:
        if len(tasks) == 0:
            return []

        process_started = time.perf_counter()
        fetcher = self._ensure_fetcher()
        source_types = fetcher.source_types
        prepared = [self._prepare_task(task, source_types) for task in tasks]
        requested_addresses = [address for prepared_task in prepared for address in prepared_task.requested_addresses]
        fetch_result, fetch_attempts = self._fetch_requested_images(requested_addresses)
        self._maybe_recycle_fetcher_after_success()

        outputs: list[InterleavedBatch] = []
        fetched_offset = 0
        for prepared_task in prepared:
            fetched_count = len(prepared_task.requested_addresses)
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
            "requested_lance_addresses": float(len(requested_addresses)),
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
            "address_mode.row_id": float(self.address_mode == "row_id"),
            "address_mode.row_address": float(self.address_mode == "row_address"),
        }
        for source, value in fetch_result.fetched_bytes_by_column.items():
            metrics[f"lance_fetched_{source}_bytes"] = float(value)
        self._log_metrics(metrics)
        return outputs

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        return self._process_tasks([task])[0]

    def process_batch(self, tasks: list[InterleavedBatch]) -> list[InterleavedBatch]:
        return self._process_tasks(tasks)
