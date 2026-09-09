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

"""Coordinate-based Lance materialization for interleaved batches."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import InterleavedBatch

from .fetch import (
    _LanceAddress,
    _LanceFetcher,
    _LanceFetchResult,
    _LanceFetchTimeoutError,
    _LanceRowAddress,
)

LanceAddressMode = Literal["row_id", "row_address"]


def _is_binary_like_type(data_type: pa.DataType) -> bool:
    return pa.types.is_binary(data_type) or pa.types.is_large_binary(data_type)


def _projected_type(source_type: pa.DataType, existing_type: pa.DataType | None) -> pa.DataType:
    if existing_type is not None:
        return existing_type
    return source_type


@dataclass(frozen=True)
class _PreparedTask:
    table: pa.Table
    requested_indices: list[int]
    requested_addresses: list[_LanceAddress]


@dataclass
class InterleavedLanceMaterializerStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Materialize interleaved columns from Lance row coordinates.

    This stage performs no URL lookup. It expects upstream enrichment to add a
    stable Lance row-id column, or Lance fragment/row-offset columns, for rows to
    fetch.
    """

    path: str
    version: int | None = None
    storage_options: dict[str, str] = field(default_factory=dict)
    address_mode: LanceAddressMode = "row_id"
    input_row_id_column: str = "lance_row_id"
    input_fragment_id_column: str = "lance_fragment_id"
    input_row_offset_column: str = "lance_row_offset"
    columns: dict[str, str] = field(default_factory=lambda: {"image": "binary_content", "mime_type": "content_type"})
    presence_column: str | None = None
    overwrite_existing: bool = False
    fetch_batch_size: int = 512
    io_threads: int = 32
    metadata_cache_size_bytes: int = 1024**3
    sort_row_ids_for_fetch: bool = False
    fetch_timeout_seconds: float = 600.0
    fetch_retries: int = 3
    name: str = "interleaved_lance_materializer"
    _fetcher: _LanceFetcher | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.path:
            msg = "path must not be empty"
            raise ValueError(msg)
        self.storage_options = dict(self.storage_options or {})
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
        if self.address_mode == "row_address":
            if not self.input_fragment_id_column:
                msg = "input_fragment_id_column must not be empty"
                raise ValueError(msg)
            if not self.input_row_offset_column:
                msg = "input_row_offset_column must not be empty"
                raise ValueError(msg)

    def _validate_projection_column_config(self) -> None:
        if not self.columns:
            msg = "columns must not be empty"
            raise ValueError(msg)
        if len(set(self.columns.values())) != len(self.columns):
            msg = "Each Lance source column must map to a distinct destination column"
            raise ValueError(msg)
        if self.presence_column in self.columns.values():
            msg = "presence_column must not also be a projected destination column"
            raise ValueError(msg)

    def _validate_batch_config(self) -> None:
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

    def _make_fetcher(self) -> _LanceFetcher:
        return _LanceFetcher(
            path=self.path,
            version=self.version,
            storage_options=self.storage_options,
            columns=self.columns,
            fetch_batch_size=self.fetch_batch_size,
            io_threads=self.io_threads,
            metadata_cache_size_bytes=self.metadata_cache_size_bytes,
            address_mode=self.address_mode,
            sort_row_ids_for_fetch=self.sort_row_ids_for_fetch,
            fetch_timeout_seconds=self.fetch_timeout_seconds,
        )

    def _ensure_fetcher(self) -> _LanceFetcher:
        if self._fetcher is None:
            self._fetcher = self._make_fetcher()
        return self._fetcher

    def _close_fetcher(self, *, wait_for_fetches: bool) -> None:
        if self._fetcher is None:
            return
        self._fetcher.close(wait_for_fetches=wait_for_fetches)
        self._fetcher = None

    def _fetch_requested(self, requested_addresses: list[_LanceAddress]) -> tuple[_LanceFetchResult, int]:
        max_attempts = self.fetch_retries + 1
        attempt = 1
        while True:
            try:
                return self._ensure_fetcher().fetch(requested_addresses), attempt
            except _LanceFetchTimeoutError as exc:
                self._close_fetcher(wait_for_fetches=False)
                if attempt >= max_attempts:
                    msg = f"Lance fetch timed out after {attempt} attempts (timeout={self.fetch_timeout_seconds:.1f}s)"
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
                if not pa.types.is_integer(input_type):
                    msg = f"Input row-address column {column!r} has type {input_type}; expected an integer column"
                    raise TypeError(msg)
            return

        if self.input_row_id_column not in table.column_names:
            msg = f"Input row-id column {self.input_row_id_column!r} does not exist"
            raise ValueError(msg)
        input_type = table.schema.field(self.input_row_id_column).type
        if not pa.types.is_integer(input_type):
            msg = f"Input row-id column has type {input_type}; expected an integer column"
            raise TypeError(msg)

    def _validate_destination_columns(self, table: pa.Table, source_types: dict[str, pa.DataType]) -> None:
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

    def _table_addresses(self, table: pa.Table) -> list[_LanceAddress | None]:
        if self.address_mode == "row_id":
            return [
                value if value is None or value >= 0 else None
                for value in table[self.input_row_id_column].combine_chunks().to_pylist()
            ]

        fragment_ids = table[self.input_fragment_id_column].combine_chunks().to_pylist()
        row_offsets = table[self.input_row_offset_column].combine_chunks().to_pylist()
        addresses: list[_LanceRowAddress | None] = []
        for fragment_id_value, row_offset_value in zip(fragment_ids, row_offsets, strict=True):
            if fragment_id_value is None or fragment_id_value < 0 or row_offset_value is None or row_offset_value < 0:
                addresses.append(None)
            else:
                addresses.append(_LanceRowAddress(fragment_id=fragment_id_value, row_offset=row_offset_value))
        return addresses

    def _requested_indices(
        self,
        table: pa.Table,
        addresses: list[_LanceAddress | None],
        presence: list[bool | None] | None,
    ) -> tuple[list[int], list[_LanceAddress]]:
        destination_validity = {
            destination: pc.is_valid(table[destination]).to_pylist()
            for destination in self.columns.values()
            if destination in table.column_names
        }
        indices: list[int] = []
        requested_addresses: list[_LanceAddress] = []
        for index, address in enumerate(addresses):
            if address is None or (presence is not None and presence[index] is False):
                continue
            if not self.overwrite_existing:
                all_populated = all(
                    destination in destination_validity and destination_validity[destination][index]
                    for destination in self.columns.values()
                )
                presence_populated = presence is None or presence[index] is not None
                if all_populated and presence_populated:
                    continue
            indices.append(index)
            requested_addresses.append(address)
        return indices, requested_addresses

    def _prepare_task(
        self,
        task: InterleavedBatch,
        source_types: dict[str, pa.DataType],
    ) -> _PreparedTask:
        table = task.to_pyarrow()
        self._validate_input_table(table, source_types)
        addresses = self._table_addresses(table)
        presence = (
            table[self.presence_column].combine_chunks().to_pylist()
            if self.presence_column and self.presence_column in table.column_names
            else None
        )
        requested_indices, requested_addresses = self._requested_indices(table, addresses, presence)
        return _PreparedTask(
            table=table,
            requested_indices=requested_indices,
            requested_addresses=requested_addresses,
        )

    @staticmethod
    def _write_column(table: pa.Table, destination: str, array: pa.ChunkedArray | pa.Array) -> pa.Table:
        idx = table.schema.get_field_index(destination)
        if idx >= 0:
            return table.set_column(idx, destination, array)
        return table.append_column(destination, array)

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

    @staticmethod
    def _replace_at_indices(
        values: pa.ChunkedArray,
        indices: list[int],
        replacements: pa.ChunkedArray | pa.Array,
    ) -> pa.ChunkedArray:
        """Replace rows without converting variable-size values to Python."""
        if len(indices) != len(replacements):
            msg = f"Received {len(replacements)} replacement values for {len(indices)} row indices"
            raise RuntimeError(msg)
        if not indices:
            return values

        mask = [False] * len(values)
        for index in indices:
            mask[index] = True
        replacement_array = (
            replacements.combine_chunks() if isinstance(replacements, pa.ChunkedArray) else replacements
        )
        return pc.replace_with_mask(values, pa.array(mask, type=pa.bool_()), replacement_array)

    def _projection_updates(
        self,
        existing: pa.ChunkedArray,
        fetched: pa.ChunkedArray | pa.Array,
        requested_indices: list[int],
    ) -> tuple[list[int], pa.ChunkedArray | pa.Array]:
        if self.overwrite_existing:
            return requested_indices, fetched

        requested_existing = pc.take(existing, pa.array(requested_indices, type=pa.int64()))
        replace_mask = pc.is_null(requested_existing)
        indices = [
            index
            for index, should_replace in zip(requested_indices, replace_mask.to_pylist(), strict=True)
            if should_replace
        ]
        return indices, pc.filter(fetched, replace_mask)

    def _scatter_fetched_columns(
        self,
        table: pa.Table,
        requested_indices: list[int],
        fetch_result: _LanceFetchResult,
        source_types: dict[str, pa.DataType],
    ) -> pa.Table:
        result = table
        if fetch_result.table.num_rows != len(requested_indices):
            msg = f"Lance returned {fetch_result.table.num_rows} rows for {len(requested_indices)} requested rows"
            raise RuntimeError(msg)

        for source, destination in self.columns.items():
            existing_type = result.schema.field(destination).type if destination in result.schema.names else None
            target_type = _projected_type(source_types[source], existing_type)
            existing = (
                result[destination]
                if existing_type is not None
                else pa.chunked_array([pa.nulls(result.num_rows, type=target_type)])
            )
            fetched = self._coerce_fetched_column(fetch_result.table[source], source_types[source], existing_type)
            indices, replacements = self._projection_updates(existing, fetched, requested_indices)
            result = self._write_column(
                result,
                destination,
                self._replace_at_indices(existing, indices, replacements),
            )
        return result

    def _apply_presence(self, table: pa.Table, requested_indices: list[int]) -> pa.Table:
        if not self.presence_column:
            return table
        presence = (
            table[self.presence_column]
            if self.presence_column in table.column_names
            else pa.chunked_array([pa.nulls(table.num_rows, type=pa.bool_())])
        )
        presence = self._replace_at_indices(
            presence,
            requested_indices,
            pa.array([True] * len(requested_indices), type=pa.bool_()),
        )
        return self._write_column(table, self.presence_column, presence)

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        process_started = time.perf_counter()
        fetcher = self._ensure_fetcher()
        source_types = fetcher.source_types
        prepared = self._prepare_task(task, source_types)
        requested_addresses = prepared.requested_addresses
        fetch_result, fetch_attempts = self._fetch_requested(requested_addresses)

        result = self._scatter_fetched_columns(
            prepared.table,
            prepared.requested_indices,
            fetch_result,
            source_types,
        )
        result = self._apply_presence(result, prepared.requested_indices)

        metrics = {
            "input_tasks": 1.0,
            "input_rows": float(prepared.table.num_rows),
            "requested_lance_addresses": float(len(requested_addresses)),
            "materializer_seconds": time.perf_counter() - process_started,
            "lance_fetch_seconds": fetch_result.fetch_seconds,
            "lance_fetch_attempts": float(fetch_attempts),
            "lance_fetch_retries": float(max(fetch_attempts - 1, 0)),
            "lance_fetch_timeout_seconds": float(self.fetch_timeout_seconds),
            "lance_fetch_max_retries": float(self.fetch_retries),
            "lance_fetched_bytes": float(sum(fetch_result.fetched_bytes_by_column.values())),
            "lance_read_bytes": float(fetch_result.read_bytes),
            "lance_read_iops": float(fetch_result.read_iops),
            "address_mode.row_id": float(self.address_mode == "row_id"),
            "address_mode.row_address": float(self.address_mode == "row_address"),
        }
        for source, value in fetch_result.fetched_bytes_by_column.items():
            metrics[f"lance_fetched_{source}_bytes"] = float(value)
        self._log_metrics(metrics)
        return InterleavedBatch(
            dataset_name=task.dataset_name,
            data=result,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
