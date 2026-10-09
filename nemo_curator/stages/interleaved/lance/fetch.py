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

"""Lance fetch utilities for interleaved materialization."""

from __future__ import annotations

import contextlib
import time
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Literal, TypeVar

import pyarrow as pa

_T = TypeVar("_T")
_LanceAddressMode = Literal["row_id", "row_address"]


class _LanceFetchTimeoutError(TimeoutError):
    """Raised when a Lance fetch batch exceeds its configured deadline."""


@dataclass(frozen=True)
class _LanceRowAddress:
    fragment_id: int
    row_offset: int


@dataclass(frozen=True)
class _LanceFragmentTakeOperation:
    fragment_id: int
    row_offsets: list[int]
    original_indices: list[int]


def _as_table(value: pa.Table | list[pa.Table]) -> pa.Table:
    if isinstance(value, list):
        if not value:
            return pa.table({})
        return value[0] if len(value) == 1 else pa.concat_tables(value, promote_options="default")
    return value


def _restore_fetched_original_order(table: pa.Table, sorted_original_indices: list[int]) -> pa.Table:
    """Restore rows fetched in optimized order back to caller-requested order."""
    if len(sorted_original_indices) <= 1:
        return table
    if table.num_rows != len(sorted_original_indices):
        msg = f"Lance returned {table.num_rows} rows for {len(sorted_original_indices)} sorted addresses"
        raise RuntimeError(msg)
    sorted_position_by_original = [0] * len(sorted_original_indices)
    for sorted_position, original_position in enumerate(sorted_original_indices):
        sorted_position_by_original[original_position] = sorted_position
    return table.take(pa.array(sorted_position_by_original, type=pa.int64()))


@dataclass
class _LanceFetchResult:
    table: pa.Table
    fetch_seconds: float
    fetched_bytes_by_column: dict[str, int]
    read_bytes: int = 0
    read_iops: int = 0


def _group_row_addresses_by_fragment(
    addresses: list[_LanceRowAddress],
    *,
    fetch_batch_size: int,
) -> list[_LanceFragmentTakeOperation]:
    grouped: dict[int, list[tuple[int, int]]] = {}
    for original_index, address in enumerate(addresses):
        grouped.setdefault(address.fragment_id, []).append((original_index, address.row_offset))

    operations: list[_LanceFragmentTakeOperation] = []
    for fragment_id in sorted(grouped):
        rows = sorted(grouped[fragment_id], key=lambda item: item[1])
        for start in range(0, len(rows), fetch_batch_size):
            chunk = rows[start : start + fetch_batch_size]
            operations.append(
                _LanceFragmentTakeOperation(
                    fragment_id=fragment_id,
                    row_offsets=[row_offset for _, row_offset in chunk],
                    original_indices=[original_index for original_index, _ in chunk],
                )
            )
    return operations


_LanceAddress = int | _LanceRowAddress


class _LanceFetcher:
    """Worker-local Lance coordinate fetcher."""

    def __init__(  # noqa: PLR0913
        self,
        path: str,
        version: int | None,
        storage_options: dict[str, str],
        columns: dict[str, str],
        fetch_batch_size: int,
        io_threads: int,
        metadata_cache_size_bytes: int,
        *,
        address_mode: _LanceAddressMode,
        sort_row_ids_for_fetch: bool,
        fetch_timeout_seconds: float,
    ) -> None:
        import lance

        self.path = path
        self.version = version
        self.storage_options = storage_options
        self.columns = columns
        self.fetch_batch_size = fetch_batch_size
        self.io_threads = io_threads
        self.metadata_cache_size_bytes = metadata_cache_size_bytes
        self.address_mode = address_mode
        self.sort_row_ids_for_fetch = sort_row_ids_for_fetch
        self.fetch_timeout_seconds = fetch_timeout_seconds
        self.session = lance.Session(metadata_cache_size_bytes=metadata_cache_size_bytes)
        self.dataset = lance.dataset(
            path,
            version=version,
            storage_options=storage_options or None,
            session=self.session,
        )
        self._validate_columns()
        self._validate_address_mode()
        self.fragments = (
            {int(fragment.fragment_id): fragment for fragment in self.dataset.get_fragments()}
            if address_mode == "row_address"
            else {}
        )
        self.executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=io_threads, thread_name_prefix="lance-fetch"
        )

    @property
    def source_types(self) -> dict[str, pa.DataType]:
        return {source: self.dataset.schema.field(source).type for source in self.columns}

    def _empty_table(self) -> pa.Table:
        return pa.table({source: pa.array([], type=data_type) for source, data_type in self.source_types.items()})

    def close(self, *, wait_for_fetches: bool = True) -> None:
        executor = self.executor
        self.executor = None
        if executor is not None:
            executor.shutdown(wait=wait_for_fetches, cancel_futures=True)
        self.dataset = None
        self.session = None

    def _validate_columns(self) -> None:
        missing = sorted(set(self.columns) - set(self.dataset.schema.names))
        if missing:
            msg = f"Requested Lance columns do not exist: {missing}"
            raise ValueError(msg)
        blob_columns = [
            source
            for source in self.columns
            if getattr(self.dataset.schema.field(source).type, "extension_name", None) == "lance.blob.v2"
        ]
        if blob_columns:
            msg = f"Lance Blob v2 columns are not supported for coordinate fetches: {blob_columns}"
            raise TypeError(msg)

    def _validate_address_mode(self) -> None:
        if self.address_mode == "row_id":
            if not self.dataset.has_stable_row_ids:
                msg = "Lance row-id materialization requires stable row IDs"
                raise ValueError(msg)
            if not callable(getattr(self.dataset, "_take_rows", None)):
                msg = "Pinned PyLance build does not expose dataset._take_rows"
                raise TypeError(msg)

    def _submit_fetches(self, futures: list[Future[_T]], *, operation: str) -> list[_T]:
        if not futures:
            return []
        timeout = self.fetch_timeout_seconds if self.fetch_timeout_seconds > 0 else None
        done, pending = wait(futures, timeout=timeout, return_when=FIRST_EXCEPTION)
        failed = next((future for future in done if future.exception() is not None), None)
        if failed is not None:
            for future in pending:
                future.cancel()
            failed.result()
        if pending:
            for future in pending:
                future.cancel()
            msg = (
                f"Timed out after {self.fetch_timeout_seconds:.1f}s waiting for {operation}; "
                f"{len(pending)}/{len(futures)} Lance fetch futures are still pending"
            )
            raise _LanceFetchTimeoutError(msg)
        return [future.result() for future in futures]

    @staticmethod
    def _io_stats(dataset: object) -> tuple[int, int]:
        io_stats_incremental = getattr(dataset, "io_stats_incremental", None)
        if not callable(io_stats_incremental):
            return 0, 0
        with contextlib.suppress(Exception):
            stats = io_stats_incremental()
            return int(getattr(stats, "read_bytes", 0)), int(getattr(stats, "read_iops", 0))
        return 0, 0

    def _fetch_result(self, table: pa.Table, fetch_seconds: float) -> _LanceFetchResult:
        fetched_bytes = {source: table[source].nbytes for source in self.columns}

        read_bytes, read_iops = self._io_stats(self.dataset)
        return _LanceFetchResult(
            table=table,
            fetch_seconds=fetch_seconds,
            fetched_bytes_by_column=fetched_bytes,
            read_bytes=read_bytes,
            read_iops=read_iops,
        )

    def _take_rows(self, row_ids: list[int]) -> pa.Table:
        if self.executor is None:
            msg = "Lance row-id fetcher is closed"
            raise RuntimeError(msg)
        projected = list(self.columns)
        chunks = [
            row_ids[start : start + self.fetch_batch_size] for start in range(0, len(row_ids), self.fetch_batch_size)
        ]
        futures = [self.executor.submit(self.dataset._take_rows, ids, columns=projected) for ids in chunks]
        tables = [
            _as_table(table)
            for table in self._submit_fetches(
                futures,
                operation=f"dataset._take_rows chunks={len(chunks)} rows={len(row_ids)}",
            )
        ]
        return _as_table(tables)

    def _fetch_row_ids(self, row_ids: list[int]) -> pa.Table:
        if self.sort_row_ids_for_fetch:
            sorted_original_indices = sorted(range(len(row_ids)), key=row_ids.__getitem__)
            sorted_row_ids = [row_ids[index] for index in sorted_original_indices]
            return _restore_fetched_original_order(self._take_rows(sorted_row_ids), sorted_original_indices)
        return self._take_rows(row_ids)

    def _take_row_addresses(self, addresses: list[_LanceRowAddress]) -> pa.Table:
        if self.executor is None:
            msg = "Lance row-address fetcher is closed"
            raise RuntimeError(msg)

        operations = _group_row_addresses_by_fragment(addresses, fetch_batch_size=self.fetch_batch_size)
        futures = [self.executor.submit(self._take_fragment_rows, operation) for operation in operations]
        fetched = self._submit_fetches(
            futures,
            operation=f"fragment.take ops={len(operations)} rows={len(addresses)}",
        )
        tables = [table for table, _ in fetched]
        original_indices = [index for _, indices in fetched for index in indices]
        table = _as_table(tables)
        return _restore_fetched_original_order(table, original_indices)

    def _take_fragment_rows(self, operation: _LanceFragmentTakeOperation) -> tuple[pa.Table, list[int]]:
        fragment = self.fragments.get(operation.fragment_id)
        if fragment is None:
            msg = f"Lance fragment {operation.fragment_id} does not exist"
            raise ValueError(msg)
        table = _as_table(fragment.take(operation.row_offsets, columns=list(self.columns)))
        return table, operation.original_indices

    def fetch(self, addresses: list[_LanceAddress]) -> _LanceFetchResult:
        if not addresses:
            return _LanceFetchResult(self._empty_table(), 0.0, dict.fromkeys(self.columns, 0))

        self._io_stats(self.dataset)
        fetch_started = time.perf_counter()
        if self.address_mode == "row_id":
            row_ids = [address for address in addresses if isinstance(address, int)]
            if len(row_ids) != len(addresses):
                msg = "row_id mode requires integer addresses"
                raise TypeError(msg)
            table = self._fetch_row_ids(row_ids)
        else:
            row_addresses = [address for address in addresses if isinstance(address, _LanceRowAddress)]
            if len(row_addresses) != len(addresses):
                msg = "row_address mode requires fragment and row-offset addresses"
                raise TypeError(msg)
            table = self._take_row_addresses(row_addresses)
        fetch_seconds = time.perf_counter() - fetch_started
        return self._fetch_result(table, fetch_seconds)
