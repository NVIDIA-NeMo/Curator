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

"""Stable-row-id Lance fetch utilities for interleaved image materialization."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar, cast

import pyarrow as pa

if TYPE_CHECKING:
    from .config import LanceTableConfig

ImageFetchMode = Literal["in_process", "subprocess", "subprocess_on_timeout"]
_T = TypeVar("_T")


class _LanceFetchTimeoutError(TimeoutError):
    """Raised when a Lance fetch batch stops making progress."""


def _as_table(value: pa.Table | list[pa.Table]) -> pa.Table:
    if isinstance(value, list):
        if not value:
            return pa.table({})
        return value[0] if len(value) == 1 else pa.concat_tables(value, promote_options="default")
    return value


def _slice_fetched_tables(tables: list[pa.Table], offset: int, length: int) -> list[pa.Table]:
    if length == 0:
        return []
    if offset < 0 or length < 0:
        msg = "offset and length must be non-negative"
        raise ValueError(msg)

    remaining_offset = offset
    remaining_length = length
    sliced: list[pa.Table] = []
    for table in tables:
        if remaining_offset >= table.num_rows:
            remaining_offset -= table.num_rows
            continue
        take = min(table.num_rows - remaining_offset, remaining_length)
        sliced.append(table.slice(remaining_offset, take))
        remaining_length -= take
        remaining_offset = 0
        if remaining_length == 0:
            break
    if remaining_length != 0:
        msg = f"Unable to slice {length} fetched rows from offset {offset}"
        raise RuntimeError(msg)
    return sliced


def _restore_fetched_original_order(tables: list[pa.Table], sorted_original_indices: list[int]) -> list[pa.Table]:
    """Restore rows fetched in sorted-row-id order back to caller-requested order."""
    if not tables or len(sorted_original_indices) <= 1:
        return tables
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")
    if table.num_rows != len(sorted_original_indices):
        msg = f"Lance returned {table.num_rows} rows for {len(sorted_original_indices)} sorted stable row IDs"
        raise RuntimeError(msg)
    sorted_position_by_original = [0] * len(sorted_original_indices)
    for sorted_position, original_position in enumerate(sorted_original_indices):
        sorted_position_by_original[original_position] = sorted_position
    return [table.take(pa.array(sorted_position_by_original, type=pa.int64()))]


@dataclass
class _RowIdFetchResult:
    tables: list[pa.Table]
    fetch_seconds: float
    fetched_bytes_by_column: dict[str, int]
    read_bytes: int = 0
    read_iops: int = 0


class _LanceRowIdFetcher:
    """Worker-local direct Lance stable-row-id image fetcher."""

    def __init__(  # noqa: PLR0913
        self,
        table_config: LanceTableConfig,
        columns: dict[str, str],
        fetch_batch_size: int,
        io_threads: int,
        metadata_cache_size_bytes: int,
        *,
        sort_row_ids_for_fetch: bool,
        fetch_timeout_seconds: float,
        fetch_mode: ImageFetchMode,
    ) -> None:
        import lance

        self.config = table_config
        self.columns = columns
        self.fetch_batch_size = fetch_batch_size
        self.io_threads = io_threads
        self.metadata_cache_size_bytes = metadata_cache_size_bytes
        self.sort_row_ids_for_fetch = sort_row_ids_for_fetch
        self.fetch_timeout_seconds = fetch_timeout_seconds
        self.fetch_mode = fetch_mode
        self.session = lance.Session(metadata_cache_size_bytes=metadata_cache_size_bytes)
        self.dataset = lance.dataset(
            table_config.uri,
            version=table_config.version,
            storage_options=table_config.storage_options or None,
            session=self.session,
        )
        self._validate_dataset()
        if not callable(getattr(self.dataset, "_take_rows", None)):
            msg = "Pinned PyLance build does not expose dataset._take_rows"
            raise TypeError(msg)
        self.executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=io_threads, thread_name_prefix="lance-rowid-fetch"
        )

    @property
    def source_types(self) -> dict[str, pa.DataType]:
        return {source: self.dataset.schema.field(source).type for source in self.columns}

    def close(self, *, wait_for_fetches: bool = True) -> None:
        executor = self.executor
        self.executor = None
        if executor is not None:
            executor.shutdown(wait=wait_for_fetches, cancel_futures=True)
        self.dataset = None
        self.session = None

    def _submit_fetches(self, futures: list[Future[_T]], *, operation: str) -> list[_T]:
        if not futures:
            return []
        deadline = time.monotonic() + self.fetch_timeout_seconds if self.fetch_timeout_seconds > 0 else None
        future_to_index = {future: index for index, future in enumerate(futures)}
        pending = set(futures)
        results: list[_T | None] = [None] * len(futures)

        while pending:
            timeout = None
            if deadline is not None:
                timeout = max(0.0, deadline - time.monotonic())
            done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
            if not done:
                for future in pending:
                    future.cancel()
                msg = (
                    f"Timed out after {self.fetch_timeout_seconds:.1f}s waiting for {operation}; "
                    f"{len(pending)}/{len(futures)} Lance fetch futures are still pending"
                )
                raise _LanceFetchTimeoutError(msg)
            for future in done:
                results[future_to_index[future]] = future.result()

        return cast("list[_T]", results)

    def _validate_dataset(self) -> None:
        missing = sorted(set(self.columns) - set(self.dataset.schema.names))
        if missing:
            msg = f"Requested Lance columns do not exist: {missing}"
            raise ValueError(msg)
        if not self.dataset.has_stable_row_ids:
            msg = "Lance row-id materialization requires stable row IDs"
            raise ValueError(msg)

    def _take_rows(self, row_ids: list[int]) -> list[pa.Table]:
        if self.fetch_mode == "subprocess":
            return [self._take_rows_subprocess(row_ids)]
        if self.executor is None:
            msg = "Lance row-id fetcher is closed"
            raise RuntimeError(msg)
        projected = list(self.columns)
        chunks = [
            row_ids[start : start + self.fetch_batch_size] for start in range(0, len(row_ids), self.fetch_batch_size)
        ]
        futures = [self.executor.submit(self.dataset._take_rows, ids, columns=projected) for ids in chunks]
        return [
            _as_table(table)
            for table in self._submit_fetches(
                futures,
                operation=f"dataset._take_rows chunks={len(chunks)} rows={len(row_ids)}",
            )
        ]

    def _take_rows_subprocess(self, row_ids: list[int]) -> pa.Table:
        timeout = self.fetch_timeout_seconds if self.fetch_timeout_seconds > 0 else None
        tmp_root = Path(os.environ.get("LANCE_MATERIALIZER_SUBPROCESS_DIR") or tempfile.gettempdir())
        with tempfile.TemporaryDirectory(prefix="lance_fetch_", dir=tmp_root) as tmp:
            tmp_path = Path(tmp)
            request_path = tmp_path / "request.json"
            output_path = tmp_path / "output.arrow"
            request_path.write_text(
                json.dumps(
                    {
                        "uri": self.config.uri,
                        "version": self.config.version,
                        "storage_options": self.config.storage_options,
                        "columns": list(self.columns),
                        "row_ids": row_ids,
                        "fetch_batch_size": self.fetch_batch_size,
                        "io_threads": self.io_threads,
                        "metadata_cache_size_bytes": self.metadata_cache_size_bytes,
                        "sort_row_ids_for_fetch": self.sort_row_ids_for_fetch,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            cmd = [
                sys.executable,
                "-m",
                "nemo_curator.stages.interleaved.lance.subprocess_fetch",
                "--request-json",
                str(request_path),
                "--output-arrow",
                str(output_path),
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # noqa: S603
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                stdout, stderr = process.communicate()
                msg = (
                    f"Timed out after {self.fetch_timeout_seconds:.1f}s waiting for subprocess Lance fetch; "
                    f"rows={len(row_ids)} child_pid={process.pid} stdout={stdout!r} stderr={stderr!r}"
                )
                raise _LanceFetchTimeoutError(msg) from exc
            if process.returncode != 0:
                msg = (
                    f"Subprocess Lance fetch failed with return code {process.returncode}; "
                    f"rows={len(row_ids)} stdout={stdout!r} stderr={stderr!r}"
                )
                raise RuntimeError(msg)
            with pa.memory_map(str(output_path), "r") as source:
                return pa.ipc.open_file(source).read_all()

    @staticmethod
    def _io_stats(dataset: object) -> tuple[int, int]:
        io_stats_incremental = getattr(dataset, "io_stats_incremental", None)
        if not callable(io_stats_incremental):
            return 0, 0
        with contextlib.suppress(Exception):
            stats = io_stats_incremental()
            return int(getattr(stats, "read_bytes", 0)), int(getattr(stats, "read_iops", 0))
        return 0, 0

    def fetch(self, row_ids: list[int]) -> _RowIdFetchResult:
        if not row_ids:
            return _RowIdFetchResult([], 0.0, dict.fromkeys(self.columns, 0))

        self._io_stats(self.dataset)
        fetch_started = time.perf_counter()
        if self.sort_row_ids_for_fetch:
            sorted_original_indices = sorted(range(len(row_ids)), key=row_ids.__getitem__)
            sorted_row_ids = [row_ids[index] for index in sorted_original_indices]
            tables = _restore_fetched_original_order(self._take_rows(sorted_row_ids), sorted_original_indices)
        else:
            tables = self._take_rows(row_ids)
        fetch_seconds = time.perf_counter() - fetch_started

        fetched_bytes = dict.fromkeys(self.columns, 0)
        for table in tables:
            for source in self.columns:
                fetched_bytes[source] += table[source].nbytes

        read_bytes, read_iops = self._io_stats(self.dataset)
        return _RowIdFetchResult(
            tables=tables,
            fetch_seconds=fetch_seconds,
            fetched_bytes_by_column=fetched_bytes,
            read_bytes=read_bytes,
            read_iops=read_iops,
        )
