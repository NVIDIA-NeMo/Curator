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

"""Curator stage for resolving interleaved image URLs to Lance addresses."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import InterleavedBatch

from .format import (
    chunked,
    connect_readonly_sidecar_shard,
    decode_rowaddr,
    decode_uint64,
    hash_url,
    manifest_has_rowaddr,
    read_lance_url_sidecar_manifest,
    shard_for_digest,
)

if TYPE_CHECKING:
    import sqlite3


@dataclass(frozen=True)
class LanceUrlSidecarCoordinate:
    """Coordinates for one URL in a Lance image table."""

    row_id: int
    fragment_id: int | None = None
    row_offset: int | None = None


@dataclass(frozen=True)
class _PreparedUrlTask:
    task: InterleavedBatch
    table: pa.Table
    keys: list[str | None]


@dataclass(frozen=True)
class _ResolutionColumns:
    row_ids: list[int | None]
    fragment_ids: list[int | None]
    row_offsets: list[int | None]
    presence_values: list[bool | None]
    error_values: list[str | None]
    found_rows: int
    missing_rows: int


def _now() -> float:
    return time.perf_counter()


def _set_or_append_column(table: pa.Table, name: str, array: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index >= 0:
        return table.set_column(index, name, array)
    return table.append_column(name, array)


@dataclass
class ShardedSqliteUrlLanceAddressResolutionStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Resolve interleaved image URL rows to Lance row-id and row-address columns."""

    sidecar_dir: str
    input_url_column: str = "source_ref"
    output_row_id_column: str | None = "lance_row_id"
    output_fragment_id_column: str | None = "lance_fragment_id"
    output_row_offset_column: str | None = "lance_row_offset"
    presence_column: str | None = "lance_address_present"
    error_column: str | None = "lance_lookup_error"
    query_batch_size: int = 512
    cache_mib: int = 128
    mmap_mib: int = 1024
    name: str = "sharded_sqlite_url_lance_address_resolution"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    _manifest: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _conns: dict[int, sqlite3.Connection] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.input_url_column:
            msg = "input_url_column must not be empty"
            raise ValueError(msg)
        for param_name, value in {
            "query_batch_size": self.query_batch_size,
            "cache_mib": self.cache_mib,
            "mmap_mib": self.mmap_mib,
        }.items():
            if value <= 0:
                msg = f"{param_name} must be greater than 0"
                raise ValueError(msg)
        output_columns = [
            column
            for column in (
                self.output_row_id_column,
                self.output_fragment_id_column,
                self.output_row_offset_column,
                self.presence_column,
                self.error_column,
            )
            if column is not None
        ]
        if not output_columns:
            msg = "at least one output column must be configured"
            raise ValueError(msg)
        if any(column == "" for column in output_columns):
            msg = "sidecar output columns must be non-empty when provided"
            raise ValueError(msg)
        if len(set(output_columns)) != len(output_columns):
            msg = f"sidecar output columns must be distinct: {output_columns}"
            raise ValueError(msg)
        if (self.output_fragment_id_column is None) != (self.output_row_offset_column is None):
            msg = "output_fragment_id_column and output_row_offset_column must be configured together"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.input_url_column]

    def outputs(self) -> tuple[list[str], list[str]]:
        columns = [
            column
            for column in (
                self.output_row_id_column,
                self.output_fragment_id_column,
                self.output_row_offset_column,
                self.presence_column,
                self.error_column,
            )
            if column is not None
        ]
        return ["data"], columns

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            self._manifest = read_lance_url_sidecar_manifest(self.sidecar_dir)
            if self.output_fragment_id_column is not None and not manifest_has_rowaddr(self._manifest):
                msg = (
                    "sidecar does not contain Lance row addresses; disable row-address outputs or rebuild the sidecar"
                )
                raise ValueError(msg)
        return self._manifest

    @property
    def shard_count(self) -> int:
        return int(self.manifest["shard_count"])

    def teardown(self) -> None:
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()

    def _conn_for_shard(self, shard_id: int) -> sqlite3.Connection:
        conn = self._conns.get(shard_id)
        if conn is not None:
            return conn
        shard_path = Path(self.sidecar_dir) / self.manifest["shards_dir"] / f"shard-{shard_id:05d}.sqlite"
        conn = connect_readonly_sidecar_shard(shard_path, cache_mib=self.cache_mib, mmap_mib=self.mmap_mib)
        self._conns[shard_id] = conn
        return conn

    def _lookup(self, urls: list[str]) -> tuple[dict[str, LanceUrlSidecarCoordinate], float]:
        started = _now()
        coordinates_by_url: dict[str, LanceUrlSidecarCoordinate] = {}
        urls_by_shard: dict[int, list[tuple[str, bytes]]] = {}
        for url in urls:
            digest = hash_url(url)
            urls_by_shard.setdefault(shard_for_digest(digest, self.shard_count), []).append((url, digest))

        select_rowaddr = self.output_fragment_id_column is not None
        for shard_id, shard_items in urls_by_shard.items():
            conn = self._conn_for_shard(shard_id)
            for item_chunk in chunked(shard_items, self.query_batch_size):
                digests = [digest for _, digest in item_chunk]
                placeholders = ",".join("?" for _ in digests)
                if select_rowaddr:
                    rows = conn.execute(
                        f"SELECT url_hash, row_id, rowaddr FROM kv WHERE url_hash IN ({placeholders})",  # noqa: S608
                        digests,
                    ).fetchall()
                    coordinates_by_hash = {}
                    for row in rows:
                        rowaddr = decode_uint64(row[2])
                        fragment_id, row_offset = decode_rowaddr(rowaddr)
                        coordinates_by_hash[row[0]] = LanceUrlSidecarCoordinate(
                            row_id=decode_uint64(row[1]),
                            fragment_id=fragment_id,
                            row_offset=row_offset,
                        )
                else:
                    rows = conn.execute(
                        f"SELECT url_hash, row_id FROM kv WHERE url_hash IN ({placeholders})",  # noqa: S608
                        digests,
                    ).fetchall()
                    coordinates_by_hash = {
                        row[0]: LanceUrlSidecarCoordinate(row_id=decode_uint64(row[1])) for row in rows
                    }
                for url, digest in item_chunk:
                    coordinate = coordinates_by_hash.get(digest)
                    if coordinate is not None:
                        coordinates_by_url[url] = coordinate
        return coordinates_by_url, _now() - started

    def _prepare_tasks(self, tasks: list[InterleavedBatch]) -> tuple[list[_PreparedUrlTask], list[str], int, int]:
        prepared: list[_PreparedUrlTask] = []
        unique_urls: list[str] = []
        seen: set[str] = set()
        input_rows = 0
        rows_without_urls = 0
        for task in tasks:
            table = task.to_pyarrow()
            if self.input_url_column not in table.column_names:
                msg = f"Input URL column {self.input_url_column!r} does not exist"
                raise ValueError(msg)
            values = table[self.input_url_column].combine_chunks().to_pylist()
            input_rows += table.num_rows
            keys: list[str | None] = []
            for value in values:
                if isinstance(value, str) and value:
                    keys.append(value)
                    if value not in seen:
                        seen.add(value)
                        unique_urls.append(value)
                else:
                    keys.append(None)
                    rows_without_urls += 1
            prepared.append(_PreparedUrlTask(task=task, table=table, keys=keys))
        return prepared, unique_urls, input_rows, rows_without_urls

    def _resolve_keys(
        self,
        keys: list[str | None],
        coordinates_by_url: dict[str, LanceUrlSidecarCoordinate],
    ) -> _ResolutionColumns:
        row_ids: list[int | None] = []
        fragment_ids: list[int | None] = []
        row_offsets: list[int | None] = []
        presence_values: list[bool | None] = []
        error_values: list[str | None] = []
        found_rows = 0
        missing_rows = 0
        for key in keys:
            if key is None:
                row_ids.append(None)
                fragment_ids.append(None)
                row_offsets.append(None)
                presence_values.append(False)
                error_values.append("missing_url")
                continue
            coordinate = coordinates_by_url.get(key)
            if coordinate is None:
                missing_rows += 1
                row_ids.append(None)
                fragment_ids.append(None)
                row_offsets.append(None)
                presence_values.append(False)
                error_values.append("not_found_in_sidecar")
                continue
            found_rows += 1
            row_ids.append(coordinate.row_id)
            fragment_ids.append(coordinate.fragment_id)
            row_offsets.append(coordinate.row_offset)
            presence_values.append(True)
            error_values.append(None)
        return _ResolutionColumns(
            row_ids=row_ids,
            fragment_ids=fragment_ids,
            row_offsets=row_offsets,
            presence_values=presence_values,
            error_values=error_values,
            found_rows=found_rows,
            missing_rows=missing_rows,
        )

    def _append_resolution_columns(self, table: pa.Table, columns: _ResolutionColumns) -> pa.Table:
        result = table
        if self.output_row_id_column is not None:
            result = _set_or_append_column(
                result,
                self.output_row_id_column,
                pa.array(columns.row_ids, type=pa.uint64(), from_pandas=True),
            )
        if self.output_fragment_id_column is not None and self.output_row_offset_column is not None:
            result = _set_or_append_column(
                result,
                self.output_fragment_id_column,
                pa.array(columns.fragment_ids, type=pa.uint32(), from_pandas=True),
            )
            result = _set_or_append_column(
                result,
                self.output_row_offset_column,
                pa.array(columns.row_offsets, type=pa.uint32(), from_pandas=True),
            )
        if self.presence_column is not None:
            result = _set_or_append_column(
                result, self.presence_column, pa.array(columns.presence_values, type=pa.bool_())
            )
        if self.error_column is not None:
            result = _set_or_append_column(
                result,
                self.error_column,
                pa.array(columns.error_values, type=pa.string()),
            )
        return result

    def _process_tasks(self, tasks: list[InterleavedBatch]) -> list[InterleavedBatch]:
        prepared, unique_urls, input_rows, rows_without_urls = self._prepare_tasks(tasks)
        coordinates_by_url, lookup_seconds = self._lookup(unique_urls)
        outputs: list[InterleavedBatch] = []
        found_rows = 0
        missing_rows = 0
        for prepared_task in prepared:
            columns = self._resolve_keys(prepared_task.keys, coordinates_by_url)
            found_rows += columns.found_rows
            missing_rows += columns.missing_rows
            result = self._append_resolution_columns(prepared_task.table, columns)
            outputs.append(
                InterleavedBatch(
                    dataset_name=prepared_task.task.dataset_name,
                    data=result,
                    _metadata=prepared_task.task._metadata,
                    _stage_perf=prepared_task.task._stage_perf,
                )
            )

        self._log_metrics(
            {
                "input_tasks": float(len(tasks)),
                "input_rows": float(input_rows),
                "requested_unique_urls": float(len(unique_urls)),
                "found_unique_urls": float(len(coordinates_by_url)),
                "found_rows": float(found_rows),
                "missing_rows": float(missing_rows),
                "rows_without_urls": float(rows_without_urls),
                "opened_shards": float(len(self._conns)),
                "lookup_seconds": lookup_seconds,
            }
        )
        return outputs

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        return self._process_tasks([task])[0]

    def process_batch(self, tasks: list[InterleavedBatch]) -> list[InterleavedBatch]:
        return self._process_tasks(tasks)
