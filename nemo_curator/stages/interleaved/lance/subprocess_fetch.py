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

"""Subprocess worker for isolated Lance image fetches."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa


def _as_table(value: pa.Table | list[pa.Table]) -> pa.Table:
    if isinstance(value, list):
        if not value:
            return pa.table({})
        return value[0] if len(value) == 1 else pa.concat_tables(value, promote_options="default")
    return value


def _restore_original_order(table: pa.Table, sorted_original_indices: list[int]) -> pa.Table:
    if len(sorted_original_indices) <= 1:
        return table
    if table.num_rows != len(sorted_original_indices):
        msg = f"Lance returned {table.num_rows} rows for {len(sorted_original_indices)} sorted image addresses"
        raise RuntimeError(msg)
    sorted_position_by_original = [0] * len(sorted_original_indices)
    for sorted_position, original_position in enumerate(sorted_original_indices):
        sorted_position_by_original[original_position] = sorted_position
    return table.take(pa.array(sorted_position_by_original, type=pa.int64()))


@dataclass(frozen=True)
class _LanceRowAddress:
    fragment_id: int
    row_offset: int


@dataclass(frozen=True)
class _LanceFragmentTakeOperation:
    fragment_id: int
    row_offsets: list[int]
    original_indices: list[int]


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


def _open_dataset(request: dict[str, Any]) -> object:
    import lance

    session = lance.Session(metadata_cache_size_bytes=int(request["metadata_cache_size_bytes"]))
    return lance.dataset(
        request["uri"],
        version=request.get("version"),
        storage_options=request.get("storage_options") or None,
        session=session,
    )


def _fetch_row_ids(request: dict[str, Any]) -> pa.Table:
    row_ids = [int(row_id) for row_id in request["row_ids"]]
    columns = list(request["columns"])
    fetch_batch_size = int(request["fetch_batch_size"])
    io_threads = int(request["io_threads"])
    sort_row_ids_for_fetch = bool(request["sort_row_ids_for_fetch"])

    dataset = _open_dataset(request)

    sorted_original_indices: list[int] | None = None
    if sort_row_ids_for_fetch:
        sorted_original_indices = sorted(range(len(row_ids)), key=row_ids.__getitem__)
        row_ids = [row_ids[index] for index in sorted_original_indices]

    chunks = [row_ids[start : start + fetch_batch_size] for start in range(0, len(row_ids), fetch_batch_size)]
    if len(chunks) == 1:
        table = _as_table(dataset._take_rows(chunks[0], columns=columns))
    else:
        with ThreadPoolExecutor(max_workers=io_threads, thread_name_prefix="lance-subprocess-fetch") as executor:
            tables = [
                _as_table(table)
                for table in executor.map(lambda ids: dataset._take_rows(ids, columns=columns), chunks)
            ]
        table = _as_table(tables)

    if sorted_original_indices is not None:
        table = _restore_original_order(table, sorted_original_indices)
    return table


def _fetch_row_addresses(request: dict[str, Any]) -> pa.Table:
    addresses = [
        _LanceRowAddress(
            fragment_id=int(address["fragment_id"]),
            row_offset=int(address["row_offset"]),
        )
        for address in request["row_addresses"]
    ]
    columns = list(request["columns"])
    fetch_batch_size = int(request["fetch_batch_size"])
    io_threads = int(request["io_threads"])

    dataset = _open_dataset(request)
    fragments = {int(fragment.fragment_id): fragment for fragment in dataset.get_fragments()}
    operations = _group_row_addresses_by_fragment(addresses, fetch_batch_size=fetch_batch_size)

    def take_fragment_rows(operation: _LanceFragmentTakeOperation) -> tuple[pa.Table, list[int]]:
        fragment = fragments.get(operation.fragment_id)
        if fragment is None:
            msg = f"Lance fragment {operation.fragment_id} does not exist"
            raise ValueError(msg)
        return _as_table(fragment.take(operation.row_offsets, columns=columns)), operation.original_indices

    if len(operations) == 1:
        fetched = [take_fragment_rows(operations[0])]
    else:
        with ThreadPoolExecutor(max_workers=io_threads, thread_name_prefix="lance-subprocess-fetch") as executor:
            fetched = list(executor.map(take_fragment_rows, operations))
    table = _as_table([table for table, _ in fetched])
    original_indices = [index for _, indices in fetched for index in indices]
    return _restore_original_order(table, original_indices)


def _fetch_rows(request: dict[str, Any]) -> pa.Table:
    if "row_addresses" in request:
        return _fetch_row_addresses(request)
    return _fetch_row_ids(request)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--output-arrow", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
    started = time.perf_counter()
    table = _fetch_rows(request)
    output = Path(args.output_arrow)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(output), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    print(
        json.dumps(
            {
                "status": "ok",
                "rows": table.num_rows,
                "nbytes": table.nbytes,
                "elapsed_s": time.perf_counter() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
