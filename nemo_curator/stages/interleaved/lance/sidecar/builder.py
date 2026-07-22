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

"""Offline builders for sharded Lance URL sidecars."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .format import (
    DEFAULT_SHARDS_DIR,
    ROW_ID_COLUMN,
    ROWADDR_COLUMN,
    SIDECAR_FORMAT,
    SIDECAR_HASH,
    SIDECAR_SCHEMA_VERSION_WITH_ROWADDR,
    UINT64_ENCODING,
    decode_rowaddr,
    encode_uint64,
    hash_url,
    shard_for_digest,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nemo_curator.stages.interleaved.lance.config import LanceTableConfig


def _now() -> float:
    return time.perf_counter()


def _connect_write_shard(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute("PRAGMA page_size=32768")
    conn.execute(
        """
        CREATE TABLE kv (
            url_hash BLOB PRIMARY KEY,
            row_id BLOB NOT NULL,
            rowaddr BLOB NOT NULL
        ) WITHOUT ROWID
        """
    )
    conn.execute(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )
    return conn


def _scanner_for(dataset: Any, *, key_column: str, max_rows: int, batch_size: int) -> Any:  # noqa: ANN401
    scanner_kwargs: dict[str, Any] = {
        "columns": [key_column],
        "with_row_id": True,
        "with_row_address": True,
    }
    if max_rows > 0:
        scanner_kwargs["limit"] = max_rows
    if batch_size > 0:
        scanner_kwargs["batch_size"] = batch_size
    try:
        return dataset.scanner(**scanner_kwargs)
    except TypeError as exc:
        if "batch_size" not in str(exc):
            raise
        scanner_kwargs.pop("batch_size", None)
        return dataset.scanner(**scanner_kwargs)


def _iter_batches(scanner: Any) -> Iterator[Any]:  # noqa: ANN401
    if hasattr(scanner, "to_batches"):
        yield from scanner.to_batches()
        return
    reader = scanner.to_reader()
    while True:
        try:
            yield reader.read_next_batch()
        except StopIteration:
            return


def _flush_shard(conn: sqlite3.Connection, rows: list[tuple[bytes, bytes, bytes]]) -> tuple[int, int]:
    before = conn.total_changes
    conn.executemany("INSERT OR IGNORE INTO kv(url_hash, row_id, rowaddr) VALUES (?, ?, ?)", rows)
    inserted = conn.total_changes - before
    return inserted, len(rows) - inserted


def _flush_pending(
    conns: list[sqlite3.Connection],
    pending: list[list[tuple[bytes, bytes, bytes]]],
    inserted_by_shard: list[int],
    duplicates_by_shard: list[int],
) -> None:
    for shard_id, rows in enumerate(pending):
        if not rows:
            continue
        inserted, duplicates = _flush_shard(conns[shard_id], rows)
        inserted_by_shard[shard_id] += inserted
        duplicates_by_shard[shard_id] += duplicates
        rows.clear()


def _commit_all(conns: list[sqlite3.Connection]) -> None:
    for conn in conns:
        conn.commit()


def _write_progress(  # noqa: PLR0913
    path: Path,
    *,
    total_rows: int,
    valid_url_rows: int,
    inserted_by_shard: list[int],
    duplicates_by_shard: list[int],
    started: float,
) -> None:
    elapsed = _now() - started
    inserted = sum(inserted_by_shard)
    payload = {
        "elapsed_s": elapsed,
        "total_rows": total_rows,
        "valid_url_rows": valid_url_rows,
        "inserted_rows": inserted,
        "duplicate_hash_rows": sum(duplicates_by_shard),
        "valid_url_rows_per_s": valid_url_rows / elapsed if elapsed else None,
        "inserted_rows_per_s": inserted / elapsed if elapsed else None,
        "updated_at_unix_s": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_sharded_sqlite_url_lance_sidecar(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    dataset: LanceTableConfig,
    output_dir: str | Path,
    key_column: str = "url",
    shard_count: int = 512,
    max_rows: int = 0,
    batch_size: int = 8192,
    insert_batch_rows: int = 8192,
    commit_every_rows: int = 5_000_000,
    progress_every_rows: int = 1_000_000,
    sample_url_count: int = 32_768,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a compact sharded URL -> Lance row-id/row-address sidecar."""

    if not key_column:
        msg = "key_column must not be empty"
        raise ValueError(msg)
    for name, value in {
        "shard_count": shard_count,
        "batch_size": batch_size,
        "insert_batch_rows": insert_batch_rows,
        "commit_every_rows": commit_every_rows,
        "progress_every_rows": progress_every_rows,
    }.items():
        if value <= 0:
            msg = f"{name} must be greater than 0"
            raise ValueError(msg)
    if max_rows < 0 or sample_url_count < 0:
        msg = "max_rows and sample_url_count must be non-negative"
        raise ValueError(msg)

    import lance

    output_path = Path(output_dir)
    if output_path.exists() and not overwrite:
        msg = f"output directory already exists: {output_path}"
        raise FileExistsError(msg)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    if output_path.exists():
        shutil.rmtree(output_path)
    shards_dir = temporary / DEFAULT_SHARDS_DIR
    shards_dir.mkdir(parents=True)

    open_started = _now()
    lance_dataset = lance.dataset(
        dataset.uri,
        version=dataset.version,
        storage_options=dataset.storage_options or None,
    )
    open_seconds = _now() - open_started
    if key_column not in lance_dataset.schema.names:
        msg = f"Lance dataset is missing key column {key_column!r}; schema={lance_dataset.schema.names}"
        raise ValueError(msg)

    conns: list[sqlite3.Connection] = []
    try:
        for shard_id in range(shard_count):
            conns.append(_connect_write_shard(shards_dir / f"shard-{shard_id:05d}.sqlite"))
        pending: list[list[tuple[bytes, bytes, bytes]]] = [[] for _ in range(shard_count)]
        inserted_by_shard = [0] * shard_count
        duplicates_by_shard = [0] * shard_count
        total_rows = 0
        valid_url_rows = 0
        sample_urls_written = 0
        progress_path = temporary / "progress.json"
        sample_path = temporary / "sample_urls.jsonl"
        scan_started = _now()
        next_commit_rows = commit_every_rows
        next_progress_rows = progress_every_rows

        with sample_path.open("w", encoding="utf-8") as sample_file:
            for batch in _iter_batches(
                _scanner_for(lance_dataset, key_column=key_column, max_rows=max_rows, batch_size=batch_size)
            ):
                batch_dict = batch.to_pydict()
                urls = batch_dict[key_column]
                row_ids = batch_dict[ROW_ID_COLUMN]
                rowaddrs = batch_dict[ROWADDR_COLUMN]
                total_rows += len(urls)
                for url, row_id, rowaddr in zip(urls, row_ids, rowaddrs, strict=True):
                    if not isinstance(url, str) or not url:
                        continue
                    valid_url_rows += 1
                    rowaddr_int = int(rowaddr)
                    digest = hash_url(url)
                    shard_id = shard_for_digest(digest, shard_count)
                    pending[shard_id].append((digest, encode_uint64(row_id), encode_uint64(rowaddr_int)))
                    if sample_urls_written < sample_url_count:
                        fragment_id, row_offset = decode_rowaddr(rowaddr_int)
                        sample_file.write(
                            json.dumps(
                                {
                                    "url": url,
                                    "row_id": int(row_id),
                                    "rowaddr": rowaddr_int,
                                    "fragment_id": fragment_id,
                                    "row_offset": row_offset,
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        sample_urls_written += 1
                    if len(pending[shard_id]) >= insert_batch_rows:
                        inserted, duplicates = _flush_shard(conns[shard_id], pending[shard_id])
                        inserted_by_shard[shard_id] += inserted
                        duplicates_by_shard[shard_id] += duplicates
                        pending[shard_id].clear()
                if total_rows >= next_commit_rows:
                    _flush_pending(conns, pending, inserted_by_shard, duplicates_by_shard)
                    _commit_all(conns)
                    _write_progress(
                        progress_path,
                        total_rows=total_rows,
                        valid_url_rows=valid_url_rows,
                        inserted_by_shard=inserted_by_shard,
                        duplicates_by_shard=duplicates_by_shard,
                        started=scan_started,
                    )
                    while next_commit_rows <= total_rows:
                        next_commit_rows += commit_every_rows
                    while next_progress_rows <= total_rows:
                        next_progress_rows += progress_every_rows
                elif total_rows >= next_progress_rows:
                    _flush_pending(conns, pending, inserted_by_shard, duplicates_by_shard)
                    _write_progress(
                        progress_path,
                        total_rows=total_rows,
                        valid_url_rows=valid_url_rows,
                        inserted_by_shard=inserted_by_shard,
                        duplicates_by_shard=duplicates_by_shard,
                        started=scan_started,
                    )
                    while next_progress_rows <= total_rows:
                        next_progress_rows += progress_every_rows

        _flush_pending(conns, pending, inserted_by_shard, duplicates_by_shard)
        _commit_all(conns)
        _write_progress(
            progress_path,
            total_rows=total_rows,
            valid_url_rows=valid_url_rows,
            inserted_by_shard=inserted_by_shard,
            duplicates_by_shard=duplicates_by_shard,
            started=scan_started,
        )
        for shard_id, conn in enumerate(conns):
            conn.execute("INSERT INTO metadata VALUES (?, ?)", ("shard_id", json.dumps(shard_id)))
            conn.execute("INSERT INTO metadata VALUES (?, ?)", ("shard_count", json.dumps(shard_count)))
            conn.execute("INSERT INTO metadata VALUES (?, ?)", ("hash", json.dumps(SIDECAR_HASH)))
            conn.execute("INSERT INTO metadata VALUES (?, ?)", ("row_id_encoding", json.dumps(UINT64_ENCODING)))
            conn.execute("INSERT INTO metadata VALUES (?, ?)", ("rowaddr_encoding", json.dumps(UINT64_ENCODING)))
            conn.execute(
                "INSERT INTO metadata VALUES (?, ?)",
                ("sidecar_schema_version", json.dumps(SIDECAR_SCHEMA_VERSION_WITH_ROWADDR)),
            )
            conn.commit()
            conn.execute("PRAGMA optimize")
    finally:
        for conn in conns:
            conn.close()

    scan_seconds = _now() - scan_started
    shard_bytes = {path.name: path.stat().st_size for path in sorted(shards_dir.glob("*.sqlite"))}
    inserted_rows = sum(inserted_by_shard)
    duplicate_hash_rows = sum(duplicates_by_shard)
    manifest = {
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION_WITH_ROWADDR,
        "format": SIDECAR_FORMAT,
        "hash": SIDECAR_HASH,
        "shard_count": shard_count,
        "shard_strategy": "uint64_be_prefix_mod_shard_count",
        "row_id_encoding": UINT64_ENCODING,
        "rowaddr_encoding": UINT64_ENCODING,
        "rowaddr_layout": "uint64: fragment_id = rowaddr >> 32, row_offset = rowaddr & 0xffffffff",
        "coordinate_columns": ["row_id", "rowaddr"],
        "image_lance_uri": dataset.uri,
        "image_lance_version": lance_dataset.version,
        "key_column": key_column,
        "row_count": inserted_rows,
        "duplicate_hash_rows": duplicate_hash_rows,
        "shards_dir": DEFAULT_SHARDS_DIR,
        "sample_urls": "sample_urls.jsonl",
    }
    report = {
        "config": {
            "image_lance_uri": dataset.uri,
            "image_lance_version": dataset.version,
            "key_column": key_column,
            "shard_count": shard_count,
            "max_rows": max_rows,
            "batch_size": batch_size,
            "insert_batch_rows": insert_batch_rows,
            "commit_every_rows": commit_every_rows,
            "progress_every_rows": progress_every_rows,
            "sample_url_count": sample_url_count,
        },
        "open_seconds": open_seconds,
        "scan_and_write_seconds": scan_seconds,
        "total_rows": total_rows,
        "valid_url_rows": valid_url_rows,
        "inserted_rows": inserted_rows,
        "duplicate_hash_rows": duplicate_hash_rows,
        "sample_urls_written": sample_urls_written,
        "valid_url_rows_per_s": valid_url_rows / scan_seconds if scan_seconds else None,
        "shard_bytes_total": sum(shard_bytes.values()),
        "shard_bytes": shard_bytes,
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (temporary / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.rename(temporary, output_path)
    return report
