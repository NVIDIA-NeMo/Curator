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

"""Shared on-disk format helpers for Lance URL sidecars."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, TypeVar

HASH_BYTES = 16
UINT64_BYTES = 8
SIDECAR_FORMAT = "sqlite-sharded-compact"
SIDECAR_HASH = "blake2b-128"
UINT64_ENCODING = "uint64-little-endian"
DEFAULT_SHARDS_DIR = "shards"
ROW_ID_COLUMN = "_rowid"
ROWADDR_COLUMN = "_rowaddr"
SIDECAR_SCHEMA_VERSION_WITH_ROWADDR = 2

T = TypeVar("T")


def hash_url(url: str) -> bytes:
    """Return the compact sidecar key for a URL."""

    return hashlib.blake2b(url.encode("utf-8"), digest_size=HASH_BYTES).digest()


def shard_for_digest(digest: bytes, shard_count: int) -> int:
    """Map a URL digest to its SQLite shard."""

    return int.from_bytes(digest[:8], byteorder="big", signed=False) % shard_count


def encode_uint64(value: object) -> bytes:
    return int(value).to_bytes(UINT64_BYTES, byteorder="little", signed=False)


def decode_uint64(value: bytes) -> int:
    if len(value) != UINT64_BYTES:
        msg = f"expected {UINT64_BYTES} bytes for uint64, got {len(value)}"
        raise ValueError(msg)
    return int.from_bytes(value, byteorder="little", signed=False)


def decode_rowaddr(rowaddr: int) -> tuple[int, int]:
    """Decode Lance's packed row address into fragment id and row offset."""

    return rowaddr >> 32, rowaddr & 0xFFFFFFFF


def chunked(values: list[T], size: int) -> list[list[T]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def connect_readonly_sidecar_shard(path: Path, *, cache_mib: int, mmap_mib: int) -> sqlite3.Connection:
    """Open a sidecar shard in read-only immutable mode."""

    uri = f"file:{path}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    if cache_mib > 0:
        conn.execute(f"PRAGMA cache_size={-cache_mib * 1024}")
    if mmap_mib > 0:
        conn.execute(f"PRAGMA mmap_size={mmap_mib * 1024 * 1024}")
    return conn


def read_lance_url_sidecar_manifest(sidecar_dir: str | Path) -> dict[str, Any]:
    """Read and validate a sharded Lance URL sidecar manifest."""

    manifest_path = Path(sidecar_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != SIDECAR_FORMAT:
        msg = f"unsupported sidecar format: {manifest.get('format')}"
        raise ValueError(msg)
    if manifest.get("hash") != SIDECAR_HASH:
        msg = f"unsupported sidecar hash: {manifest.get('hash')}"
        raise ValueError(msg)
    if manifest.get("row_id_encoding") != UINT64_ENCODING:
        msg = f"unsupported row-id encoding: {manifest.get('row_id_encoding')}"
        raise ValueError(msg)
    if "shard_count" not in manifest or int(manifest["shard_count"]) <= 0:
        msg = "sidecar manifest must contain a positive shard_count"
        raise ValueError(msg)
    if not manifest.get("shards_dir"):
        msg = "sidecar manifest must contain shards_dir"
        raise ValueError(msg)
    sidecar_version = int(manifest.get("sidecar_schema_version", 1))
    if sidecar_version >= SIDECAR_SCHEMA_VERSION_WITH_ROWADDR and manifest.get("rowaddr_encoding") != UINT64_ENCODING:
        msg = f"unsupported row-address encoding: {manifest.get('rowaddr_encoding')}"
        raise ValueError(msg)
    return manifest


def manifest_has_rowaddr(manifest: dict[str, Any]) -> bool:
    return (
        int(manifest.get("sidecar_schema_version", 1)) >= SIDECAR_SCHEMA_VERSION_WITH_ROWADDR
        and manifest.get("rowaddr_encoding") == UINT64_ENCODING
    )
