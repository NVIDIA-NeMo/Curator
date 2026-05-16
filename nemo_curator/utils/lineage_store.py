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
"""LMDB-backed lineage store for task DAG checkpointing.

Stores, per task ``_udid``:

- parent ``_udid``s
- child ``_udid``s
- ``task_type`` ("source" | "middle" | "leaf" | "source_leaf")
- ``completed`` flag (reserved for future resumability work; never auto-set today)

Architecture:

- :class:`LineageStore` — direct LMDB owner. Used inside the writer actor for
  the active pipeline, and also opened standalone (e.g., after a run finishes,
  in a fresh process) to read records.
- :class:`LineageWriterActor` — named Ray actor that wraps a single
  :class:`LineageStore`. The only writer during a pipeline run, which is what
  lets the file live safely on NFS / Lustre.
- :func:`record_lineage` — write helper called from stages. No-op unless a
  :class:`LineageWriterActor` is registered in the cluster.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import lmdb
import ray
from loguru import logger

LINEAGE_ACTOR_NAME = "nemo_curator_lineage_writer"

_IN_EDGES_DB = b"in_edges"
_OUT_EDGES_DB = b"out_edges"
_TASK_TYPE_DB = b"task_type"
_COMPLETED_DB = b"completed"

_DEFAULT_MAP_SIZE = 1 << 34  # 16 GiB; sparse on Linux so effectively free

_TYPE_SOURCE = b"source"
_TYPE_MIDDLE = b"middle"
_TYPE_LEAF = b"leaf"
_TYPE_SOURCE_LEAF = b"source_leaf"


def _classify(has_parent: bool, has_child: bool) -> bytes:
    if has_parent and has_child:
        return _TYPE_MIDDLE
    if has_parent:
        return _TYPE_LEAF
    if has_child:
        return _TYPE_SOURCE
    return _TYPE_SOURCE_LEAF


def _udid_to_key(udid: str) -> bytes:
    return udid.encode("ascii")


def _key_to_udid(key: bytes) -> str:
    return key.decode("ascii")


def _path_to_udid(lineage_path: str) -> str:
    """Mirror of the udid derivation in ``Task._set_lineage`` ([tasks.py:59])."""
    return hashlib.sha256(lineage_path.encode()).hexdigest()[:32]


@dataclass
class LineageRecord:
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    task_type: str = "source_leaf"
    completed: bool = False


class LineageStore:
    """Direct LMDB owner for the lineage checkpoint.

    Not safe to use from multiple processes concurrently. The writer actor uses
    one of these as its backing store during a pipeline run; tests and
    post-pipeline inspection tools instantiate one directly to read records.
    """

    def __init__(self, path: str | Path, map_size: int = _DEFAULT_MAP_SIZE):
        self._path = str(Path(path).absolute())
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._env = lmdb.open(
            self._path,
            subdir=False,
            lock=False,
            max_dbs=4,
            map_size=map_size,
            metasync=False,
            sync=True,
            readahead=False,
        )
        self._in_db = self._env.open_db(_IN_EDGES_DB, dupsort=True)
        self._out_db = self._env.open_db(_OUT_EDGES_DB, dupsort=True)
        self._type_db = self._env.open_db(_TASK_TYPE_DB)
        self._completed_db = self._env.open_db(_COMPLETED_DB)

    @staticmethod
    def _has_dup(txn: lmdb.Transaction, db: lmdb._Database, key: bytes) -> bool:
        with txn.cursor(db=db) as cur:
            return cur.set_key(key)

    def _record_emission_once(self, parent_udids: list[str], child_udids: list[str]) -> None:
        parent_keys = [_udid_to_key(u) for u in parent_udids]
        child_keys = [_udid_to_key(u) for u in child_udids]
        with self._env.begin(write=True) as txn:
            for child_key in child_keys:
                for parent_key in parent_keys:
                    # In dupsort dbs, the default flags allow multiple distinct values
                    # per key and silently drop exact (key, value) duplicates. We
                    # deliberately do NOT pass overwrite=False — that maps to
                    # MDB_NOOVERWRITE which refuses any new value once the key has
                    # any existing value, blocking incremental parent attribution.
                    txn.put(child_key, parent_key, db=self._in_db)
                    txn.put(parent_key, child_key, db=self._out_db)

            affected = {*child_keys, *parent_keys}
            for udid_key in affected:
                has_parent = self._has_dup(txn, self._in_db, udid_key)
                has_child = self._has_dup(txn, self._out_db, udid_key)
                txn.put(udid_key, _classify(has_parent, has_child), db=self._type_db, overwrite=True)

    def record_emission(self, parent_udids: list[str], child_udids: list[str]) -> None:
        """Append edges for ``(parent, child)`` pairs and refresh ``task_type``
        for every affected udid. Idempotent under retries and incremental
        parent attribution; no-op when ``child_udids`` is empty."""
        if not child_udids:
            return
        try:
            self._record_emission_once(parent_udids, child_udids)
        except lmdb.MapFullError:
            new_size = self._env.info()["map_size"] * 2
            logger.warning(f"LMDB map full at {self._path}; growing to {new_size} bytes")
            self._env.set_mapsize(new_size)
            self._record_emission_once(parent_udids, child_udids)

    def mark_completed(self, udid: str) -> None:
        with self._env.begin(write=True) as txn:
            txn.put(_udid_to_key(udid), b"1", db=self._completed_db, overwrite=True)

    def is_completed(self, udid: str) -> bool:
        with self._env.begin() as txn:
            return txn.get(_udid_to_key(udid), db=self._completed_db) is not None

    def get(self, udid: str) -> LineageRecord | None:
        key = _udid_to_key(udid)
        with self._env.begin() as txn:
            task_type = txn.get(key, db=self._type_db)
            if task_type is None:
                return None
            parents: list[str] = []
            with txn.cursor(db=self._in_db) as cur:
                if cur.set_key(key):
                    parents = [_key_to_udid(v) for v in cur.iternext_dup()]
            children: list[str] = []
            with txn.cursor(db=self._out_db) as cur:
                if cur.set_key(key):
                    children = [_key_to_udid(v) for v in cur.iternext_dup()]
            completed = txn.get(key, db=self._completed_db) is not None
            return LineageRecord(
                parents=parents,
                children=children,
                task_type=task_type.decode("ascii"),
                completed=completed,
            )

    def iter_records(self) -> list[tuple[str, LineageRecord]]:
        results: list[tuple[str, LineageRecord]] = []
        with self._env.begin() as txn, txn.cursor(db=self._type_db) as cur:
            for key, _ in cur:
                udid = _key_to_udid(key)
                rec = self.get(udid)
                if rec is not None:
                    results.append((udid, rec))
        return results

    def _traverse(self, udid: str, attr: str) -> dict[str, LineageRecord]:
        start = self.get(udid)
        if start is None:
            return {}
        result: dict[str, LineageRecord] = {}
        queue: deque[str] = deque(getattr(start, attr))
        while queue:
            neighbor = queue.popleft()
            if neighbor == udid or neighbor in result:
                continue
            rec = self.get(neighbor)
            if rec is None:
                continue
            result[neighbor] = rec
            queue.extend(getattr(rec, attr))
        return result

    def get_all_parents(self, udid: str) -> dict[str, LineageRecord]:
        """Return every ancestor of ``udid`` (transitive parents) keyed by udid.

        Excludes ``udid`` itself. Returns ``{}`` when ``udid`` is unknown or
        has no parents.
        """
        return self._traverse(udid, "parents")

    def get_all_children(self, udid: str) -> dict[str, LineageRecord]:
        """Return every descendant of ``udid`` (transitive children) keyed by udid.

        Excludes ``udid`` itself. Returns ``{}`` when ``udid`` is unknown or
        has no children.
        """
        return self._traverse(udid, "children")

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None  # type: ignore[assignment]


@ray.remote(num_cpus=0)
class LineageWriterActor:
    """Singleton owner of the LMDB env, spawned by the executor when
    ``Pipeline.run(checkpoint_path=...)`` is provided. Workers send lineage
    events here via :func:`record_lineage`. Because it is the only process
    that writes to the LMDB file, no cross-process file lock is required and
    the file may safely live on NFS or Lustre."""

    def __init__(self, path: str, map_size: int = _DEFAULT_MAP_SIZE):
        self._store = LineageStore(path, map_size=map_size)

    def record_emission(self, parent_udids: list[str], child_udids: list[str]) -> None:
        self._store.record_emission(parent_udids, child_udids)

    def mark_completed(self, udid: str) -> None:
        self._store.mark_completed(udid)

    def is_completed(self, udid: str) -> bool:
        return self._store.is_completed(udid)

    def get(self, udid: str) -> LineageRecord | None:
        return self._store.get(udid)

    def iter_records(self) -> list[tuple[str, LineageRecord]]:
        return self._store.iter_records()

    def get_all_parents(self, udid: str) -> dict[str, LineageRecord]:
        return self._store.get_all_parents(udid)

    def get_all_children(self, udid: str) -> dict[str, LineageRecord]:
        return self._store.get_all_children(udid)

    def close(self) -> None:
        self._store.close()


def record_lineage(parent_udids: list[str], child_udids: list[str]) -> None:
    """Persist parent/child edges via the named :class:`LineageWriterActor`.

    No-op when Ray is not initialized or no such actor is registered. The
    actor is spawned by the executor only when ``Pipeline.run`` is called
    with ``checkpoint_path``, so the absence of the actor is what gates
    recording.

    Intended to be called from inside ``process_batch`` right after
    :func:`nemo_curator.stages.base.assign_child_lineage`. Pass the parent
    tasks' ``_udid`` values (typically ``[task._udid]`` for 1:N stages, or one
    udid per parent for joins) and the emitted children's ``_udid`` values.
    Empty udids (``EmptyTask`` roots and tasks that haven't been lineage-assigned
    yet) are filtered out, so source tasks naturally end up with empty
    ``in_edges``.
    """
    if not ray.is_initialized():
        return
    try:
        actor = ray.get_actor(LINEAGE_ACTOR_NAME)
    except ValueError:
        return

    parent_udids = [u for u in parent_udids if u]
    child_udids = [u for u in child_udids if u]
    if not child_udids:
        return

    ray.get(actor.record_emission.remote(parent_udids, child_udids))
