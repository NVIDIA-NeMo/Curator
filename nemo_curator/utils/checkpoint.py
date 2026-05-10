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

"""Checkpoint manager for pipeline resumability.

Uses a single LMDB environment with two named databases:
  - partitions: resumability_key_hash → struct(<II?>: expected, count, finalized)
  - completions: resumability_key_hash:resumability_task_key_hash → b"1"

A partition is considered done only when its ``finalized`` flag is set, which
happens inside ``mark_completed`` once ``count`` catches up to ``expected``.
Storing ``count`` denormalized lets ``mark_completed`` avoid an O(leaves) scan,
and the explicit ``finalized`` flag prevents a false-completion window after
``reset_partition`` rewrites ``expected`` back down to 1 on resume.

All LMDB access is serialized through a singleton Ray actor so workers never
open the database file directly (it lives on the driver/head node).
"""

import asyncio
import concurrent.futures
import hashlib
import os
import struct
from typing import Any

import lmdb
from loguru import logger


def _checkpoint_get(ref: Any) -> Any:  # noqa: ANN401
    """Call ray.get() safely from both sync and async Ray actor contexts.

    Xenna wraps each stage in an async Ray actor. Calling ray.get() directly
    inside an async actor blocks the event loop and triggers Ray's warning
    "Using blocking ray.get inside async actor." This helper detects an active
    event loop and offloads the blocking call to a ThreadPoolExecutor instead.
    """
    import ray

    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(ray.get, ref).result()
    except RuntimeError:
        return ray.get(ref)


def _key_hash(resumability_key: str) -> bytes:
    return hashlib.sha256(resumability_key.encode()).hexdigest()[:16].encode()


_PARTITION_STRUCT = struct.Struct("<II?")


def _pack_partition(expected: int, count: int, finalized: bool) -> bytes:
    return _PARTITION_STRUCT.pack(expected, count, finalized)


def _unpack_partition(raw: bytes) -> tuple[int, int, bool]:
    expected, count, finalized = _PARTITION_STRUCT.unpack(raw)
    return expected, count, finalized


class CheckpointManager:
    """Manages checkpoint state in a local LMDB database.

    One LMDB environment, two named databases:
      - partitions: tracks how many completions are expected per source partition
      - completions: records each finished (or dropped) leaf task
    """

    def __init__(self, checkpoint_path: str) -> None:
        os.makedirs(checkpoint_path, exist_ok=True)
        db_path = os.path.join(checkpoint_path, "checkpoint.lmdb")
        self._env = lmdb.open(db_path, max_dbs=2, map_size=1 << 30)
        self._partitions_db = self._env.open_db(b"partitions")
        self._completions_db = self._env.open_db(b"completions")

    # ------------------------------------------------------------------
    # Actor-facing methods (called from _CheckpointActor)
    # ------------------------------------------------------------------

    def init_partition(self, resumability_key: str) -> None:
        """Register a new source partition (no-op if already present)."""
        h = _key_hash(resumability_key)
        with self._env.begin(write=True) as txn:
            if txn.get(h, db=self._partitions_db) is None:
                txn.put(h, _pack_partition(1, 0, False), db=self._partitions_db)

    def is_task_completed(self, resumability_key: str) -> bool:
        """Return True if this partition was finalized (all expected leaves recorded)."""
        h = _key_hash(resumability_key)
        with self._env.begin() as txn:
            raw = txn.get(h, db=self._partitions_db)
            if raw is None:
                return False
            _expected, _count, finalized = _unpack_partition(raw)
        return finalized

    def mark_completed(self, resumability_task_key: str, resumability_key: str) -> None:
        """Record a completed (or dropped) leaf task. Idempotent.

        Increments the partition's denormalized ``count`` only on the first write
        for a given leaf, and flips ``finalized`` once ``count >= expected``.
        """
        h = _key_hash(resumability_key)
        entry_key = h + b":" + _key_hash(resumability_task_key)
        with self._env.begin(write=True) as txn:
            if txn.get(entry_key, db=self._completions_db) is not None:
                return
            txn.put(entry_key, b"1", db=self._completions_db)

            raw = txn.get(h, db=self._partitions_db)
            if raw is None:
                # Partition was never registered; treat this leaf as a 1-of-1 partition.
                txn.put(h, _pack_partition(1, 1, True), db=self._partitions_db)
                return
            expected, count, finalized = _unpack_partition(raw)
            count += 1
            if not finalized and count >= expected:
                finalized = True
            txn.put(h, _pack_partition(expected, count, finalized), db=self._partitions_db)

    def are_leaves_completed(self, pairs: list[tuple[str, str]]) -> list[bool]:
        """Batch-check whether each (resumability_key, resumability_task_key) leaf is recorded.

        Single read transaction; one B-tree lookup per pair.  Returns a parallel
        list of booleans aligned with ``pairs``.
        """
        with self._env.begin() as txn:
            return [
                txn.get(_key_hash(key) + b":" + _key_hash(task_key), db=self._completions_db) is not None
                for key, task_key in pairs
            ]

    def add_expected(self, resumability_key: str, increment: int) -> None:
        """Atomically increase the expected completion count for a partition."""
        h = _key_hash(resumability_key)
        with self._env.begin(write=True) as txn:
            raw = txn.get(h, db=self._partitions_db)
            if raw is None:
                expected, count, finalized = 1, 0, False
            else:
                expected, count, finalized = _unpack_partition(raw)
            expected += increment
            if not finalized and count >= expected:
                finalized = True
            txn.put(h, _pack_partition(expected, count, finalized), db=self._partitions_db)

    def reset_partition(self, resumability_key: str) -> None:
        """Reset a partition's expected count for a fresh attempt while preserving completed leaves.

        A previous interrupted run may have called add_expected (fan-out) without
        writing all completions, inflating the expected count. We rewind expected
        to 1 so the upcoming run's fan-outs re-register cleanly via add_expected.
        Completion entries are kept so that ``_drop_completed_inputs`` can skip
        leaves already finished in earlier runs. ``finalized`` is cleared and
        ``count`` is recomputed from the surviving completion entries; without
        the explicit flag, a crash between this reset and the first fan-out
        would leave ``count >= expected=1`` and the next resume would falsely
        treat the partition as done.
        """
        h = _key_hash(resumability_key)
        prefix = h + b":"
        with self._env.begin(write=True) as txn:
            count = 0
            cursor = txn.cursor(db=self._completions_db)
            if cursor.set_range(prefix):
                for k in cursor.iternext(keys=True, values=False):
                    if not k.startswith(prefix):
                        break
                    count += 1
            txn.put(h, _pack_partition(1, count, False), db=self._partitions_db)

    def close(self) -> None:
        self._env.close()


# ---------------------------------------------------------------------------
# Ray actor
# ---------------------------------------------------------------------------


def _get_actor_name(checkpoint_path: str) -> str:
    digest = hashlib.sha256(os.path.abspath(checkpoint_path).encode()).hexdigest()[:16]
    return f"nemo_curator_ckpt_{digest}"


def get_or_create_checkpoint_actor(checkpoint_path: str) -> Any:  # noqa: ANN401
    """Return (or create) the singleton named checkpoint actor for this path."""
    import ray

    name = _get_actor_name(checkpoint_path)

    @ray.remote(num_cpus=0.1, max_restarts=-1)
    class _CheckpointActor:
        def __init__(self, path: str) -> None:
            self._mgr = CheckpointManager(path)

        def init_partition(self, resumability_key: str) -> None:
            self._mgr.init_partition(resumability_key)

        def is_task_completed(self, resumability_key: str) -> bool:
            return self._mgr.is_task_completed(resumability_key)

        def mark_completed(self, resumability_task_key: str, resumability_key: str) -> None:
            self._mgr.mark_completed(resumability_task_key, resumability_key)

        def are_leaves_completed(self, pairs: list[tuple[str, str]]) -> list[bool]:
            return self._mgr.are_leaves_completed(pairs)

        def add_expected(self, resumability_key: str, increment: int) -> None:
            self._mgr.add_expected(resumability_key, increment)

        def reset_partition(self, resumability_key: str) -> None:
            self._mgr.reset_partition(resumability_key)

    try:
        actor = ray.get_actor(name)
        logger.debug(f"Reusing existing checkpoint actor: {name}")
    except ValueError:
        logger.info(f"Creating new checkpoint actor: {name}")
        # get_if_exists=True makes this safe under concurrent setup() calls:
        # if another worker races to create the actor first, Ray returns the
        # existing handle instead of raising ActorAlreadyExistsError.
        actor = _CheckpointActor.options(  # type: ignore[attr-defined]
            name=name,
            lifetime="detached",
            get_if_exists=True,
        ).remote(checkpoint_path)
    return actor
