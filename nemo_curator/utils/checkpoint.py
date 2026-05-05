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
  - partitions: resumability_key_hash → expected completion count
  - completions: resumability_key_hash:task_uuid → b"1"

All LMDB access is serialized through a singleton Ray actor so workers never
open the database file directly (it lives on the driver/head node).
"""

import hashlib
import os
import struct
from typing import Any

import lmdb
from loguru import logger


def _key_hash(resumability_key: str) -> bytes:
    return hashlib.sha256(resumability_key.encode()).hexdigest()[:16].encode()


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
        """Register a new source partition with expected=1 (no-op if already present)."""
        h = _key_hash(resumability_key)
        with self._env.begin(write=True) as txn:
            if txn.get(h, db=self._partitions_db) is None:
                txn.put(h, struct.pack("<I", 1), db=self._partitions_db)

    def is_task_completed(self, resumability_key: str) -> bool:
        """Return True if all expected leaf tasks for this partition have completed."""
        h = _key_hash(resumability_key)
        with self._env.begin() as txn:
            raw = txn.get(h, db=self._partitions_db)
            if raw is None:
                return False
            expected = struct.unpack("<I", raw)[0]

            prefix = h + b":"
            count = 0
            cursor = txn.cursor(db=self._completions_db)
            if cursor.set_range(prefix):
                for k in cursor.iternext(keys=True, values=False):
                    if not k.startswith(prefix):
                        break
                    count += 1

        return count >= expected

    def mark_completed(self, resumability_uuid: str, resumability_key: str) -> None:
        """Record a completed (or dropped) leaf task.  Idempotent."""
        h = _key_hash(resumability_key)
        entry_key = h + b":" + resumability_uuid.encode()
        with self._env.begin(write=True) as txn:
            txn.put(entry_key, b"1", db=self._completions_db)

    def add_expected(self, resumability_key: str, increment: int) -> None:
        """Atomically increase the expected completion count for a partition."""
        h = _key_hash(resumability_key)
        with self._env.begin(write=True) as txn:
            raw = txn.get(h, db=self._partitions_db)
            current = struct.unpack("<I", raw)[0] if raw else 1
            txn.put(h, struct.pack("<I", current + increment), db=self._partitions_db)

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

        def mark_completed(self, resumability_uuid: str, resumability_key: str) -> None:
            self._mgr.mark_completed(resumability_uuid, resumability_key)

        def add_expected(self, resumability_key: str, increment: int) -> None:
            self._mgr.add_expected(resumability_key, increment)

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
