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

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import fsspec
from loguru import logger

from nemo_curator.utils.file_utils import get_fs


def _path_join(base: str, *parts: str) -> str:
    """Join path components, handling both local and remote (s3://, gs://) paths."""
    if "://" in base:
        return "/".join([base.rstrip("/")] + list(parts))
    import os

    return os.path.join(base, *parts)


class CheckpointManager:
    """Manages pipeline checkpoint state for resumability.

    Uses sharded files to avoid distributed write conflicts — one file per event:

    - ``completed/{task_id}.json``: written by _CheckpointRecorderStage after each leaf task.
    - ``expected_increments/{source_key_hash}_{task_id}.json``: written by BaseStageAdapter
      whenever a stage produces N > 1 outputs from a single input that already has
      ``source_files``. Each increment file stores ``increment = N - 1``.

    On load, ``expected[source_key] = 1 + sum(increments_for_source_key)``.
    ``is_task_completed()`` returns True when ``completed_count >= expected``.

    This correctly handles chained fan-outs: if Stage1 produces 10 from 1 partition
    and Stage2 produces 3 from each of those 10 batches, the partition is only marked
    complete after all 30 leaf tasks record completion.
    """

    def __init__(self, checkpoint_path: str, storage_options: dict[str, Any] | None = None):
        self.checkpoint_path = checkpoint_path
        self.storage_options = storage_options or {}
        self._completed_path = _path_join(checkpoint_path, "completed")
        self._increments_path = _path_join(checkpoint_path, "expected_increments")
        # Populated by load()
        self._expected: dict[str, int] = {}
        self._completed_counts: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def load(self) -> "CheckpointManager":
        """Read all shard files from the checkpoint directory.

        Builds ``_expected`` and ``_completed_counts`` in memory.
        Safe to call concurrently with writers (worst case: a newly written shard
        is missed and the task is processed again — never incorrect data).
        """
        fs = get_fs(self.checkpoint_path, self.storage_options)

        # Load expected increments
        self._expected = {}
        if fs.exists(self._increments_path):
            try:
                increment_files = fs.ls(self._increments_path, detail=False)
                for fpath in increment_files:
                    if not str(fpath).endswith(".json"):
                        continue
                    try:
                        with fsspec.open(fpath, "r", **self.storage_options) as f:
                            data = json.load(f)
                        source_key = data["source_key"]
                        increment = int(data["increment"])
                        self._expected[source_key] = self._expected.get(source_key, 1) + increment
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"Checkpoint: failed to read increment file {fpath}: {e}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Checkpoint: failed to list increment files: {e}")

        # Load completed counts
        self._completed_counts = defaultdict(int)
        if fs.exists(self._completed_path):
            try:
                completed_files = fs.ls(self._completed_path, detail=False)
                for fpath in completed_files:
                    if not str(fpath).endswith(".json"):
                        continue
                    try:
                        with fsspec.open(fpath, "r", **self.storage_options) as f:
                            data = json.load(f)
                        source_key = data["source_key"]
                        self._completed_counts[source_key] += 1
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"Checkpoint: failed to read completed file {fpath}: {e}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Checkpoint: failed to list completed files: {e}")

        total_completed = sum(self._completed_counts.values())
        n_fanout_keys = sum(1 for v in self._expected.values() if v > 1)
        logger.info(
            f"Checkpoint loaded from {self.checkpoint_path}: "
            f"{total_completed} completed task shards, "
            f"{n_fanout_keys} source keys with secondary fan-outs"
        )
        return self

    def get_completed_source_keys(self) -> frozenset[str]:
        """Return source_keys that have completed all expected leaf tasks."""
        return frozenset(
            key
            for key in self._completed_counts
            if self._completed_counts[key] >= self._expected.get(key, 1)
        )

    def is_task_completed(self, source_files: list[str]) -> bool:
        """Return True if all expected leaf tasks for this source partition are done."""
        if not source_files:
            return False
        source_key = "|".join(sorted(source_files))
        expected = self._expected.get(source_key, 1)
        completed = self._completed_counts.get(source_key, 0)
        return completed >= expected

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def mark_completed(self, task_id: str, source_files: list[str]) -> None:
        """Write a completed shard for the given task.

        Each call writes a unique file (keyed by task_id), so concurrent writers
        never clobber each other.
        """
        if not source_files:
            return
        source_key = "|".join(sorted(source_files))
        safe_task_id = task_id.replace("/", "_").replace(" ", "_").replace(":", "_")
        path = _path_join(self._completed_path, f"{safe_task_id}.json")
        self._write_json(
            path,
            {
                "source_key": source_key,
                "completed_at": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    def write_expected_increment(
        self, source_key: str, triggering_task_id: str, increment: int
    ) -> None:
        """Write an expected-count increment shard.

        Called by BaseStageAdapter when it detects a fan-out:
        one input with source_files produces N > 1 outputs → increment = N - 1.

        Handles chained fan-outs correctly because each triggering task writes its
        own file (unique filename = no races), and expected is summed on load.
        """
        key_hash = hashlib.sha256(source_key.encode()).hexdigest()[:16]
        safe_task_id = (
            triggering_task_id.replace("/", "_").replace(" ", "_").replace(":", "_")
        )
        filename = f"{key_hash}_{safe_task_id}.json"
        path = _path_join(self._increments_path, filename)
        self._write_json(path, {"source_key": source_key, "increment": increment})

    # ------------------------------------------------------------------
    # Class-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def exists(
        checkpoint_path: str, storage_options: dict[str, Any] | None = None
    ) -> bool:
        """Return True if any checkpoint data exists at the given path."""
        try:
            fs = get_fs(checkpoint_path, storage_options or {})
            return fs.exists(checkpoint_path)
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_json(self, path: str, data: dict[str, Any]) -> None:
        """Write JSON data to path, ensuring parent directories exist."""
        fs = get_fs(path, self.storage_options)
        parent = path.rsplit("/", 1)[0] if "/" in path else path
        try:
            fs.makedirs(parent, exist_ok=True)
        except Exception:  # noqa: BLE001
            pass  # S3 / GCS don't need explicit directory creation
        with fsspec.open(path, "w", **self.storage_options) as f:
            json.dump(data, f)
