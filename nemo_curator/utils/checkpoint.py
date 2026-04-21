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
        return "/".join([base.rstrip("/"), *list(parts)])
    import os

    return os.path.join(base, *parts)


class CheckpointManager:
    """Manages pipeline checkpoint state for resumability.

    Uses sharded files to avoid distributed write conflicts — one file per event:

    - ``completed/{source_key_hash}_{task_id}.json``: written by _CheckpointRecorderStage after each leaf task.
    - ``filtered/{source_key_hash}_{task_id}.json``: written by BaseStageAdapter when a task is
      dropped by a filter stage (produces 0 outputs). Counts toward the completion total so
      partitions with some filtered tasks can still be marked complete.
    - ``expected_increments/{source_key_hash}_{task_id}.json``: written by BaseStageAdapter
      whenever a single-input stage produces N > 1 outputs. Each increment file stores ``increment = N - 1``.

    On load, ``expected[source_key] = 1 + sum(increments_for_source_key)``.
    ``is_task_completed()`` returns True when ``completed + filtered >= expected``.

    This correctly handles chained fan-outs: if Stage1 produces 10 from 1 partition
    and Stage2 produces 3 from each of those 10 batches, the partition is only marked
    complete after all 30 leaf tasks record completion.
    """

    def __init__(self, checkpoint_path: str, storage_options: dict[str, Any] | None = None):
        self.checkpoint_path = checkpoint_path
        self.storage_options = storage_options or {}
        self._completed_path = _path_join(checkpoint_path, "completed")
        self._filtered_path = _path_join(checkpoint_path, "filtered")
        self._increments_path = _path_join(checkpoint_path, "expected_increments")
        # Populated by load()
        self._expected: dict[str, int] = {}
        self._completed_counts: dict[str, int] = defaultdict(int)
        self._filtered_counts: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def load(self) -> CheckpointManager:
        """Read all shard files from the checkpoint directory.

        Builds ``_expected``, ``_completed_counts``, and ``_filtered_counts`` in memory.
        Safe to call concurrently with writers (worst case: a newly written shard
        is missed and the task is processed again — never incorrect data).
        """
        fs = get_fs(self.checkpoint_path, self.storage_options)
        self._expected = self._load_increments(fs)
        self._completed_counts = self._load_shard_counts(fs, self._completed_path, "completed")
        self._filtered_counts = self._load_shard_counts(fs, self._filtered_path, "filtered")

        total_completed = sum(self._completed_counts.values())
        total_filtered = sum(self._filtered_counts.values())
        n_fanout_keys = sum(1 for v in self._expected.values() if v > 1)
        logger.info(
            f"Checkpoint loaded from {self.checkpoint_path}: "
            f"{total_completed} completed task shards, "
            f"{total_filtered} filtered task shards, "
            f"{n_fanout_keys} source keys with secondary fan-outs"
        )
        return self

    def _load_increments(self, fs: fsspec.AbstractFileSystem) -> dict[str, int]:
        """Load expected-increment shards and return source_key → expected count."""
        expected: dict[str, int] = {}
        if not fs.exists(self._increments_path):
            return expected
        try:
            for fpath in fs.ls(self._increments_path, detail=False):
                if not str(fpath).endswith(".json"):
                    continue
                try:
                    with fsspec.open(fpath, "r", **self.storage_options) as f:
                        data = json.load(f)
                    source_key = data["source_key"]
                    expected[source_key] = expected.get(source_key, 1) + int(data["increment"])
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Checkpoint: failed to read increment file {fpath}: {e}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Checkpoint: failed to list increment files: {e}")
        return expected

    def _load_shard_counts(self, fs: fsspec.AbstractFileSystem, directory: str, label: str) -> defaultdict[str, int]:
        """Count shards per source_key in *directory*; returns a defaultdict(int)."""
        counts: defaultdict[str, int] = defaultdict(int)
        if not fs.exists(directory):
            return counts
        try:
            for fpath in fs.ls(directory, detail=False):
                if not str(fpath).endswith(".json"):
                    continue
                try:
                    with fsspec.open(fpath, "r", **self.storage_options) as f:
                        data = json.load(f)
                    counts[data["source_key"]] += 1
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Checkpoint: failed to read {label} file {fpath}: {e}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Checkpoint: failed to list {label} files: {e}")
        return counts

    def get_completed_source_keys(self) -> frozenset[str]:
        """Return source_keys that have completed all expected leaf tasks."""
        all_keys = set(self._completed_counts) | set(self._filtered_counts)
        return frozenset(
            key
            for key in all_keys
            if self._completed_counts[key] + self._filtered_counts[key] >= self._expected.get(key, 1)
        )

    def is_task_completed(self, source_files: list[str]) -> bool:
        """Return True if all expected leaf tasks for this source partition are done.

        A task counts as done if it either completed the full pipeline or was
        legitimately filtered at some stage (both recorded in their respective shards).
        """
        if not source_files:
            return False
        source_key = "|".join(sorted(source_files))
        expected = self._expected.get(source_key, 1)
        done = self._completed_counts.get(source_key, 0) + self._filtered_counts.get(source_key, 0)
        return done >= expected

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def mark_completed(self, task_id: str, source_files: list[str]) -> None:
        """Write a completed shard for the given task.

        The filename is ``{source_key_hash}_{task_id}.json`` — keyed on BOTH the
        source partition and the task ID.  This prevents collisions when multiple
        source partitions produce leaf tasks with the same ``task_id`` (e.g.
        ``ImageReaderStage`` always produces ``image_batch_0``, ``image_batch_1``,
        … starting from 0 for every input tar file).  The source_key prefix makes
        each (source_partition, task_id) pair map to a distinct file while keeping
        writes idempotent for retries.
        """
        if not source_files:
            return
        source_key = "|".join(sorted(source_files))
        source_key_hash = hashlib.sha256(source_key.encode()).hexdigest()[:16]
        safe_task_id = task_id.replace("/", "_").replace(" ", "_").replace(":", "_")
        filename = f"{source_key_hash}_{safe_task_id}.json"
        path = _path_join(self._completed_path, filename)
        self._write_json(
            path,
            {
                "source_key": source_key,
                "completed_at": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    def mark_filtered(self, task_id: str, source_files: list[str]) -> None:
        """Write a filtered shard for a task that was dropped by a filter stage.

        Filtered tasks count toward the completion total so that a source partition
        with some tasks filtered out can still be marked complete on resume.
        Uses the same filename scheme as ``mark_completed`` for idempotency.
        """
        if not source_files:
            return
        source_key = "|".join(sorted(source_files))
        source_key_hash = hashlib.sha256(source_key.encode()).hexdigest()[:16]
        safe_task_id = task_id.replace("/", "_").replace(" ", "_").replace(":", "_")
        filename = f"{source_key_hash}_{safe_task_id}.json"
        path = _path_join(self._filtered_path, filename)
        self._write_json(path, {"source_key": source_key})

    def write_expected_increment(self, source_key: str, triggering_task_id: str, increment: int) -> None:
        """Write an expected-count increment shard.

        Called by BaseStageAdapter when it detects a fan-out:
        one input with source_files produces N > 1 outputs → increment = N - 1.

        Handles chained fan-outs correctly because each triggering task writes its
        own file (unique filename = no races), and expected is summed on load.
        """
        key_hash = hashlib.sha256(source_key.encode()).hexdigest()[:16]
        safe_task_id = triggering_task_id.replace("/", "_").replace(" ", "_").replace(":", "_")
        filename = f"{key_hash}_{safe_task_id}.json"
        path = _path_join(self._increments_path, filename)
        self._write_json(path, {"source_key": source_key, "increment": increment})

    # ------------------------------------------------------------------
    # Class-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def exists(checkpoint_path: str, storage_options: dict[str, Any] | None = None) -> bool:
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
        except Exception as e:  # noqa: BLE001
            # S3 / GCS don't need explicit directory creation; log for local paths
            logger.debug(f"Checkpoint: makedirs({parent!r}) skipped: {e}")
        try:
            with fsspec.open(path, "w", **self.storage_options) as f:
                json.dump(data, f)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"Checkpoint: failed to write {path!r} — checkpoint shard lost. "
                f"Error: {exc!r}. "
                "Check filesystem permissions and that the checkpoint path is on a "
                "shared filesystem accessible to all Ray workers. "
                "The pipeline will continue but this task will be re-processed on resume."
            )
