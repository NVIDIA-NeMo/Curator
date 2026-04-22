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

    Stores one JSON file per source partition in a flat directory:

        {checkpoint_path}/{source_key_hash}.json

    Each file contains all state for that source:

        {
          "source_key": "file_a.tar|file_b.tar",
          "completed": ["leaf_task_0", "leaf_task_1", ...],
          "filtered":  ["leaf_task_99"],
          "increments": [{"triggering_task_id": "file_group_abc", "increment": 9}]
        }

    Writes use read-modify-write.  ``mark_completed`` is called only from the
    single-actor ``_CheckpointRecorderStage``, so those writes are sequential.
    ``mark_filtered`` and ``write_expected_increment`` are called from parallel
    ``BaseStageAdapter`` actors; a rare concurrent-write race may drop one update,
    causing at most one task to be re-processed on resume (acceptable — all stages
    are idempotent).

    Completion check: ``len(completed) + len(filtered) >= 1 + sum(increments)``.
    This handles chained fan-outs: if Stage1 produces 10 from 1 partition and
    Stage2 produces 3 from each of those 10 batches, the partition is only marked
    complete after all 30 leaf tasks record completion.
    """

    def __init__(self, checkpoint_path: str, storage_options: dict[str, Any] | None = None):
        self.checkpoint_path = checkpoint_path
        self.storage_options = storage_options or {}
        # Populated by load()
        self._expected: dict[str, int] = {}
        self._completed_counts: dict[str, int] = defaultdict(int)
        self._filtered_counts: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def load(self) -> CheckpointManager:
        """Read all per-source files from the checkpoint directory.

        Builds ``_expected``, ``_completed_counts``, and ``_filtered_counts`` in memory.
        Safe to call concurrently with writers (worst case: a newly written file
        is missed and the task is processed again — never incorrect data).
        """
        fs = get_fs(self.checkpoint_path, self.storage_options)
        self._expected = {}
        self._completed_counts = defaultdict(int)
        self._filtered_counts = defaultdict(int)

        if not fs.exists(self.checkpoint_path):
            logger.info("Checkpoint loaded: no checkpoint directory found — running fresh.")
            return self

        try:
            for fpath in fs.ls(self.checkpoint_path, detail=False):
                if not str(fpath).endswith(".json"):
                    continue
                try:
                    with fsspec.open(fpath, "r", **self.storage_options) as f:
                        data = json.load(f)
                    source_key = data.get("source_key", "")
                    if not source_key:
                        continue
                    increments = data.get("increments", [])
                    if increments:
                        self._expected[source_key] = 1 + sum(inc.get("increment", 0) for inc in increments)
                    self._completed_counts[source_key] = len(set(data.get("completed", [])))
                    self._filtered_counts[source_key] = len(set(data.get("filtered", [])))
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Checkpoint: failed to read {fpath}: {e}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Checkpoint: failed to list checkpoint files: {e}")

        total_completed = sum(self._completed_counts.values())
        total_filtered = sum(self._filtered_counts.values())
        n_fanout_keys = sum(1 for v in self._expected.values() if v > 1)
        logger.info(
            f"Checkpoint loaded from {self.checkpoint_path}: "
            f"{total_completed} completed tasks, "
            f"{total_filtered} filtered tasks, "
            f"{n_fanout_keys} source keys with secondary fan-outs"
        )
        return self

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
        legitimately filtered at some stage (both recorded in their respective fields).
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
        """Record a completed leaf task for the given source partition.

        Updates ``{checkpoint_path}/{source_key_hash}.json`` by appending
        ``task_id`` to the ``completed`` list (deduplicated, so retries are safe).
        """
        if not source_files:
            return
        source_key = "|".join(sorted(source_files))
        source_key_hash = hashlib.sha256(source_key.encode()).hexdigest()[:16]
        self._update_source_file(source_key_hash, source_key, "completed", task_id)

    def mark_filtered(self, task_id: str, source_files: list[str]) -> None:
        """Record a task that was dropped by a filter stage.

        Filtered tasks count toward the completion total so that a source partition
        with some tasks filtered out can still be marked complete on resume.
        """
        if not source_files:
            return
        source_key = "|".join(sorted(source_files))
        source_key_hash = hashlib.sha256(source_key.encode()).hexdigest()[:16]
        self._update_source_file(source_key_hash, source_key, "filtered", task_id)

    def write_expected_increment(self, source_key: str, triggering_task_id: str, increment: int) -> None:
        """Record a fan-out increment for the given source partition.

        Called by BaseStageAdapter when it detects a fan-out:
        one input with source_files produces N > 1 outputs → increment = N - 1.

        Handles chained fan-outs correctly: each triggering task appends its own
        entry (deduplicated by triggering_task_id), and expected is summed on load.
        """
        key_hash = hashlib.sha256(source_key.encode()).hexdigest()[:16]
        self._update_source_file(
            key_hash,
            source_key,
            "increments",
            {"triggering_task_id": triggering_task_id, "increment": increment},
        )

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

    def _read_source_file(self, source_key_hash: str) -> dict[str, Any]:
        """Read the per-source file; return default empty structure if missing or unreadable."""
        path = _path_join(self.checkpoint_path, f"{source_key_hash}.json")
        fs = get_fs(path, self.storage_options)
        if not fs.exists(path):
            return {"source_key": "", "completed": [], "filtered": [], "increments": []}
        try:
            with fsspec.open(path, "r", **self.storage_options) as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Checkpoint: failed to read {path!r}, treating as empty: {e}")
            return {"source_key": "", "completed": [], "filtered": [], "increments": []}

    def _update_source_file(self, source_key_hash: str, source_key: str, field: str, value: str | dict[str, Any]) -> None:
        """Read-modify-write a per-source checkpoint file.

        For ``completed`` and ``filtered``: appends ``value`` (a task_id string)
        only if not already present — retries are idempotent.
        For ``increments``: appends ``value`` (a dict with ``triggering_task_id``)
        only if that triggering_task_id is not already recorded.
        """
        data = self._read_source_file(source_key_hash)
        data["source_key"] = source_key

        if field in ("completed", "filtered"):
            if value not in data[field]:
                data[field].append(value)
        elif field == "increments":
            existing_ids = {inc["triggering_task_id"] for inc in data["increments"]}
            if value["triggering_task_id"] not in existing_ids:
                data["increments"].append(value)

        path = _path_join(self.checkpoint_path, f"{source_key_hash}.json")
        self._write_json(path, data)

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
                f"Checkpoint: failed to write {path!r} — checkpoint update lost. "
                f"Error: {exc!r}. "
                "Check filesystem permissions and that the checkpoint path is on a "
                "shared filesystem accessible to all Ray workers. "
                "The pipeline will continue but this task will be re-processed on resume."
            )
