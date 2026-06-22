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
"""Per-writer LMDB owner that tracks per-source pending counters for
resumability.

Resumability state lives in a shared ``.nemo_curator_metadata`` directory so
that independent runs pointed at the same checkpoint location can record
completed sources without corrupting a shared file. The motivating case is a
SLURM array: each array task is its own job → its own Ray cluster → its own
actor, often on a different node.

LMDB cannot be safely written by multiple processes on different hosts against
a single file — its inter-process lock lives in a memory-mapped lock file that
is not shared across nodes on a networked filesystem (e.g. Lustre). So instead
of one shared file:

- each actor WRITES ONLY to its own file ``<dir>/<host>-<pid>.mdb`` (it is the
  sole writer of that file, so no cross-process locking is needed), and
- on startup it READS THE UNION of completed sources across every ``*.mdb``
  file in the directory.

A rerun therefore sees everything every prior writer finished and skips it.
This assumes each writer owns a disjoint set of sources (the usual
shard-per-task model); two writers completing the *same* source id is harmless
(idempotent — the source is simply marked done).

Workers fire ``apply_deltas`` fire-and-forget. The actor:

- Maintains ``_pending: dict[source_id, int]`` (counter per in-flight source)
- Maintains ``_applied: dict[task_id, delta]`` — ``task_id`` is the output task
  id used as the dedup key (Ray-retry dedup; on a retry firing a *different*
  delta, rewrites the pending counter to reflect the newest observation rather
  than raising)
- Writes a single LMDB row to its own file per source when its counter hits zero

By design, ``apply_deltas`` **never raises**. The two anomaly cases we detect —
same task id producing a different delta on retry, and a delta arriving for a
source that is already completed — are handled in-place: the first by
rewriting, the second by logging a warning and skipping. Resumability is never
the cause of a pipeline failure.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import TYPE_CHECKING

import lmdb
import ray
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterable


_COMPLETED_DB = b"completed_sources"
_DEFAULT_MAP_SIZE = 1 << 30  # 1 GiB; sparse on Linux so effectively free
# Subdirectory (under the user-provided checkpoint dir) that holds the
# per-writer LMDB files. Hidden so it sits unobtrusively next to outputs.
METADATA_DIRNAME = ".nemo_curator_metadata"


@ray.remote(num_cpus=0, max_concurrency=1)
class ResumabilityActor:
    """Per-writer counter + LMDB owner.

    Workers fire ``apply_deltas`` via ``.remote()`` and do NOT ``ray.get`` the
    returned ref. The actor never raises; anomalies are logged and handled
    inline (see ``apply_deltas`` docstring).

    Spawned by ``Pipeline.run`` with ``lifetime="detached"`` so it survives
    executor-local ``ray.shutdown`` calls. ``Pipeline.run`` closes it
    explicitly at end-of-run.

    Writes go only to this actor's own file in the shared metadata directory;
    reads (``are_completed``) reflect the union of all writers' files as of
    this actor's startup, plus this actor's own progress.
    """

    def __init__(self, base_dir: str, map_size: int = _DEFAULT_MAP_SIZE, writer_id: str | None = None):
        # base_dir is the user-provided checkpoint directory; the per-writer
        # LMDB files live in <base_dir>/.nemo_curator_metadata/.
        self._dir = Path(base_dir).absolute() / METADATA_DIRNAME
        self._dir.mkdir(parents=True, exist_ok=True)
        # This actor's own file — the ONLY file it ever writes to. Keyed by
        # this writer's identity (default host+pid, unique among concurrently
        # running writers: distinct hosts, or distinct pids on one host). A
        # rerun whose pid recycles simply reopens and appends to its old file,
        # which is safe (sequential in time).
        wid = writer_id or f"{socket.gethostname()}-{os.getpid()}"
        self._path = str(self._dir / f"{wid}.mdb")
        self._env = lmdb.open(
            self._path,
            subdir=False,
            lock=False,  # sole writer of this file → no inter-process lock needed
            max_dbs=1,
            map_size=map_size,
            metasync=False,
            sync=True,
            readahead=False,
        )
        self._db = self._env.open_db(_COMPLETED_DB)
        self._pending: dict[str, int] = {}
        # Union of completed sources across ALL writer files in the dir.
        self._completed: set[str] = self._load_completed()
        # task_id -> delta we previously applied for this task.
        # Dual-purpose:
        #   1. Dedup: a Ray retry firing the same delta is a silent skip.
        #   2. Rewrite-on-conflict: a retry firing a *different* delta
        #      replaces the old delta. The pending counter is adjusted by
        #      `(-old_delta + new_delta)` so the latest observation wins.
        self._applied: dict[str, int] = {}

    def _read_completed_from(self, env: lmdb.Environment) -> set[str]:
        """Read the completed-source ids from an already-open LMDB env. Returns
        an empty set if this env has no completed-sources sub-db yet (a writer
        that has only recorded in-flight, not-yet-finished sources)."""
        try:
            db = env.open_db(_COMPLETED_DB)
        except lmdb.Error:
            return set()
        with env.begin() as txn, txn.cursor(db=db) as cur:
            return {k.decode() for k, _ in cur}

    def _load_completed(self) -> set[str]:
        """Union of completed source ids across every writer's LMDB file in the
        metadata dir (other writers' files read read-only; our own via the open
        write handle). A file that cannot be opened — e.g. it is mid-write by a
        live writer, or already open in THIS process during tests — is skipped
        with a warning rather than failing the run."""
        done = self._read_completed_from(self._env)  # our own (possibly reused) file
        for mdb in sorted(self._dir.glob("*.mdb")):
            if str(mdb) == self._path:
                continue
            try:
                env = lmdb.open(str(mdb), subdir=False, readonly=True, lock=False, max_dbs=1)
            except lmdb.Error as e:
                logger.warning(f"resumability: skipping unreadable checkpoint {mdb}: {e}")
                continue
            try:
                done |= self._read_completed_from(env)
            finally:
                env.close()
        return done

    # ------------------------------------------------------------ read

    def are_completed(self, source_ids: list[str]) -> list[bool]:
        """Returns a parallel bool list indicating which source_ids are
        already marked complete (and thus should be skipped on rerun). Reflects
        the union of all writers' files as of startup, plus this actor's own
        completions since."""
        return [sid in self._completed for sid in source_ids]

    # ------------------------------------------------------------ write

    def apply_deltas(self, per_task: list[tuple[str, str, int]]) -> None:
        """Apply per-task counter deltas. Workers call this fire-and-forget
        (no ``ray.get`` on the returned ref).

        Each tuple is ``(task_id, source_id, delta)``. Behavior:

        - ``task_id`` already in ``_applied`` with the same delta →
          silent skip (Ray retry idempotency).
        - ``task_id`` already in ``_applied`` with a different delta and
          ``source_id`` NOT in ``_completed`` → rewrite: adjust
          ``_pending[source_id]`` by ``(-old + new)`` and update the
          recorded delta. Reflects the latest observation.
        - ``task_id`` not in ``_applied`` and ``source_id`` already in
          ``_completed`` → log a loud warning **and remove the source
          from the completed set (in-memory + LMDB)** so it will be
          reprocessed on the next run. Same treatment when a different
          delta arrives for a task whose source is already completed.
          These two cases indicate a bug — the cleanest recovery is to
          un-complete the source rather than silently drop the update.
        - Otherwise → normal apply.

        Never raises.
        """
        newly_done: list[str] = []
        for task_id, sid, d in per_task:
            existing = self._applied.get(task_id)
            if existing is not None:
                if existing == d:
                    continue  # idempotent re-fire
                if sid in self._completed:
                    # Source already finalized but we're getting a different
                    # delta for one of its tasks — the source wasn't actually
                    # done. Un-complete it so it reruns next launch.
                    logger.warning(
                        f"resumability: task {task_id} delta changed from "
                        f"{existing} to {d} but source {sid!r} is already "
                        f"completed. Removing {sid!r} from the completed set "
                        f"so it will be reprocessed on the next run. Please "
                        f"file an issue at "
                        f"https://github.com/NVIDIA-NeMo/Curator if this is "
                        f"unexpected."
                    )
                    self._remove_from_completed(sid)
                    continue
                # Rewrite-on-conflict: the newest delta wins.
                self._applied[task_id] = d
                self._pending[sid] = self._pending.get(sid, 0) + (-existing) + d
            else:
                # New task id.
                if sid in self._completed:
                    logger.warning(
                        f"resumability: source {sid!r} got update for new "  # noqa: S608
                        f"task {task_id} (delta={d}) after being completed. "
                        f"Removing {sid!r} from the completed set so it will "
                        f"be reprocessed on the next run. Please file an "
                        f"issue at https://github.com/NVIDIA-NeMo/Curator."
                    )
                    self._remove_from_completed(sid)
                    continue
                self._applied[task_id] = d
                self._pending[sid] = self._pending.get(sid, 0) + d
            if self._pending[sid] == 0:
                newly_done.append(sid)
        if newly_done:
            self._persist_completed(newly_done)
            for sid in newly_done:
                self._completed.add(sid)
                self._pending.pop(sid, None)

    def _persist_completed(self, sids: Iterable[str]) -> None:
        with self._env.begin(write=True) as txn:
            for sid in sids:
                txn.put(sid.encode(), b"1", db=self._db, overwrite=True)

    def _remove_from_completed(self, sid: str) -> None:
        """Remove ``sid`` from the in-memory completed set and from our own
        LMDB file. Used when we detect that a source was prematurely marked
        complete (a late delta arrives after completion); the safest recovery
        is to un-complete so it reruns on next launch. Note: if ``sid`` was
        completed by a *different* writer's file we can't delete it there (we
        only ever write our own file), so it may reappear from the union on the
        next startup — acceptable for this rare anomaly path."""
        self._completed.discard(sid)
        with self._env.begin(write=True) as txn:
            txn.delete(sid.encode(), db=self._db)

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"failed to close LMDB env: {e}")
            self._env = None  # type: ignore[assignment]
