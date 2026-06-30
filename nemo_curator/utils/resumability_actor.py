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
"""Per-writer LMDB owner tracking per-source completion for resumability.

LMDB can't be safely shared by writers across hosts (its lock lives in an
mmap'd file not shared on a networked FS), so each actor writes ONLY its own
``<dir>/<host>-<pid>.mdb`` and on startup reads the UNION of completed sources
across every ``*.mdb`` in the dir. A rerun thus skips everything any prior
writer finished — letting the tasks of a SLURM array share one checkpoint dir.

``apply_deltas`` is fire-and-forget and never raises; see its docstring for the
dedup/rewrite/anomaly rules.
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
    """Per-writer counter + LMDB owner. Spawned by ``Pipeline.run`` with
    ``lifetime="detached"`` and closed at end-of-run; ``apply_deltas`` is
    fire-and-forget and never raises."""

    def __init__(self, base_dir: str, map_size: int = _DEFAULT_MAP_SIZE, writer_id: str | None = None):
        # Per-writer LMDB files live under <base_dir>/.nemo_curator_metadata/.
        self._dir = Path(base_dir).absolute() / METADATA_DIRNAME
        self._dir.mkdir(parents=True, exist_ok=True)
        # The ONLY file this actor writes, keyed by writer id (default host-pid,
        # unique across concurrent writers; a pid-recycled rerun safely reuses it).
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
        # task_id -> last delta applied: same delta = dedup skip; different = rewrite.
        self._applied: dict[str, int] = {}

    def _read_completed_from(self, env: lmdb.Environment) -> set[str]:
        """Completed-source ids from an open LMDB env (empty if it has no completed-sources db yet)."""
        try:
            db = env.open_db(_COMPLETED_DB)
        except lmdb.Error:
            return set()
        with env.begin() as txn, txn.cursor(db=db) as cur:
            return {k.decode() for k, _ in cur}

    def _load_completed(self) -> set[str]:
        """Union of completed sources across all writer files; unreadable files
        (mid-write, or open in-process during tests) are skipped with a warning."""
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
        """Parallel bool list: which source_ids are complete (skip on rerun)."""
        return [sid in self._completed for sid in source_ids]

    def wait(self) -> None:
        """No-op the caller ``ray.get``s after spawning the actor: it blocks until
        ``__init__`` (the checkpoint scan) has finished and surfaces any startup
        error (e.g. an LMDB open failure) before the pipeline begins."""

    # ------------------------------------------------------------ write

    def apply_deltas(self, per_task: list[tuple[str, str, int]]) -> None:
        """Apply per-task counter deltas (fire-and-forget; no ``ray.get``).

        Each tuple is ``(task_id, source_id, delta)``:
        - seen ``task_id``, same delta → skip (Ray-retry idempotency).
        - seen ``task_id``, different delta → rewrite ``_pending`` by ``-old+new``.
        - any delta for an already-completed source → warn and un-complete it
          (in-memory + LMDB) so it reprocesses next run (indicates a bug).
        - else → apply; persist the source when its counter hits 0.

        Never raises.
        """
        if self._env is None:
            return  # closing/closed: drop late fire-and-forget deltas (durable rows already on disk)
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
        """Un-complete ``sid`` (in-memory + our LMDB file) so it reruns. If a
        *different* writer completed it, that entry can't be removed and may
        reappear from the union next startup — acceptable for this rare path."""
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


# --- actor lifecycle ---------------------------------------------------------
# These do NOT manage the Ray session: the pipeline wraps create/shutdown in
# ``with ray.init()`` against a pre-existing cluster (RayClient), keeping that
# init un-nested from the executor's own ray.init so the executor's env vars
# still propagate. The actor is detached + namespaced (namespace == name, like
# id_generator) so it survives the executor's separate ray.init/shutdown and
# workers find it by (name, namespace).


def create_resumability_actor(checkpoint_path: str) -> None:
    """Spawn the detached resumability actor and block until it has scanned the
    checkpoint dir (so the first ``apply_deltas``/``are_completed`` works, and any
    LMDB startup error surfaces here). Must be called with an active Ray
    connection — the pipeline wraps it in ``with ray.init()``."""
    from nemo_curator.utils.resumability_client import ACTOR_NAME

    actor = ResumabilityActor.options(  # type: ignore[attr-defined]
        name=ACTOR_NAME, namespace=ACTOR_NAME, lifetime="detached", get_if_exists=True
    ).remote(checkpoint_path)
    ray.get(actor.wait.remote())


def shutdown_resumability_actor() -> None:
    """Flush and kill the detached actor. ``ray.kill`` always runs even if
    ``close`` fails/times out, so a stale actor can't leak into the next run.
    A no-op if Ray is already down (the actor then dies with the cluster; its
    LMDB rows are durable, written sync per delta)."""
    from nemo_curator.utils.resumability_client import ACTOR_NAME

    if not ray.is_initialized():
        return
    try:
        actor = ray.get_actor(name=ACTOR_NAME, namespace=ACTOR_NAME)
    except ValueError:
        return
    try:
        ray.get(actor.close.remote(), timeout=30)  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"resumability actor close failed: {e}")
    ray.kill(actor)
