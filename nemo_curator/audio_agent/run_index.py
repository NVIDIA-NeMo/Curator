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

"""Rebuildable SQLite index over run records and artifacts.

The JSON files under ``.audio_agent_runs/`` remain the source of truth -- readable, diffable,
and the thing you can recover from. This index is a pure **cache** at
``.audio_agent_runs/index.db`` that turns the two queries reuse actually needs into O(1):

* probe a ``step_key`` (does prior work for this exact step exist?)
* find runs by dataset / stage / date (what has been done to this corpus?)

Nothing lives only in the DB. Every write is best-effort and every read tolerates a missing,
locked or corrupt database by falling back to the JSON scan, and ``reindex()`` rebuilds the
whole thing from disk. See ``REUSE_ARCHITECTURE.md``.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_curator.audio_agent.artifacts import Artifact
    from nemo_curator.audio_agent.contracts import RunRecord

_TABLES = """
CREATE TABLE IF NOT EXISTS artifacts (
    step_key    TEXT PRIMARY KEY,
    stage_ref   TEXT,
    stage_index INTEGER,
    dataset_key TEXT,
    uri         TEXT,
    kind        TEXT,
    rows_out    INTEGER,
    duration_sec REAL,
    status      TEXT,
    run_id      TEXT,
    created_at  TEXT,
    origin_config_hash TEXT,
    workspace_id       TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    config_hash   TEXT,
    semantic_hash TEXT,
    dataset_key   TEXT,
    data_source   TEXT,
    status        TEXT,
    elapsed_sec   REAL,
    created_at    TEXT,
    steps         TEXT
);
"""

# Kept apart from the tables and applied AFTER the column migration below: an index over a
# column added later cannot be created until that column exists, and a fresh database and an
# upgraded one have to reach the same place.
_INDEXES = """
CREATE INDEX IF NOT EXISTS artifacts_by_dataset ON artifacts (dataset_key);
CREATE INDEX IF NOT EXISTS artifacts_by_stage   ON artifacts (stage_ref);
CREATE INDEX IF NOT EXISTS artifacts_by_origin  ON artifacts (origin_config_hash);
CREATE INDEX IF NOT EXISTS runs_by_dataset      ON runs (dataset_key);
CREATE INDEX IF NOT EXISTS runs_by_semantic     ON runs (semantic_hash);
"""


# Columns added after the table shipped. ``CREATE TABLE IF NOT EXISTS`` is a no-op on an
# existing database, so without this an index built by an older build would reject every
# write and silently stop caching until someone ran ``reindex``.
_ADDED_ARTIFACT_COLUMNS = (("origin_config_hash", "TEXT"), ("workspace_id", "TEXT"))


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Bring an older artifacts table up to the current schema. Idempotent.

    ``executescript`` commits before it runs, so the following ``_INDEXES`` script is what
    makes these durable -- DDL has not been implicitly committed since Python 3.6, and a
    connection closed with the ALTER still open would roll it straight back.
    """
    present = {row["name"] for row in conn.execute("PRAGMA table_info(artifacts)")}
    for column, kind in _ADDED_ARTIFACT_COLUMNS:
        if column not in present:
            # Interpolated, not bound: sqlite takes no parameters in DDL. Both halves come
            # from the fixed tuple above, never from a record.
            conn.execute(f"ALTER TABLE artifacts ADD COLUMN {column} {kind}")


def index_path() -> str:
    from nemo_curator.audio_agent.run_store import runs_dir

    return os.path.join(runs_dir(), "index.db")


@contextlib.contextmanager
def _db(*, write: bool = False):  # noqa: ANN202 - a contextmanager over sqlite3.Connection
    """Open the index, creating it on demand. Yields ``None`` if it is unusable.

    Structured as open-then-use so that exactly one ``yield`` runs on every path. Opening and
    using used to share one ``try``, which meant a failure at or after the ``yield`` -- a locked
    database on ``commit()`` being the realistic one -- ran the fallback ``yield None`` as a
    SECOND yield, and a contextmanager that yields twice raises ``RuntimeError``. The cache is
    advisory, so its failures must stay inside it; callers fall back to reading JSON.
    """
    path = index_path()
    if write:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    elif not os.path.isfile(path):
        yield None
        return
    try:
        conn = sqlite3.connect(path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.executescript(_TABLES)
        _add_missing_columns(conn)
        conn.executescript(_INDEXES)
    except sqlite3.Error:
        yield None
        return
    try:
        # Writers commit explicitly so they can truthfully report whether the
        # cache record became durable. Retrying implicitly here could commit a
        # transaction after its caller already reported failure.
        yield conn
    except sqlite3.Error:
        pass  # the caller's own use of the handle failed; an advisory cache swallows that
    finally:
        with contextlib.suppress(sqlite3.Error):
            conn.close()


def index_artifact(artifact: Artifact) -> bool:
    """Upsert one artifact; return whether the cache write was committed."""
    with _db(write=True) as conn:
        if conn is None:
            return False
        try:
            conn.execute(
                "INSERT OR REPLACE INTO artifacts ("
                "step_key, stage_ref, stage_index, dataset_key, uri, kind, rows_out, "
                "duration_sec, status, run_id, created_at, origin_config_hash, workspace_id"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    artifact.step_key,
                    artifact.stage_ref,
                    artifact.stage_index,
                    artifact.dataset_key,
                    artifact.uri,
                    artifact.kind,
                    artifact.rows_out,
                    artifact.duration_sec,
                    artifact.status,
                    artifact.run_id,
                    artifact.created_at,
                    artifact.origin_config_hash,
                    artifact.workspace_id,
                ),
            )
            conn.commit()
        except sqlite3.Error:
            return False
    return True


def index_run(record: RunRecord) -> bool:
    """Upsert one run record; return whether the cache write was committed."""
    with _db(write=True) as conn:
        if conn is None:
            return False
        try:
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    record.run_id,
                    record.config_hash,
                    record.semantic_hash,
                    record.dataset_key,
                    record.data_source,
                    record.status,
                    record.elapsed_sec,
                    record.created_at,
                    json.dumps(list(record.steps or [])),
                ),
            )
            conn.commit()
        except sqlite3.Error:
            return False
    return True


def probe_step(step_key: str) -> dict[str, Any] | None:
    """O(1) "has this step been done?", for a caller that wants the answer without the record.

    Not what the reuse scan uses: ``reuse.scan`` goes through ``artifacts.lookup``, which must
    read the JSON record anyway to decide validity (marker, digest, dataset, code version), so
    it would pay for the row and then discard it. This stays because the index is the cheaper
    answer whenever existence alone is the question.
    """
    with _db() as conn:
        if conn is None:
            return None
        try:
            row = conn.execute("SELECT * FROM artifacts WHERE step_key = ?", (step_key,)).fetchone()
        except sqlite3.Error:
            return None
        return dict(row) if row else None


def _json_file_count(directory: str) -> int:
    """Number of ``*.json`` record files on disk -- the source-of-truth cardinality the
    best-effort SQLite cache must match. A smaller cache count means writes were dropped
    (a lock/error swallowed by the best-effort ``index_*`` calls) and the cache is partial.
    """
    try:
        return sum(1 for name in os.listdir(directory) if name.endswith(".json"))
    except OSError:
        return 0


def _run_record_count() -> int:
    from nemo_curator.audio_agent.run_store import runs_dir

    return _json_file_count(runs_dir())


def _artifact_record_count() -> int:
    from nemo_curator.audio_agent.artifacts import artifacts_dir

    return _json_file_count(artifacts_dir())


def find_runs(
    *,
    dataset_key: str | None = None,
    semantic_hash: str | None = None,
    data_source: str | None = None,
    since: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Runs matching a dataset / semantic identity / source path / date, newest first.

    ``data_source`` matches the resolved source path regardless of recipe or dataset key. It is
    the axis the step-key matchers structurally cannot use: a changed source stage or a changed
    corpus moves every key, but the folder a run READ is the same folder, and that is what a
    "you curated this before" notice is anchored on. The column is unindexed -- the same folder
    is asked about once per scan, not per row -- so the ``ORDER BY`` + ``LIMIT`` bound the cost.
    """
    where: list[str] = []
    params: list[Any] = []
    if dataset_key:
        where.append("dataset_key = ?")
        params.append(dataset_key)
    if semantic_hash:
        where.append("semantic_hash = ?")
        params.append(semantic_hash)
    if data_source:
        where.append("data_source = ?")
        params.append(data_source)
    if since:
        where.append("created_at >= ?")
        params.append(since)
    sql = "SELECT * FROM runs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(int(limit))
    with _db() as conn:
        if conn is None:
            return _find_runs_in_json(
                dataset_key=dataset_key,
                semantic_hash=semantic_hash,
                data_source=data_source,
                since=since,
                limit=limit,
            )
        try:
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
            index_total = int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
        except sqlite3.Error:
            return _find_runs_in_json(
                dataset_key=dataset_key,
                semantic_hash=semantic_hash,
                data_source=data_source,
                since=since,
                limit=limit,
            )
    # The cache is best-effort: fall back to the authoritative JSON scan whenever it is
    # empty OR its cardinality disagrees with the on-disk records -- otherwise a PARTIAL
    # index (some writes dropped) silently under-reports real runs until the next reindex().
    if not rows or index_total != _run_record_count():
        return _find_runs_in_json(
            dataset_key=dataset_key,
            semantic_hash=semantic_hash,
            data_source=data_source,
            since=since,
            limit=limit,
        )
    return rows


def find_artifacts(
    *,
    dataset_key: str | None = None,
    stage_ref: str | None = None,
    since: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Artifacts for a dataset and/or a stage, newest first."""
    where: list[str] = []
    params: list[Any] = []
    if dataset_key:
        where.append("dataset_key = ?")
        params.append(dataset_key)
    if stage_ref:
        where.append("stage_ref = ?")
        params.append(stage_ref)
    if since:
        where.append("created_at >= ?")
        params.append(since)
    sql = "SELECT * FROM artifacts"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(int(limit))
    with _db() as conn:
        if conn is None:
            return _find_artifacts_in_json(
                dataset_key=dataset_key,
                stage_ref=stage_ref,
                since=since,
                limit=limit,
            )
        try:
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
            index_total = int(conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0])
        except sqlite3.Error:
            return _find_artifacts_in_json(
                dataset_key=dataset_key,
                stage_ref=stage_ref,
                since=since,
                limit=limit,
            )
    # Same reconciliation as find_runs: a partial artifact cache must not hide real
    # artifacts, so any cardinality mismatch with the on-disk records forces the JSON scan.
    if not rows or index_total != _artifact_record_count():
        return _find_artifacts_in_json(
            dataset_key=dataset_key,
            stage_ref=stage_ref,
            since=since,
            limit=limit,
        )
    return rows


def _bounded(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Apply SQLite-compatible LIMIT behavior to JSON fallback rows."""
    count = int(limit)
    if count < 0:
        return rows
    return rows[:count]


def _find_runs_in_json(
    *,
    dataset_key: str | None,
    semantic_hash: str | None,
    data_source: str | None = None,
    since: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Source-of-truth fallback for :func:`find_runs`."""
    from nemo_curator.audio_agent import run_store

    rows: list[dict[str, Any]] = []
    for summary in run_store.list_runs():
        record = run_store.load(str(summary.get("run_id") or ""))
        if record is None:
            continue
        if dataset_key and record.dataset_key != dataset_key:
            continue
        if semantic_hash and record.semantic_hash != semantic_hash:
            continue
        if data_source and record.data_source != data_source:
            continue
        # ``str(... or "")`` because a record on disk can carry an explicit null and
        # ``from_dict`` keeps it, so comparing to ``since`` raised TypeError out of the JSON
        # scan -- the leg that runs exactly when SQLite is unavailable, so the degraded path was
        # the fragile one. An absent timestamp now filters as "", matching SQLite's NULL.
        if since and str(record.created_at or "") < since:
            continue
        rows.append(
            {
                "run_id": record.run_id,
                "config_hash": record.config_hash,
                "semantic_hash": record.semantic_hash,
                "dataset_key": record.dataset_key,
                "data_source": record.data_source,
                "status": record.status,
                "elapsed_sec": record.elapsed_sec,
                "created_at": record.created_at,
                "steps": json.dumps(list(record.steps or [])),
            }
        )
    rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    return _bounded(rows, limit)


def _find_artifacts_in_json(
    *,
    dataset_key: str | None,
    stage_ref: str | None,
    since: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Source-of-truth fallback for :func:`find_artifacts`."""
    from nemo_curator.audio_agent import artifacts as art_mod

    rows = [
        {
            "step_key": artifact.step_key,
            "stage_ref": artifact.stage_ref,
            "stage_index": artifact.stage_index,
            "dataset_key": artifact.dataset_key,
            "uri": artifact.uri,
            "kind": artifact.kind,
            "rows_out": artifact.rows_out,
            "duration_sec": artifact.duration_sec,
            "status": artifact.status,
            "run_id": artifact.run_id,
            "created_at": artifact.created_at,
        }
        for artifact in art_mod.list_artifacts()
        if (not dataset_key or artifact.dataset_key == dataset_key)
        and (not stage_ref or artifact.stage_ref == stage_ref)
        # Same null-timestamp coercion as the run scan above, and as this function's own sort.
        and (not since or str(artifact.created_at or "") >= since)
    ]
    rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    return _bounded(rows, limit)


def dataset_keys(*, limit: int = 50) -> list[str]:
    """Distinct source datasets that have artifacts, most recently written first."""
    sql = (
        "SELECT dataset_key, MAX(created_at) AS last_at FROM artifacts "
        "WHERE dataset_key != '' GROUP BY dataset_key ORDER BY last_at DESC LIMIT ?"
    )
    with _db() as conn:
        if conn is None:
            return []
        try:
            return [r["dataset_key"] for r in conn.execute(sql, (int(limit),)).fetchall()]
        except sqlite3.Error:
            return []


def reindex() -> dict[str, Any]:
    """Rebuild the whole index from the JSON records (the source of truth)."""
    from nemo_curator.audio_agent import artifacts as art_mod
    from nemo_curator.audio_agent import run_store

    path = index_path()
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError as exc:
            return {
                "status": "error",
                "index": path,
                "runs_indexed": 0,
                "artifacts_indexed": 0,
                "runs_failed": 0,
                "artifacts_failed": 0,
                "errors": [f"could not replace the existing index: {exc}"],
            }

    arts = art_mod.list_artifacts()
    artifacts_indexed = sum(1 for art in arts if index_artifact(art))
    run_records = []
    for summary in run_store.list_runs():
        rec = run_store.load(summary["run_id"])
        if rec is not None:
            run_records.append(rec)
    runs_indexed = sum(1 for record in run_records if index_run(record))
    artifacts_failed = len(arts) - artifacts_indexed
    runs_failed = len(run_records) - runs_indexed
    status = "ok" if not artifacts_failed and not runs_failed else "error"
    result = {
        "status": status,
        "index": path,
        "runs_indexed": runs_indexed,
        "artifacts_indexed": artifacts_indexed,
        "runs_failed": runs_failed,
        "artifacts_failed": artifacts_failed,
    }
    if status == "error":
        result["errors"] = ["one or more JSON source records could not be committed to the index"]
    return result
