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

"""Regressions for durable publication and the rebuildable run index."""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
from collections.abc import Iterator  # noqa: TC003

import pytest

from nemo_curator.audio_agent import artifacts, run_index, run_store
from nemo_curator.audio_agent.artifacts import Artifact
from nemo_curator.audio_agent.contracts import RunRecord


def test_publish_does_not_register_an_artifact_when_marker_write_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = Artifact(
        step_key="step-marker-failure",
        uri="/unused/output.jsonl",
        kind="manifest",
    )
    saved: list[Artifact] = []
    monkeypatch.setattr(artifacts, "measure", lambda *_args: (3, 17))
    monkeypatch.setattr(artifacts, "content_digest", lambda *_args: "sha256:payload")
    monkeypatch.setattr(artifacts, "write_marker", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(artifacts, "save", saved.append)

    with pytest.raises(OSError, match="completion marker"):
        artifacts.publish(artifact)

    assert saved == []


@contextlib.contextmanager
def _unavailable_db(*, write: bool = False) -> Iterator[None]:
    del write
    yield None


def test_filtered_queries_fall_back_to_json_records(
    tmp_path,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path))
    run_store.save(
        RunRecord(
            run_id="run-json-fallback",
            config_hash="cfg",
            semantic_hash="semantic-a",
            dataset_key="stat:dataset-a",
            data_source="/data/a.jsonl",
            status="completed",
            elapsed_sec=2.5,
            created_at="2026-07-28T12:00:00Z",
            steps=["step-json-fallback"],
        )
    )
    artifacts.save(
        Artifact(
            step_key="step-json-fallback",
            stage_ref="ManifestWriterStage",
            stage_index=2,
            dataset_key="stat:dataset-a",
            uri="/outputs/a.jsonl",
            kind="manifest",
            rows_out=9,
            duration_sec=1.25,
            run_id="run-json-fallback",
            created_at="2026-07-28T12:01:00Z",
        )
    )
    monkeypatch.setattr(run_index, "_db", _unavailable_db)

    runs = run_index.find_runs(
        dataset_key="stat:dataset-a",
        semantic_hash="semantic-a",
        since="2026-07-28T00:00:00Z",
    )
    assert [row["run_id"] for row in runs] == ["run-json-fallback"]
    assert runs[0]["steps"] == '["step-json-fallback"]'
    assert run_index.find_runs(dataset_key="stat:other") == []
    assert run_index.find_runs(since="2026-07-29T00:00:00Z") == []

    found = run_index.find_artifacts(
        dataset_key="stat:dataset-a",
        stage_ref="ManifestWriterStage",
        since="2026-07-28T00:00:00Z",
    )
    assert [row["step_key"] for row in found] == ["step-json-fallback"]
    assert run_index.find_artifacts(stage_ref="OtherStage") == []
    assert run_index.find_artifacts(since="2026-07-29T00:00:00Z") == []


def test_reindex_reports_only_successfully_committed_records(
    tmp_path,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path))
    run_store.save(
        RunRecord(
            run_id="run-reindex-failure",
            dataset_key="stat:dataset-a",
            created_at="2026-07-28T12:00:00Z",
        )
    )
    artifacts.save(
        Artifact(
            step_key="step-reindex-failure",
            dataset_key="stat:dataset-a",
            created_at="2026-07-28T12:01:00Z",
        )
    )
    monkeypatch.setattr(run_index, "index_run", lambda _record: False)
    monkeypatch.setattr(run_index, "index_artifact", lambda _artifact: False)

    result = run_index.reindex()

    assert result["status"] == "error"
    assert result["runs_indexed"] == 0
    assert result["artifacts_indexed"] == 0
    assert result["runs_failed"] == 1
    assert result["artifacts_failed"] == 1


def test_index_writes_report_a_failed_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CommitFailure:
        def execute(self, *_args, **_kwargs) -> None:
            return None

        def commit(self) -> None:
            raise sqlite3.OperationalError("database is locked")  # noqa: EM101

    @contextlib.contextmanager
    def locked_db(*, write: bool = False) -> Iterator[CommitFailure]:
        del write
        yield CommitFailure()

    monkeypatch.setattr(run_index, "_db", locked_db)

    assert run_index.index_artifact(Artifact(step_key="step")) is False
    assert run_index.index_run(RunRecord(run_id="run")) is False


def test_partial_index_reconciles_against_json_truth(
    tmp_path,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A dropped best-effort index write leaves the cache with FEWER rows than the JSON
    # truth. find_runs / find_artifacts must reconcile against the on-disk record count and
    # fall back to the authoritative scan -- NOT only when the cache is empty (the old bug,
    # where a partial index silently under-reported real runs/artifacts until a reindex).
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path))

    # Record + artifact A are indexed normally (a real SQLite cache is created in tmp_path).
    run_store.save(
        RunRecord(run_id="run-a", dataset_key="stat:d", status="completed", created_at="2026-08-01T00:00:00Z")
    )
    artifacts.save(
        Artifact(
            step_key="step-a",
            stage_ref="ManifestWriterStage",
            dataset_key="stat:d",
            uri="/out/a.jsonl",
            kind="manifest",
            run_id="run-a",
            created_at="2026-08-01T00:01:00Z",
        )
    )

    # Record + artifact B: their index writes are DROPPED, so they exist in JSON but not
    # the cache -> the cache is now partial (1 of 2).
    monkeypatch.setattr(run_index, "index_run", lambda _record: False)
    monkeypatch.setattr(run_index, "index_artifact", lambda _artifact: False)
    run_store.save(
        RunRecord(run_id="run-b", dataset_key="stat:d", status="completed", created_at="2026-08-02T00:00:00Z")
    )
    artifacts.save(
        Artifact(
            step_key="step-b",
            stage_ref="ManifestWriterStage",
            dataset_key="stat:d",
            uri="/out/b.jsonl",
            kind="manifest",
            run_id="run-b",
            created_at="2026-08-02T00:01:00Z",
        )
    )

    # The partial cache alone would return only {A}; reconciliation forces the JSON scan.
    assert {row["run_id"] for row in run_index.find_runs(dataset_key="stat:d")} == {"run-a", "run-b"}
    assert {row["step_key"] for row in run_index.find_artifacts(dataset_key="stat:d")} == {"step-a", "step-b"}


def _null_out_created_at(path: str) -> None:
    """Rewrite a stored record so its timestamp is an explicit JSON ``null``."""
    with open(path, encoding="utf-8") as handle:
        stored = json.load(handle)
    stored["created_at"] = None
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stored, handle)


def test_a_record_with_a_null_timestamp_does_not_break_the_json_scan(
    tmp_path,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``RunRecord.created_at`` is annotated ``str`` and defaults to ``""``, but ``from_dict``
    keeps whatever the file holds, so a record written with an explicit null loads as ``None``.

    Comparing that to ``since`` raised ``TypeError`` out of the JSON scan. That scan is the leg
    that runs exactly when the SQLite cache is missing or disagrees with the records on disk --
    so the fallback, the thing meant to be authoritative, was the one that fell over. Both
    ``find_runs`` and ``find_artifacts`` had it, each one line above a sort that already
    coerced the same field.
    """
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path))
    run_store.save(
        RunRecord(run_id="run-null-ts", dataset_key="stat:d", status="completed", created_at="2026-08-15T00:00:00Z")
    )
    run_store.save(
        RunRecord(run_id="run-dated", dataset_key="stat:d", status="completed", created_at="2026-08-20T00:00:00Z")
    )
    _null_out_created_at(str(run_store.record_path("run-null-ts")))

    artifacts.save(
        Artifact(
            step_key="step-null-ts",
            stage_ref="ManifestWriterStage",
            dataset_key="stat:d",
            uri=str(tmp_path),
            status="complete",
            created_at="2026-08-15T00:00:00Z",
        )
    )
    for name in os.listdir(artifacts.artifacts_dir()):
        if name.endswith(".json"):
            _null_out_created_at(os.path.join(artifacts.artifacts_dir(), name))

    monkeypatch.setattr(run_index, "_db", _unavailable_db)

    # Excluded from a window, exactly as SQLite excludes NULL from `created_at >= ?` ...
    assert [row["run_id"] for row in run_index.find_runs(since="2026-08-01T00:00:00Z")] == ["run-dated"]
    assert run_index.find_artifacts(since="2026-08-01T00:00:00Z") == []
    # ... but still a record, and still listed when nothing is being filtered on.
    assert {row["run_id"] for row in run_index.find_runs()} == {"run-dated", "run-null-ts"}
    assert [row["step_key"] for row in run_index.find_artifacts()] == ["step-null-ts"]


def test_the_sqlite_index_and_the_json_scan_answer_alike(
    tmp_path,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two implementations of one query. The index is an advisory cache over the JSON records,
    and ``find_runs`` silently prefers whichever it trusts -- so a divergence would not surface
    as an error, it would surface as reuse deciding differently depending on whether the cache
    happened to be warm.
    """
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path))
    for i in range(6):
        run_store.save(
            RunRecord(
                run_id=f"run-{i}",
                semantic_hash="sem-a" if i % 2 == 0 else "sem-b",
                dataset_key="stat:d1" if i < 4 else "stat:d2",
                status="completed",
                created_at=f"2026-08-{10 + i:02d}T12:00:00Z",
            )
        )

    # Only queries that MATCH something: an empty index result deliberately re-checks the JSON
    # records ("fall back whenever it is empty"), so no such query can be served by SQLite
    # alone and none can take part in the trapped leg below.
    queries: list[dict[str, object]] = [
        {},
        {"dataset_key": "stat:d1"},
        {"semantic_hash": "sem-a"},
        {"dataset_key": "stat:d1", "semantic_hash": "sem-a"},
        {"since": "2026-08-12T00:00:00Z"},
        {"limit": 2},
    ]

    # The comparison is only worth anything if the first leg really is SQLite: with the JSON
    # scan booby-trapped, a query that still answers proves the cache served it.
    def _explode(**_kwargs: object) -> list[dict[str, object]]:
        msg = "the JSON scan ran; this leg was supposed to be served by the SQLite index"
        raise AssertionError(msg)

    monkeypatch.setattr(run_index, "_find_runs_in_json", _explode)
    via_index = [run_index.find_runs(**query) for query in queries]  # type: ignore[arg-type]
    monkeypatch.undo()

    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(run_index, "_db", _unavailable_db)
    via_json = [run_index.find_runs(**query) for query in queries]  # type: ignore[arg-type]

    for query, indexed, scanned in zip(queries, via_index, via_json, strict=True):
        assert [row["run_id"] for row in indexed] == [row["run_id"] for row in scanned], query
        assert [sorted(row) for row in indexed] == [sorted(row) for row in scanned], (
            f"{query}: the two legs return different field sets"
        )

    assert run_index.find_runs(dataset_key="nope") == [], "a miss stays a miss on either leg"
