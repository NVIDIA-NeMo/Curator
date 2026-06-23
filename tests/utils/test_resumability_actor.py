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
"""Unit tests for :class:`ResumabilityActor` (counter math, dedup,
rewrite-on-conflict, LMDB persistence). Instantiates the actor class directly
(no ``@ray.remote``), so no live Ray cluster is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from nemo_curator.utils.resumability_actor import ResumabilityActor

if TYPE_CHECKING:
    from pathlib import Path


def _new_actor(tmp_path: Path, writer_id: str | None = None) -> ResumabilityActor:
    """Bypass ``@ray.remote`` and instantiate the actor class directly.

    Ray's ``@ray.remote`` decorator stashes the original class on
    ``__ray_metadata__.modified_class``. ``tmp_path`` is the checkpoint
    directory; the actor keeps its LMDB file under
    ``tmp_path/.nemo_curator_metadata/``. ``writer_id`` distinguishes writers
    sharing that directory (defaults to host+pid in production); pass distinct
    ids to simulate concurrent runs / SLURM-array tasks.
    """
    cls = ResumabilityActor.__ray_metadata__.modified_class  # type: ignore[attr-defined]
    return cls(str(tmp_path), writer_id=writer_id)


class TestApplyDeltasCounterMath:
    def test_source_emit_increments_pending(self, tmp_path: Path) -> None:
        actor = _new_actor(tmp_path)
        actor.apply_deltas([("h0", "0", +1), ("h1", "1", +1)])
        assert actor._pending == {"0": 1, "1": 1}
        assert actor._completed == set()
        actor.close()

    def test_counter_reaches_zero_persists_to_lmdb(self, tmp_path: Path) -> None:
        actor = _new_actor(tmp_path)
        actor.apply_deltas([("h0", "0", +1)])
        actor.apply_deltas([("h_sink", "0", -1)])
        assert actor._completed == {"0"}
        assert "0" not in actor._pending
        actor.close()

        # Reopen the actor and confirm "0" survives in LMDB.
        actor2 = _new_actor(tmp_path)
        assert actor2._completed == {"0"}
        actor2.close()

    def test_nonsink_real_task_is_zero_delta(self, tmp_path: Path) -> None:
        actor = _new_actor(tmp_path)
        actor.apply_deltas([("h0", "0", +1)])
        actor.apply_deltas([("h_passthrough", "0", 0)])
        assert actor._pending == {"0": 1}
        assert actor._completed == set()
        actor.close()

    def test_nonetask_decrements(self, tmp_path: Path) -> None:
        actor = _new_actor(tmp_path)
        actor.apply_deltas([("h0", "0", +1)])
        actor.apply_deltas([("h_filter", "0", -1)])
        assert actor._completed == {"0"}
        actor.close()

    def test_fanout_grows_counter(self, tmp_path: Path) -> None:
        actor = _new_actor(tmp_path)
        actor.apply_deltas([("h0", "0", +1)])
        # Fan-out 1→3 emits delta = (3-1) = +2 on the parent's source.
        actor.apply_deltas([("h_fanout", "0", +2)])
        assert actor._pending == {"0": 3}
        actor.close()


class TestDedupAndRewrite:
    def test_same_task_same_delta_is_idempotent(self, tmp_path: Path) -> None:
        actor = _new_actor(tmp_path)
        actor.apply_deltas([("h0", "0", +1)])
        actor.apply_deltas([("h_t", "0", -1)])
        # Second identical fire — should be a no-op (Ray retry idempotency).
        actor.apply_deltas([("h_t", "0", -1)])
        assert actor._completed == {"0"}
        actor.close()

    def test_same_task_different_delta_rewrites(self, tmp_path: Path) -> None:
        """When a Ray retry fires a different delta for the same task hash,
        the actor adjusts pending by (-old + new) so the latest observation
        wins. Never raises."""
        actor = _new_actor(tmp_path)
        actor.apply_deltas([("h0", "0", +1)])
        # First the worker says delta=0 (real Task passed through).
        actor.apply_deltas([("h_t", "0", 0)])
        assert actor._pending == {"0": 1}

        # Retry says delta=-1 (NoneTask this time). Rewrite: pending += -0 + -1.
        actor.apply_deltas([("h_t", "0", -1)])
        assert actor._completed == {"0"}
        # And the recorded delta is updated to the new value.
        assert actor._applied["h_t"] == -1
        actor.close()

    def test_rewrite_does_not_raise(self, tmp_path: Path) -> None:
        """apply_deltas never raises; rewrite is silent."""
        actor = _new_actor(tmp_path)
        actor.apply_deltas([("h0", "0", +1)])
        # Multiple conflicting deltas for the same task: should not raise.
        actor.apply_deltas([("h_t", "0", 0)])
        actor.apply_deltas([("h_t", "0", +5)])
        actor.apply_deltas([("h_t", "0", -1)])
        # Final state reflects the last delta.
        assert actor._applied["h_t"] == -1
        actor.close()


class TestUncompleteOnAnomaly:
    def test_new_task_after_source_completed_warns_and_uncompletes(self, tmp_path: Path) -> None:
        """If a delta arrives for a never-seen task on an already-completed
        source, the source wasn't actually done. Un-complete it (in-memory
        and in LMDB) so it reruns next launch."""
        actor = _new_actor(tmp_path)
        actor.apply_deltas([("h0", "0", +1), ("h_t", "0", -1)])
        assert actor._completed == {"0"}

        with patch("nemo_curator.utils.resumability_actor.logger") as mock_logger:
            actor.apply_deltas([("h_late", "0", -1)])
        mock_logger.warning.assert_called_once()
        warn_msg = mock_logger.warning.call_args[0][0]
        assert "Removing" in warn_msg
        assert "completed set" in warn_msg

        # Source has been removed from the in-memory completed set.
        assert "0" not in actor._completed
        # And from LMDB — reopen and confirm.
        actor.close()
        actor2 = _new_actor(tmp_path)
        assert "0" not in actor2._completed
        actor2.close()

    def test_rewrite_attempt_after_source_completed_warns_and_uncompletes(self, tmp_path: Path) -> None:
        actor = _new_actor(tmp_path)
        actor.apply_deltas([("h0", "0", +1)])
        actor.apply_deltas([("h_t", "0", -1)])
        assert actor._completed == {"0"}

        # Same task tries to rewrite to a different delta after completion.
        with patch("nemo_curator.utils.resumability_actor.logger") as mock_logger:
            actor.apply_deltas([("h_t", "0", 0)])
        mock_logger.warning.assert_called_once()
        warn_msg = mock_logger.warning.call_args[0][0]
        assert "Removing" in warn_msg

        # Source has been uncompleted.
        assert "0" not in actor._completed
        actor.close()
        actor2 = _new_actor(tmp_path)
        assert "0" not in actor2._completed
        actor2.close()

    def test_apply_deltas_never_raises(self, tmp_path: Path) -> None:
        """The whole point of removing the error machinery — no path through
        apply_deltas should raise."""
        actor = _new_actor(tmp_path)
        # Throw lots of weird stuff at it.
        actor.apply_deltas([("h0", "0", +1)])
        actor.apply_deltas([("h_t", "0", -1)])  # completes 0
        actor.apply_deltas([("h_t", "0", +5)])  # rewrite on completed source: warn + uncomplete
        actor.apply_deltas([("h_new", "0", -1)])  # new hash, source no longer in completed
        actor.apply_deltas([("h1", "1", +1), ("h_t1", "1", -5)])  # negative pending
        # Reached this line without raising — pass.
        actor.close()


class TestAreCompleted:
    def test_returns_parallel_bool_list(self, tmp_path: Path) -> None:
        actor = _new_actor(tmp_path)
        actor.apply_deltas([("h0", "0", +1), ("h1", "1", +1)])
        actor.apply_deltas([("h_t", "0", -1)])
        assert actor.are_completed(["0", "1", "unknown"]) == [True, False, False]
        actor.close()

    def test_loads_from_lmdb_on_construction(self, tmp_path: Path) -> None:
        actor = _new_actor(tmp_path)
        actor.apply_deltas([("h_a", "a", +1), ("h_t", "a", -1)])
        assert actor._completed == {"a"}
        actor.close()

        actor2 = _new_actor(tmp_path)
        assert actor2.are_completed(["a", "b"]) == [True, False]
        actor2.close()


class TestLifecycle:
    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        actor = _new_actor(tmp_path)
        actor.close()
        actor.close()  # second close is a no-op

    def test_one_lmdb_write_per_completed_source(self, tmp_path: Path) -> None:
        """Sanity-check the 'write only when a counter hits zero' contract:
        a still-pending source is never persisted; once it completes it is.

        We verify via close/reopen rather than a concurrent second reader:
        lmdb refuses to open the same env file twice in one process, and in
        production a single detached actor owns each checkpoint file anyway.
        """
        actor = _new_actor(tmp_path)
        actor.apply_deltas([("h0", "0", +1)])
        # Source 0 is pending (counter != 0) — nothing recorded as completed.
        assert actor._completed == set()
        # Counter hits zero — now it's recorded.
        actor.apply_deltas([("h_t", "0", -1)])
        assert actor._completed == {"0"}
        actor.close()

        # A fresh actor loads exactly the one completed source from LMDB.
        actor_c = _new_actor(tmp_path)
        assert actor_c._completed == {"0"}
        actor_c.close()


def test_no_lmdb_writes_for_pending_only_deltas(tmp_path: Path) -> None:
    """Pending counters change in-memory only; LMDB is touched solely
    when a counter hits zero."""
    actor = _new_actor(tmp_path)
    # Lots of activity, but no source resolves.
    actor.apply_deltas([("h0", "0", +1), ("h1", "1", +1), ("h_fanout_0", "0", +2)])
    actor.close()

    # Fresh actor: nothing persisted.
    actor2 = _new_actor(tmp_path)
    assert actor2._completed == set()
    actor2.close()


class TestMultipleWriters:
    """Shared metadata dir with one LMDB file per writer (the SLURM-array
    model): each writer records ONLY its own completions; later writers read
    the union across all writers' files on startup."""

    def test_union_of_completed_across_writers(self, tmp_path: Path) -> None:
        # Writer A finishes source "0".
        a = _new_actor(tmp_path, writer_id="hostA-1")
        a.apply_deltas([("hA", "0", +1), ("hA_sink", "0", -1)])
        assert a._completed == {"0"}
        a.close()

        # Writer B starts later, sees A's completion in the union, finishes "1".
        b = _new_actor(tmp_path, writer_id="hostB-2")
        assert b.are_completed(["0", "1"]) == [True, False]
        b.apply_deltas([("hB", "1", +1), ("hB_sink", "1", -1)])
        b.close()

        # A fresh writer sees the union of everything finished so far.
        c = _new_actor(tmp_path, writer_id="hostC-3")
        assert c.are_completed(["0", "1", "2"]) == [True, True, False]
        c.close()

        # Each writer wrote its OWN file — nothing is shared.
        files = sorted(p.name for p in (tmp_path / ".nemo_curator_metadata").glob("*.mdb"))
        assert files == ["hostA-1.mdb", "hostB-2.mdb", "hostC-3.mdb"]

    def test_writer_does_not_write_other_writers_files(self, tmp_path: Path) -> None:
        # A finishes "s"; B finishes nothing. B must not have touched A's file,
        # and A's completion is still readable on its own.
        a = _new_actor(tmp_path, writer_id="A")
        a.apply_deltas([("h", "s", +1), ("h_sink", "s", -1)])
        a.close()

        b = _new_actor(tmp_path, writer_id="B")  # finishes nothing
        b.close()

        reader = _new_actor(tmp_path, writer_id="reader")
        assert reader.are_completed(["s"]) == [True]
        reader.close()
