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

"""Unit tests for CheckpointManager (LMDB-backed).  No Ray cluster required."""

import hashlib
from pathlib import Path

import pytest

from nemo_curator.utils.checkpoint import CheckpointManager


def _key(s: str) -> str:
    """Expected 16-char hash prefix used internally."""
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _task_key(key: str, pos: int) -> str:
    """Build a path-style resumability_task_key for tests (matches adapter convention)."""
    return f"{key}::{pos}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mgr(tmp_path: Path) -> CheckpointManager:
    return CheckpointManager(str(tmp_path))


# ---------------------------------------------------------------------------
# init_partition
# ---------------------------------------------------------------------------


class TestInitPartition:
    def test_sets_expected_to_one(self, mgr: CheckpointManager) -> None:
        mgr.init_partition("key_a")
        assert not mgr.is_task_completed("key_a")  # 0 completed, 1 expected → not done

    def test_idempotent(self, mgr: CheckpointManager) -> None:
        mgr.init_partition("key_a")
        mgr.add_expected("key_a", 2)  # expected → 3
        mgr.init_partition("key_a")  # must NOT reset to 1
        mgr.mark_completed(_task_key("key_a", 0), "key_a")
        assert not mgr.is_task_completed("key_a")  # still 1/3

    def test_unknown_key_not_completed(self, mgr: CheckpointManager) -> None:
        assert not mgr.is_task_completed("never_seen")


# ---------------------------------------------------------------------------
# is_task_completed
# ---------------------------------------------------------------------------


class TestIsTaskCompleted:
    def test_not_completed_before_any_marks(self, mgr: CheckpointManager) -> None:
        mgr.init_partition("p")
        assert not mgr.is_task_completed("p")

    def test_completed_after_single_mark(self, mgr: CheckpointManager) -> None:
        key = "single"
        mgr.init_partition(key)
        mgr.mark_completed(_task_key(key, 0), key)
        assert mgr.is_task_completed(key)

    def test_fanout_needs_all_marks(self, mgr: CheckpointManager) -> None:
        key = "fan3"
        mgr.init_partition(key)  # expected = 1
        mgr.add_expected(key, 2)  # expected = 3

        mgr.mark_completed(_task_key(key, 1), key)
        assert not mgr.is_task_completed(key)

        mgr.mark_completed(_task_key(key, 2), key)
        assert not mgr.is_task_completed(key)

        mgr.mark_completed(_task_key(key, 3), key)
        assert mgr.is_task_completed(key)

    def test_dropped_task_counts_as_completed(self, mgr: CheckpointManager) -> None:
        key = "drop"
        mgr.init_partition(key)
        mgr.mark_completed(_task_key(key, 0), key)  # dropped → still counts
        assert mgr.is_task_completed(key)

    def test_idempotent_mark_does_not_overcount(self, mgr: CheckpointManager) -> None:
        key = "idem"
        mgr.init_partition(key)  # expected = 1
        mgr.add_expected(key, 1)  # expected = 2
        same_task_key = _task_key(key, 1)
        mgr.mark_completed(same_task_key, key)
        mgr.mark_completed(same_task_key, key)  # duplicate — should not count twice
        assert not mgr.is_task_completed(key)  # only 1 unique done, 2 expected

    def test_different_keys_isolated(self, mgr: CheckpointManager) -> None:
        mgr.init_partition("k1")
        mgr.init_partition("k2")
        mgr.mark_completed(_task_key("k1", 0), "k1")
        assert mgr.is_task_completed("k1")
        assert not mgr.is_task_completed("k2")


# ---------------------------------------------------------------------------
# add_expected
# ---------------------------------------------------------------------------


class TestAddExpected:
    def test_increments_expected(self, mgr: CheckpointManager) -> None:
        key = "inc"
        mgr.init_partition(key)  # expected = 1
        mgr.add_expected(key, 4)  # expected = 5
        for i in range(4):
            mgr.mark_completed(_task_key(key, i + 1), key)
        assert not mgr.is_task_completed(key)
        mgr.mark_completed(_task_key(key, 0), key)
        assert mgr.is_task_completed(key)

    def test_chained_fanout(self, mgr: CheckpointManager) -> None:
        key = "chain"
        mgr.init_partition(key)  # expected = 1
        mgr.add_expected(key, 2)  # first fan-out: 1→3, expected = 3
        mgr.add_expected(key, 3)  # second fan-out: one of those 3 fans to 4, expected = 6
        for i in range(6):
            mgr.mark_completed(_task_key(key, i), key)
        assert mgr.is_task_completed(key)


# ---------------------------------------------------------------------------
# are_leaves_completed
# ---------------------------------------------------------------------------


class TestAreLeavesCompleted:
    def test_empty_input(self, mgr: CheckpointManager) -> None:
        assert mgr.are_leaves_completed([]) == []

    def test_returns_false_before_marks(self, mgr: CheckpointManager) -> None:
        mgr.init_partition("p")
        flags = mgr.are_leaves_completed([("p", _task_key("p", 0)), ("p", _task_key("p", 1))])
        assert flags == [False, False]

    def test_returns_true_after_mark(self, mgr: CheckpointManager) -> None:
        key = "p"
        mgr.init_partition(key)
        mgr.mark_completed(_task_key(key, 0), key)
        flags = mgr.are_leaves_completed([(key, _task_key(key, 0)), (key, _task_key(key, 1))])
        assert flags == [True, False]

    def test_isolation_across_partitions(self, mgr: CheckpointManager) -> None:
        mgr.init_partition("k1")
        mgr.init_partition("k2")
        mgr.mark_completed(_task_key("k1", 0), "k1")
        flags = mgr.are_leaves_completed(
            [
                ("k1", _task_key("k1", 0)),  # marked
                ("k2", _task_key("k1", 0)),  # same task_key, different partition → not marked
                ("k1", _task_key("k2", 0)),  # different task_key under k1 → not marked
            ]
        )
        assert flags == [True, False, False]

    def test_mixed_batch_preserves_order(self, mgr: CheckpointManager) -> None:
        key = "p"
        mgr.init_partition(key)
        for i in (1, 3):
            mgr.mark_completed(_task_key(key, i), key)
        flags = mgr.are_leaves_completed([(key, _task_key(key, i)) for i in range(5)])
        assert flags == [False, True, False, True, False]


# ---------------------------------------------------------------------------
# reset_partition
# ---------------------------------------------------------------------------


class TestResetPartition:
    def test_preserves_completions_and_unfinalizes(self, mgr: CheckpointManager) -> None:
        """Reset must keep leaf entries so already-done leaves are skipped on resume."""
        key = "k"
        mgr.init_partition(key)
        mgr.add_expected(key, 2)  # expected = 3
        for i in range(3):
            mgr.mark_completed(_task_key(key, i), key)
        assert mgr.is_task_completed(key)

        mgr.reset_partition(key)

        # All three leaf entries survive
        flags = mgr.are_leaves_completed([(key, _task_key(key, i)) for i in range(3)])
        assert flags == [True, True, True]
        # Partition is no longer finalized
        assert not mgr.is_task_completed(key)

    def test_does_not_falsely_complete_after_reset(self, mgr: CheckpointManager) -> None:
        """The bug guard: with completions kept and expected rewound to 1, a crash
        before fan-out re-fires must NOT make the next resume mistake the partition
        for finished. Without an explicit ``finalized`` flag, ``count >= expected``
        would be true post-reset and the partition would be incorrectly skipped.
        """
        key = "interrupted"
        mgr.init_partition(key)
        mgr.add_expected(key, 2)  # expected = 3
        mgr.mark_completed(_task_key(key, 0), key)  # 1 of 3 leaves done

        mgr.reset_partition(key)

        # Even though count (1) >= expected (1) post-reset, the partition is NOT done.
        assert not mgr.is_task_completed(key)

    def test_partial_completion_resumes_correctly(self, mgr: CheckpointManager) -> None:
        """End-to-end: partial completion → reset → re-fan-out → only missing leaves run."""
        key = "partial"
        # Simulate previous run: expected=3, 1 leaf done, crashed.
        mgr.init_partition(key)
        mgr.add_expected(key, 2)
        mgr.mark_completed(_task_key(key, 0), key)
        assert not mgr.is_task_completed(key)

        # Resume: filter stage resets.
        mgr.reset_partition(key)
        # Fan-out re-fires.
        mgr.add_expected(key, 2)  # expected back to 3, count still 1, not finalized
        assert not mgr.is_task_completed(key)

        # Only the missing leaves are recorded this run.
        mgr.mark_completed(_task_key(key, 1), key)
        mgr.mark_completed(_task_key(key, 2), key)
        assert mgr.is_task_completed(key)

    def test_no_completions_resets_to_clean_state(self, mgr: CheckpointManager) -> None:
        """A partition that had no completions should still reset cleanly."""
        key = "clean"
        mgr.init_partition(key)
        mgr.add_expected(key, 4)  # inflated by fanout

        mgr.reset_partition(key)

        # Clean run: 1 leaf marks it complete (expected back to 1, no prior leaves).
        mgr.mark_completed(_task_key(key, 0), key)
        assert mgr.is_task_completed(key)


# ---------------------------------------------------------------------------
# resumability_key construction for FilePartitioningStage
# ---------------------------------------------------------------------------


class TestResumabilityKey:
    def test_sorted_files_eliminate_ordering_variance(self) -> None:
        files_a = ["b.jsonl", "a.jsonl"]
        files_b = ["a.jsonl", "b.jsonl"]
        key_a = "|".join(sorted(files_a)) + "::0"
        key_b = "|".join(sorted(files_b)) + "::0"
        assert key_a == key_b

    def test_partition_index_discriminates_same_files(self) -> None:
        files = ["x.jsonl"]
        key_0 = "|".join(sorted(files)) + "::0"
        key_1 = "|".join(sorted(files)) + "::1"
        assert key_0 != key_1

    def test_different_file_groups_different_keys(self) -> None:
        key_0 = "|".join(sorted(["a.jsonl", "b.jsonl"])) + "::0"
        key_1 = "|".join(sorted(["c.jsonl", "d.jsonl"])) + "::0"
        assert key_0 != key_1
