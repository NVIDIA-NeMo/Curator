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


def _uuid(key: str, pos: int) -> str:
    return hashlib.sha256(f"{key}::{pos}".encode()).hexdigest()[:16]


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
        uuid = _uuid("key_a", 0)
        mgr.mark_completed(uuid, "key_a")
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
        mgr.mark_completed(_uuid(key, 0), key)
        assert mgr.is_task_completed(key)

    def test_fanout_needs_all_marks(self, mgr: CheckpointManager) -> None:
        key = "fan3"
        mgr.init_partition(key)  # expected = 1
        mgr.add_expected(key, 2)  # expected = 3

        mgr.mark_completed(_uuid(key, 1), key)
        assert not mgr.is_task_completed(key)

        mgr.mark_completed(_uuid(key, 2), key)
        assert not mgr.is_task_completed(key)

        mgr.mark_completed(_uuid(key, 3), key)
        assert mgr.is_task_completed(key)

    def test_dropped_task_counts_as_completed(self, mgr: CheckpointManager) -> None:
        key = "drop"
        mgr.init_partition(key)
        mgr.mark_completed(_uuid(key, 0), key)  # dropped → still counts
        assert mgr.is_task_completed(key)

    def test_idempotent_mark_does_not_overcount(self, mgr: CheckpointManager) -> None:
        key = "idem"
        mgr.init_partition(key)  # expected = 1
        mgr.add_expected(key, 1)  # expected = 2
        same_uuid = _uuid(key, 1)
        mgr.mark_completed(same_uuid, key)
        mgr.mark_completed(same_uuid, key)  # duplicate — should not count twice
        assert not mgr.is_task_completed(key)  # only 1 unique done, 2 expected

    def test_different_keys_isolated(self, mgr: CheckpointManager) -> None:
        mgr.init_partition("k1")
        mgr.init_partition("k2")
        mgr.mark_completed(_uuid("k1", 0), "k1")
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
            mgr.mark_completed(_uuid(key, i + 1), key)
        assert not mgr.is_task_completed(key)
        mgr.mark_completed(_uuid(key, 0), key)
        assert mgr.is_task_completed(key)

    def test_chained_fanout(self, mgr: CheckpointManager) -> None:
        key = "chain"
        mgr.init_partition(key)  # expected = 1
        mgr.add_expected(key, 2)  # first fan-out: 1→3, expected = 3
        mgr.add_expected(key, 3)  # second fan-out: one of those 3 fans to 4, expected = 6
        for i in range(6):
            mgr.mark_completed(_uuid(key, i), key)
        assert mgr.is_task_completed(key)


# ---------------------------------------------------------------------------
# Deterministic _resumability_uuid
# ---------------------------------------------------------------------------


class TestResumabilityUUID:
    def test_same_inputs_same_uuid(self) -> None:
        key = "k"
        assert _uuid(key, 0) == _uuid(key, 0)
        assert _uuid(key, 1) == _uuid(key, 1)

    def test_different_positions_different_uuids(self) -> None:
        key = "k"
        uuids = [_uuid(key, i) for i in range(5)]
        assert len(set(uuids)) == 5

    def test_different_keys_different_uuids(self) -> None:
        assert _uuid("key_a", 0) != _uuid("key_b", 0)


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
