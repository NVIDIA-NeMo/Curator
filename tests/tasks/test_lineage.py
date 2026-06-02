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
"""Unit tests for Task lineage primitives.

Covers:
- Task._set_lineage (always overwrites task_id with the lineage path)
- Task.get_deterministic_id (default None)
- assign_root_lineage (roots initial tasks at "0")

The actual per-stage task_id assignment lives in the executor adapter and is
covered in tests/backends/test_task_id_postprocess.py.
FileGroupTask.get_deterministic_id is covered in test_file_group_tasks.py.
"""

from dataclasses import dataclass

from nemo_curator.pipeline.pipeline import assign_root_lineage
from nemo_curator.tasks import EmptyTask, Task, _EmptyTask


@dataclass
class _SimpleTask(Task[list[int]]):
    @property
    def num_items(self) -> int:
        return len(self.data) if self.data is not None else 0

    def validate(self) -> bool:
        return True


def _new() -> _SimpleTask:
    return _SimpleTask(dataset_name="d", data=[1, 2, 3])


class TestRootLineage:
    def test_empty_task_id_is_zero(self) -> None:
        # EmptyTask is the implicit root of the lineage tree.
        assert EmptyTask.task_id == "0"
        assert _EmptyTask(dataset_name="d", data=None).task_id == "0"

    def test_assign_root_lineage_roots_user_tasks_at_zero(self) -> None:
        tasks = [_new(), _new(), _new()]
        assign_root_lineage(tasks)
        # User-provided initial tasks are children of root "0".
        assert [t.task_id for t in tasks] == ["0_0", "0_1", "0_2"]

    def test_assign_root_lineage_skips_empty_tasks(self) -> None:
        et = _EmptyTask(dataset_name="d", data=None)
        real = _new()
        assign_root_lineage([et, real])
        # EmptyTask stays "0"; the real task is rooted by its position.
        assert et.task_id == "0"
        assert real.task_id == "0_1"


class TestSetLineage:
    def test_initial_state(self) -> None:
        t = _new()
        # task_id is empty until _set_lineage runs.
        assert t.task_id == ""

    def test_set_lineage_sets_task_id(self) -> None:
        t = _new()
        t._set_lineage([], 3)
        # task_id is the lineage path itself (no hashing).
        assert t.task_id == "3"

    def test_set_lineage_filters_empty_parent_ids(self) -> None:
        t = _new()
        # Empty strings in parent ids are stripped (so an unassigned parent
        # doesn't contribute a leading/spurious "_" to the path).
        t._set_lineage(["", "5", ""], 3)
        assert t.task_id == "5_3"

    def test_set_lineage_always_overwrites(self) -> None:
        """No idempotency — each call recomputes the id. This is what
        lets a task object passing through N stages get N distinct
        task_ids (one per stage boundary)."""
        t = _new()
        t._set_lineage([], 0)
        first_id = t.task_id

        t._set_lineage(["0"], 7)
        assert t.task_id == "0_7"
        assert t.task_id != first_id

    def test_string_suffix(self) -> None:
        """Source stages pass a content-based hash (str) instead of a
        positional index (int) as the suffix."""
        t = _new()
        t._set_lineage([], "abc123")
        assert t.task_id == "abc123"

    def test_suffix_under_parent_lineage(self) -> None:
        """A task under a non-empty parent id encodes both."""
        t = _new()
        t._set_lineage(["root"], "abc123")
        assert t.task_id == "root_abc123"


class TestGetDeterministicId:
    def test_default_returns_none(self) -> None:
        t = _new()
        assert t.get_deterministic_id() is None
