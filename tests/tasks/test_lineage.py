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
"""Unit tests for Task lineage assignment.

Covers:
- Task._set_lineage (always overwrites; sets task_id and _lineage_path)
- Task.get_deterministic_id (default None)
- FileGroupTask.get_deterministic_id override (content-based hash)
- Default ProcessingStage.process_batch calls _set_lineage on each child
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import FileGroupTask, Task


@dataclass
class _SimpleTask(Task[list[int]]):
    @property
    def num_items(self) -> int:
        return len(self.data) if self.data is not None else 0

    def validate(self) -> bool:
        return True


def _new(tid: str = "x") -> _SimpleTask:
    return _SimpleTask(task_id=tid, dataset_name="d", data=[1, 2, 3])


class TestSetLineage:
    def test_initial_state(self) -> None:
        t = _new()
        # User-provided task_id stays until _set_lineage runs.
        assert t.task_id == "x"
        assert t._lineage_path == ""

    def test_set_lineage_sets_path_and_task_id(self) -> None:
        t = _new()
        t._set_lineage([], 3)
        assert t._lineage_path == "3"
        # task_id is now the sha256-32 of the lineage path; user-provided
        # value is overwritten.
        assert t.task_id == hashlib.sha256(b"3").hexdigest()[:32]
        assert t.task_id != "x"

    def test_set_lineage_filters_empty_parent_paths(self) -> None:
        t = _new()
        # Empty strings in parent paths are stripped (EmptyTask's default
        # _lineage_path is "").
        t._set_lineage(["", "5", ""], 3)
        assert t._lineage_path == "5_3"

    def test_set_lineage_always_overwrites(self) -> None:
        """No idempotency — each call recomputes lineage. This is what
        lets a task object passing through N stages get N distinct
        task_ids (one per stage boundary)."""
        t = _new()
        t._set_lineage([], 0)
        first_id = t.task_id
        first_path = t._lineage_path

        t._set_lineage(["0"], 7)
        assert t._lineage_path == "0_7"
        assert t._lineage_path != first_path
        assert t.task_id != first_id

    def test_string_child_segment(self) -> None:
        """Source stages pass a content-based hash (str) instead of a
        positional index (int) as the child_segment."""
        t = _new()
        t._set_lineage([], "abc123")
        assert t._lineage_path == "abc123"
        assert t.task_id == hashlib.sha256(b"abc123").hexdigest()[:32]

    def test_child_segment_under_parent_lineage(self) -> None:
        """Source under a non-empty parent lineage encodes both."""
        t = _new()
        t._set_lineage(["root"], "abc123")
        assert t._lineage_path == "root_abc123"


class TestGetDeterministicId:
    def test_default_returns_none(self) -> None:
        t = _new()
        assert t.get_deterministic_id() is None

    def test_file_group_task_hashes_sorted_paths(self) -> None:
        a = FileGroupTask(task_id="t1", dataset_name="d", data=["b.parquet", "a.parquet"])
        b = FileGroupTask(task_id="t2", dataset_name="d", data=["a.parquet", "b.parquet"])
        # Same set of files in different orders → same content id.
        assert a.get_deterministic_id() == b.get_deterministic_id()

    def test_file_group_task_different_files_different_ids(self) -> None:
        a = FileGroupTask(task_id="t", dataset_name="d", data=["a.parquet"])
        b = FileGroupTask(task_id="t", dataset_name="d", data=["b.parquet"])
        assert a.get_deterministic_id() != b.get_deterministic_id()

    def test_file_group_id_is_string(self) -> None:
        t = FileGroupTask(task_id="t", dataset_name="d", data=["a.parquet"])
        result = t.get_deterministic_id()
        assert isinstance(result, str)
        assert len(result) == 12  # get_deterministic_hash returns 12 hex chars


class _Repeat(ProcessingStage[_SimpleTask, _SimpleTask]):
    times: int = 3
    name: str = "repeat"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _SimpleTask) -> list[_SimpleTask]:
        return [
            _SimpleTask(task_id="placeholder", dataset_name=task.dataset_name, data=task.data)
            for _ in range(self.times)
        ]


class _Passthrough(ProcessingStage[_SimpleTask, _SimpleTask]):
    name: str = "passthrough"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _SimpleTask) -> _SimpleTask:
        return _SimpleTask(task_id="placeholder", dataset_name=task.dataset_name, data=task.data)


class _Filter(ProcessingStage[_SimpleTask, _SimpleTask]):
    name: str = "filter"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _SimpleTask) -> _SimpleTask | None:
        return None  # always filter out


class TestDefaultProcessBatchAssignsLineage:
    def test_single_task_single_output_gets_lineage(self) -> None:
        parent = _new("p")
        parent._set_lineage([], 0)
        out = _Passthrough().process_batch([parent])
        assert len(out) == 1
        # Child's lineage is parent's lineage + "_0"
        assert out[0]._lineage_path == "0_0"

    def test_fanout_outputs_get_unique_lineage_paths(self) -> None:
        parent = _new("p")
        parent._set_lineage([], 0)
        out = _Repeat(times=4).process_batch([parent])
        assert len(out) == 4
        paths = [t._lineage_path for t in out]
        assert paths == ["0_0", "0_1", "0_2", "0_3"]
        # And task_ids are unique deterministic hashes.
        task_ids = [t.task_id for t in out]
        assert len(set(task_ids)) == 4

    def test_filtered_out_returns_empty(self) -> None:
        parent = _new("p")
        parent._set_lineage([], 0)
        out = _Filter().process_batch([parent])
        assert out == []

    def test_batch_inputs_keep_per_parent_lineage(self) -> None:
        """Each parent in the batch contributes its own lineage to its
        child(ren)."""
        parents = [_new(f"p{i}") for i in range(3)]
        for i, p in enumerate(parents):
            p._set_lineage([], i)

        out = _Passthrough().process_batch(parents)
        assert len(out) == 3
        # Each child's lineage = parent's lineage + "_0".
        for i, child in enumerate(out):
            assert child._lineage_path == f"{i}_0"

    def test_deterministic_across_calls(self) -> None:
        """Running the same pipeline shape on the same inputs produces
        byte-identical task_ids."""
        parent_a = _new("a")
        parent_a._set_lineage([], 0)
        out_a = _Repeat(times=2).process_batch([parent_a])

        parent_b = _new("b")  # different user-provided task_id
        parent_b._set_lineage([], 0)
        out_b = _Repeat(times=2).process_batch([parent_b])

        # Same lineage paths despite different user-provided task_ids — the
        # user-provided values were overwritten by _set_lineage.
        assert [t.task_id for t in out_a] == [t.task_id for t in out_b]
