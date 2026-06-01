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
- Task._set_lineage (always overwrites task_id with the lineage path)
- Task.get_deterministic_id (default None)
- Default ProcessingStage.process_batch calls _set_lineage on each child

FileGroupTask.get_deterministic_id is covered in test_file_group_tasks.py.
"""

from dataclasses import dataclass

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import FileGroupTask, Task, _EmptyTask


@dataclass
class _SimpleTask(Task[list[int]]):
    @property
    def num_items(self) -> int:
        return len(self.data) if self.data is not None else 0

    def validate(self) -> bool:
        return True


def _new() -> _SimpleTask:
    return _SimpleTask(dataset_name="d", data=[1, 2, 3])


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
        # Empty strings in parent ids are stripped (EmptyTask's default
        # task_id is "").
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


class _Repeat(ProcessingStage[_SimpleTask, _SimpleTask]):
    times: int = 3
    name: str = "repeat"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _SimpleTask) -> list[_SimpleTask]:
        return [_SimpleTask(dataset_name=task.dataset_name, data=task.data) for _ in range(self.times)]


class _Passthrough(ProcessingStage[_SimpleTask, _SimpleTask]):
    name: str = "passthrough"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _SimpleTask) -> _SimpleTask:
        return _SimpleTask(dataset_name=task.dataset_name, data=task.data)


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
        parent = _new()
        parent._set_lineage([], 0)
        out = _Passthrough().process_batch([parent])
        assert len(out) == 1
        # Child's id is parent's id + "_0"
        assert out[0].task_id == "0_0"

    def test_fanout_outputs_get_unique_task_ids(self) -> None:
        parent = _new()
        parent._set_lineage([], 0)
        out = _Repeat(times=4).process_batch([parent])
        assert len(out) == 4
        task_ids = [t.task_id for t in out]
        assert task_ids == ["0_0", "0_1", "0_2", "0_3"]
        assert len(set(task_ids)) == 4

    def test_filtered_out_returns_empty(self) -> None:
        parent = _new()
        parent._set_lineage([], 0)
        out = _Filter().process_batch([parent])
        assert out == []

    def test_batch_inputs_keep_per_parent_lineage(self) -> None:
        """Each parent in the batch contributes its own lineage to its
        child(ren)."""
        parents = [_new() for i in range(3)]
        for i, p in enumerate(parents):
            p._set_lineage([], i)

        out = _Passthrough().process_batch(parents)
        assert len(out) == 3
        # Each child's id = parent's id + "_0".
        for i, child in enumerate(out):
            assert child.task_id == f"{i}_0"

    def test_deterministic_across_calls(self) -> None:
        """Running the same pipeline shape on the same inputs produces
        byte-identical task_ids."""
        parent_a = _new()
        parent_a._set_lineage([], 0)
        out_a = _Repeat(times=2).process_batch([parent_a])

        parent_b = _new()
        parent_b._set_lineage([], 0)
        out_b = _Repeat(times=2).process_batch([parent_b])

        # Same inputs → same lineage paths → same task_ids.
        assert [t.task_id for t in out_a] == [t.task_id for t in out_b]


class _FileGroupSource(ProcessingStage[_EmptyTask, FileGroupTask]):
    """Test-only source stage that emits FileGroupTasks. Source-stage flag
    set to True so the default process_batch uses content-based ids."""

    name: str = "file_group_source"
    is_source_stage: bool = True

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _EmptyTask) -> list[FileGroupTask]:
        return [FileGroupTask(dataset_name="d", data=[f"file_{i}.parquet"]) for i in range(3)]


class _NoContentSource(ProcessingStage[_EmptyTask, _SimpleTask]):
    """Source stage whose output Task subclass doesn't implement
    get_deterministic_id — should fall back to positional indices."""

    name: str = "no_content_source"
    is_source_stage: bool = True

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: _EmptyTask) -> list[_SimpleTask]:
        return [_SimpleTask(dataset_name="d", data=[i]) for i in range(3)]


class TestSourceStageSegment:
    def test_uses_content_hash_when_get_deterministic_id_returns_value(self) -> None:
        empty = _EmptyTask(dataset_name="empty", data=None)
        out = _FileGroupSource().process_batch([empty])
        assert len(out) == 3
        # Each output's task_id should be exactly its content hash — NOT the
        # positional index. EmptyTask's task_id is "" (filtered), so the id
        # is just the content hash itself.
        for child in out:
            assert child.task_id == child.get_deterministic_id()

    def test_falls_back_to_positional_index_when_none(self) -> None:
        empty = _EmptyTask(dataset_name="empty", data=None)
        out = _NoContentSource().process_batch([empty])
        assert len(out) == 3
        task_ids = [t.task_id for t in out]
        # EmptyTask filtered; suffix is positional index.
        assert task_ids == ["0", "1", "2"]

    def test_non_source_stage_uses_positional_index(self) -> None:
        """A stage without is_source_stage=True ignores get_deterministic_id
        even when the Task subclass implements one (e.g. FileGroupTask)."""

        class _NonSource(ProcessingStage[FileGroupTask, FileGroupTask]):
            name: str = "non_source"
            # is_source_stage stays False by default.

            def inputs(self) -> tuple[list[str], list[str]]:
                return [], []

            def outputs(self) -> tuple[list[str], list[str]]:
                return [], []

            def process(self, task: FileGroupTask) -> FileGroupTask:
                return FileGroupTask(dataset_name="d", data=task.data)

        parent = FileGroupTask(dataset_name="d", data=["a.parquet"])
        parent._set_lineage([], 0)
        out = _NonSource().process_batch([parent])
        # Lineage uses positional "0", not content hash.
        assert out[0].task_id == "0_0"
