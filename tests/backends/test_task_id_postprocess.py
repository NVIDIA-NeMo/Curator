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
"""Unit tests for ``BaseStageAdapter._post_process_task_ids`` — the single
place every backend assigns a deterministic ``task_id`` to emitted tasks.

The happy-path flow (fan-out, 1:1, source content ids) is exercised
end-to-end against real backends in tests/backends/test_integration.py
(``test_task_ids``). This file keeps only the cases that are awkward or
impossible to trigger through a real pipeline: filter-``None`` positional
alignment, the ambiguous-cardinality ``"r"``-uuid fallback, preservation of
    framework overwrite semantics, and source content-id selection."""

from dataclasses import dataclass

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.pipeline.payload_refs import PayloadRef
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, EmptyTask, FileGroupTask, Task
from nemo_curator.tasks.task_terminals import (
    TERMINAL_COUNT_KEY,
    TERMINAL_DROPPED_KEY,
    TERMINAL_GROUP_ID_KEY,
    TERMINAL_INDEX_KEY,
)


@dataclass
class _NoopStage(ProcessingStage[Task, Task]):
    name: str = "noop"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: Task) -> Task:
        return task


@dataclass
class _DropSegmentRowStage(ProcessingStage[AudioTask, AudioTask]):
    name: str = "drop_segment_row"
    skip_me_key: str = "_skip_me"
    _curator_preserves_terminal_tasks: bool = True

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def terminal_tombstone_drop_data_keys(self) -> tuple[str, ...]:
        return ("large_payload",)

    def process(self, task: AudioTask) -> AudioTask | None:
        if task.data.get("drop"):
            return None
        return task


@dataclass
class _SimpleTask(Task[list[int]]):
    @property
    def num_items(self) -> int:
        return 0

    def validate(self) -> bool:
        return True


def _task(task_id: str = "") -> _SimpleTask:
    t = _SimpleTask(dataset_name="d", data=[])
    t.task_id = task_id
    return t


def _assign(tasks: list[Task], results: list[Task | None], *, is_source: bool = False) -> list[Task]:
    stage = _NoopStage()
    stage.is_source_stage = is_source
    return BaseStageAdapter(stage)._post_process_task_ids(tasks, results)


class TestPostProcessTaskIds:
    def test_filter_stage_keeps_positional_alignment(self) -> None:
        # A filter stage returns None in the filtered slot. None is NOT
        # dropped before the length check, so the surviving outputs still map
        # to their OWN parents (not shifted), then None slots are removed.
        p0, p1, p2 = _task("0_0"), _task("0_1"), _task("0_2")
        c0, c2 = _task(), _task()
        out = _assign([p0, p1, p2], [c0, None, c2])
        assert out == [c0, c2]
        assert c0.task_id == "0_0_0"  # child of p0, not shifted
        assert c2.task_id == "0_2_0"  # child of p2, not p1

    def test_in_place_return_is_reassigned(self) -> None:
        t = _task("0_5")
        out = _assign([t], [t])
        assert out == [t]
        assert t.task_id == "0_5_0"

    def test_ambiguous_batch_fanout_falls_back_to_uuid(self) -> None:
        # M inputs → K outputs (K != M, M > 1): mapping is ambiguous, so each
        # output gets a random uuid rather than being left empty.
        p0, p1 = _task("0_0"), _task("0_1")
        c0, c1, c2 = _task(), _task(), _task()
        out = _assign([p0, p1], [c0, c1, c2])
        assert len(out) == 3
        assert all(t.task_id for t in out), "no output should be left without an id"
        assert len({t.task_id for t in out}) == 3, "uuid ids should be unique"
        # Non-deterministic fallback ids are flagged with an "r" prefix.
        assert all(t.task_id.startswith("r") for t in out)
        assert all("_" not in t.task_id for t in out)

    def test_dropped_segment_row_is_preserved_as_tombstone(self) -> None:
        keep = AudioTask(
            dataset_name="d",
            data={
                "_curator_segment_parent_id": "manifest:0:0",
                "_curator_segment_idx": 0,
                "_curator_segment_count": 2,
            },
        )
        drop = AudioTask(
            dataset_name="d",
            data={
                "drop": True,
                "large_payload": object(),
                "payload_ref": PayloadRef(
                    payload_id="p",
                    owner_node_id="node",
                    store_actor_name="store",
                    admission_actor_name="admission",
                    amount_bytes=1,
                    sample_rate=16000,
                    num_samples=1,
                ),
                "_curator_segment_parent_id": "manifest:0:0",
                "_curator_segment_idx": 1,
                "_curator_segment_count": 2,
            },
        )
        keep.task_id = "0_0"
        drop.task_id = "0_1"

        out = BaseStageAdapter(_DropSegmentRowStage()).process_batch([keep, drop])

        assert len(out) == 2
        assert out[0] is keep
        tombstone = out[1]
        assert tombstone.data["_skip_me"] == "dropped_segment_row"
        assert tombstone.data["_curator_segment_dropped"] is True
        assert tombstone.data["_curator_segment_idx"] == 1
        assert "large_payload" not in tombstone.data
        assert "payload_ref" not in tombstone.data
        assert tombstone.task_id == "0_1_0"

    def test_dropped_generic_terminal_row_is_preserved_as_tombstone(self) -> None:
        keep = AudioTask(
            dataset_name="d",
            data={
                TERMINAL_GROUP_ID_KEY: "parent-0",
                TERMINAL_INDEX_KEY: 0,
                TERMINAL_COUNT_KEY: 2,
            },
        )
        drop = AudioTask(
            dataset_name="d",
            data={
                "drop": True,
                "large_payload": object(),
                TERMINAL_GROUP_ID_KEY: "parent-0",
                TERMINAL_INDEX_KEY: 1,
                TERMINAL_COUNT_KEY: 2,
            },
        )
        keep.task_id = "0_0"
        drop.task_id = "0_1"

        out = BaseStageAdapter(_DropSegmentRowStage()).process_batch([keep, drop])

        assert len(out) == 2
        tombstone = out[1]
        assert tombstone.data["_skip_me"] == "dropped_segment_row"
        assert tombstone.data[TERMINAL_DROPPED_KEY] is True
        assert "_curator_segment_dropped" not in tombstone.data
        assert "large_payload" not in tombstone.data
        assert tombstone.task_id == "0_1_0"


class TestSourceStage:
    def test_uses_content_id_rooted_at_input(self) -> None:
        # FileGroupTask.get_deterministic_id() hashes its files; output with no
        # content ids are rooted at the framework EmptyTask id.
        empty = EmptyTask(dataset_name="empty", data=None)
        a = FileGroupTask(dataset_name="d", data=["a.parquet"])
        b = FileGroupTask(dataset_name="d", data=["b.parquet"])
        _assign([empty], [a, b], is_source=True)
        assert a.task_id == f"0_{a.get_deterministic_id()}"
        assert b.task_id == f"0_{b.get_deterministic_id()}"

    def test_n_to_n_source_parents_each_output_by_position(self) -> None:
        # A source stage can also be N→N (each input → one partition). Each
        # output must descend from ITS positional parent, not all from
        # tasks[0]; the content id is the segment.
        p0, p1 = _task("0_0"), _task("0_1")
        a = FileGroupTask(dataset_name="d", data=["a.parquet"])
        b = FileGroupTask(dataset_name="d", data=["b.parquet"])
        _assign([p0, p1], [a, b], is_source=True)
        assert a.task_id == f"0_0_{a.get_deterministic_id()}"
        assert b.task_id == f"0_1_{b.get_deterministic_id()}"

    def test_non_source_stage_ignores_content_id(self) -> None:
        # The same FileGroupTask outputs from a NON-source stage use the
        # positional index, not the content id.
        parent = _task("0_2")
        a = FileGroupTask(dataset_name="d", data=["a.parquet"])
        _assign([parent], [a], is_source=False)
        assert a.task_id == "0_2_0"
