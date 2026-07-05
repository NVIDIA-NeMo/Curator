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
"""Unit tests for the resumability counter step (``_apply_resumability_counters``)
and the ``None``->``NoneTask`` normalization in ``BaseStageAdapter``, with the
actor RPCs mocked out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import FailedTask, NoneTask, Task


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
class _DropStage(ProcessingStage[Task, Task]):
    """A non-source stage that filters every input (returns ``None``)."""

    name: str = "drop"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: Task) -> None:
        return None


@dataclass
class _SimpleTask(Task[list[int]]):
    @property
    def num_items(self) -> int:
        return 0

    def validate(self) -> bool:
        return True


def _task(task_id: str = "", source_id: str = "") -> _SimpleTask:
    t = _SimpleTask(dataset_name="d", data=[])
    t.task_id = task_id  # pretend _post_process_task_ids already ran
    t._source_id = source_id
    return t


def _counters(
    stage: ProcessingStage,
    input_tasks: list[Task],
    output_tasks: list[Any],
    *,
    completed: set[str] | None = None,
) -> tuple[list[Task], list[tuple[str, str, int]]]:
    """Run ``_apply_resumability_counters`` with the actor RPCs patched.
    Returns ``(surviving_outputs, captured_deltas)``."""
    captured: list[tuple[str, str, int]] = []

    with (
        patch("nemo_curator.backends.base.flush_resumability_deltas", side_effect=captured.extend),
        patch("nemo_curator.backends.base.completed_resumability_sources", return_value=completed or set()),
    ):
        out = BaseStageAdapter(stage)._apply_resumability_counters(input_tasks, output_tasks)
    return out, captured


def _process(
    stage: ProcessingStage,
    tasks: list[Task],
    *,
    completed: set[str] | None = None,
) -> tuple[list[Task], list[tuple[str, str, int]]]:
    """Run the full ``process_batch`` with the resumability actor patched
    active. Returns ``(surviving_outputs, captured_deltas)``."""
    captured: list[tuple[str, str, int]] = []
    with (
        patch("nemo_curator.backends.base.is_resumability_actor_active", return_value=True),
        patch("nemo_curator.backends.base.flush_resumability_deltas", side_effect=captured.extend),
        patch("nemo_curator.backends.base.completed_resumability_sources", return_value=completed or set()),
    ):
        out = BaseStageAdapter(stage).process_batch(tasks)
    return out, captured


class TestNoneNormalization:
    """A returned ``None`` is normalized to a ``NoneTask`` inside
    ``process_batch``: it decrements its slot's source counter and is then
    stripped, so it never reaches the next stage."""

    def test_returned_none_decrements_and_is_stripped(self) -> None:
        parent = _task("s_0", source_id="s")
        out, captured = _process(_DropStage(), [parent])
        assert out == []  # the NoneTask sentinel is stripped from the output
        # Keyed on the NoneTask's assigned OUTPUT id ("s_0_0"), not the parent.
        assert captured == [("s_0_0", "s", -1)]  # the filtered slot is consumed


class TestSourceStage:
    def _src_stage(self) -> _NoopStage:
        s = _NoopStage()
        s.is_source_stage = True
        return s

    def test_stamps_source_id_and_fires_plus_one(self) -> None:
        empty = _task("0")  # EmptyTask-like root
        a, b = _task("0_aaa"), _task("0_bbb")
        out, captured = _counters(self._src_stage(), [empty], [a, b])
        assert out == [a, b]
        # _source_id is the task_id's last segment (its content id / index).
        assert a._source_id == "aaa"
        assert b._source_id == "bbb"
        assert sorted(captured) == [("0_aaa", "aaa", 1), ("0_bbb", "bbb", 1)]

    def test_drops_already_completed_sources(self) -> None:
        empty = _task("0")
        a, b, c = _task("0_a"), _task("0_b"), _task("0_c")
        out, captured = _counters(self._src_stage(), [empty], [a, b, c], completed={"b"})
        assert out == [a, c]
        assert {sid for _, sid, _ in captured} == {"a", "c"}


class TestNonSourceStage:
    def test_pre_source_is_noop(self) -> None:
        # Inputs carry no _source_id yet -> nothing tracked, outputs untouched.
        a = _task("0_0")
        out, captured = _counters(_NoopStage(), [a], [a])
        assert out == [a]
        assert captured == []

    def test_one_to_one_nonsink_zero_delta(self) -> None:
        stage = _NoopStage()
        stage.is_sink_stage = False
        parent = _task("s_0", source_id="s")
        child = _task("s_0_0")
        _out, captured = _counters(stage, [parent], [child])
        # Keyed on the OUTPUT id (child), not the parent.
        assert captured == [("s_0_0", "s", 0)]
        assert child._source_id == "s"  # inherited

    def test_one_to_one_sink_minus_one(self) -> None:
        stage = _NoopStage()
        stage.is_sink_stage = True
        parent = _task("s_0", source_id="s")
        _out, captured = _counters(stage, [parent], [_task("s_0_0")])
        assert captured == [("s_0_0", "s", -1)]

    def test_nonetask_slot_decrements(self) -> None:
        # NoneTask keys on its own assigned OUTPUT id (here "s_0_0"), so it can't
        # collide with the source's +1 for the same partition ("s_0").
        parent = _task("s_0", source_id="s")
        nt = NoneTask()
        nt.task_id = "s_0_0"
        _out, captured = _counters(_NoopStage(), [parent], [nt])
        assert captured == [("s_0_0", "s", -1)]

    def test_failedtask_slot_zero_delta(self) -> None:
        # Failed fires delta 0 (keyed on its output id): the source's +1 stays,
        # so the source never completes and reruns. No sink test for Failed.
        parent = _task("s_0", source_id="s")
        ft = FailedTask()
        ft.task_id = "s_0_0"
        _out, captured = _counters(_NoopStage(), [parent], [ft])
        assert captured == [("s_0_0", "s", 0)]

    def test_fanout_grows_counter(self) -> None:
        stage = _NoopStage()
        stage.is_sink_stage = False
        parent = _task("s_0", source_id="s")
        c0, c1, c2 = _task("s_0_0"), _task("s_0_1"), _task("s_0_2")
        _out, captured = _counters(stage, [parent], [c0, c1, c2])
        # 1 input -> 3 real children: net +2, keyed on output[0] ("s_0_0").
        assert captured == [("s_0_0", "s", 2)]
        assert all(c._source_id == "s" for c in (c0, c1, c2))

    def test_fanout_nonsink_mixed_real_none_failed(self) -> None:
        # 1 -> [real, None, Failed], non-sink: real continues (+1), Failed keeps
        # the source open (+1), None contributes 0, parent consumed (-1).
        stage = _NoopStage()
        stage.is_sink_stage = False
        parent = _task("s_0", source_id="s")
        real = _task("s_0_0")
        nt, ft = NoneTask(), FailedTask()
        nt.task_id, ft.task_id = "s_0_1", "s_0_2"
        _out, captured = _counters(stage, [parent], [real, nt, ft])
        # net = continuing(1) + failed(1) - 1 = 1, keyed on output[0].
        assert captured == [("s_0_0", "s", 1)]

    def test_fanout_sink_real_outputs_leave(self) -> None:
        # 1 -> [real, real, Failed] at a SINK: real outputs leave (0), Failed
        # stays (+1), parent consumed (-1) -> net 0, source stays open.
        stage = _NoopStage()
        stage.is_sink_stage = True
        parent = _task("s_0", source_id="s")
        r0, r1 = _task("s_0_0"), _task("s_0_1")
        ft = FailedTask()
        ft.task_id = "s_0_2"
        _out, captured = _counters(stage, [parent], [r0, r1, ft])
        assert captured == [("s_0_0", "s", 0)]

    def test_empty_output_skips(self) -> None:
        # A stage that emits nothing (not even a NoneTask): no output to key a
        # delta on, so nothing is fired.
        parent = _task("s_0", source_id="s")
        out, captured = _counters(_NoopStage(), [parent], [])
        assert captured == []
        assert out == []

    def test_ambiguous_batch_skips_counters(self) -> None:
        # 2 inputs -> 3 outputs: can't attribute, so no deltas are fired.
        p0, p1 = _task("s_0", source_id="s"), _task("s_1", source_id="s")
        out, captured = _counters(_NoopStage(), [p0, p1], [_task(), _task(), _task()])
        assert captured == []
        assert len(out) == 3
