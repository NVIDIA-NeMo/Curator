# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import Task


@dataclass
class SimpleTask(Task[list[int]]):
    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return True


@dataclass
class Repeat(ProcessingStage[SimpleTask, SimpleTask]):
    """
    Dummy stage that returns `times` new instances of the incoming task.
    """

    times: int = 3
    name: str = "repeat"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: SimpleTask) -> list[SimpleTask]:
        # Construct fresh Task objects. task_id is assigned later by the
        # executor adapter, not by process()/process_batch.
        return [
            SimpleTask(
                dataset_name=task.dataset_name,
                data=task.data,
                _metadata=task._metadata.copy(),
                _stage_perf=task._stage_perf.copy(),
            )
            for _ in range(self.times)
        ]


def _sample_task() -> SimpleTask:
    return SimpleTask(dataset_name="test", data=[1, 2, 3])


def test_default_process_batch_does_not_assign_task_id():
    """``process_batch`` (and ``process``) do not touch ``task_id`` — that's
    the executor adapter's job (``BaseStageAdapter._post_process_task_ids``).
    So fanned-out children come back here with empty ids; task_id assignment
    is covered in tests/backends/test_task_id_postprocess.py."""
    task = _sample_task()
    output = Repeat(times=3).process_batch([task])

    assert len(output) == 3
    assert all(t.task_id == "" for t in output)


class TestSetTaskId:
    """``Task._set_task_id``: the id is the parent id and this task's own
    segment joined by ``"_"`` (no hashing)."""

    def test_no_parent_uses_suffix_only(self) -> None:
        t = _sample_task()
        t._set_task_id("", 3)
        # An empty parent id is dropped, so no leading "_".
        assert t.task_id == "3"

    def test_joins_parent_and_suffix(self) -> None:
        t = _sample_task()
        t._set_task_id("0", 7)
        assert t.task_id == "0_7"

    def test_always_overwrites(self) -> None:
        """No idempotency — each stage boundary re-derives the id, so one
        object passing through N stages gets N distinct task_ids."""
        t = _sample_task()
        t._set_task_id("", 0)
        t._set_task_id("0", 7)
        assert t.task_id == "0_7"

    def test_string_suffix(self) -> None:
        """Source stages pass a content-based hash (str) as the suffix
        instead of a positional index."""
        t = _sample_task()
        t._set_task_id("root", "abc123")
        assert t.task_id == "root_abc123"


def test_get_deterministic_id_defaults_to_none():
    """Base ``Task`` has no content identity, so source stages fall back to
    the positional index. ``FileGroupTask`` overrides this — see
    tests/tasks/test_file_group_tasks.py."""
    assert _sample_task().get_deterministic_id() is None
