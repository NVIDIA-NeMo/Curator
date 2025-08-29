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
    _name: str = "repeat"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: SimpleTask) -> list[SimpleTask]:
        # Important: construct fresh Task objects so each gets a fresh _uuid
        return [
            SimpleTask(
                task_id=f"{task.task_id}_{i}",
                dataset_name=task.dataset_name,
                data=task.data,
                _metadata=task._metadata.copy(),
                _stage_perf=task._stage_perf.copy(),
            )
            for i in range(self.times)
        ]


def _sample_task() -> SimpleTask:
    return SimpleTask(task_id="t0", dataset_name="test", data=[1, 2, 3])


def test_fanout_tasks_have_unique_uuid():
    task = _sample_task()
    stage = Repeat(times=3)
    output = stage.process(task)

    assert len(output) == 3
    uuids = [t._uuid for t in output]
    assert len(set(uuids)) == 3, f"Expected unique _uuid per task, got {uuids}"
