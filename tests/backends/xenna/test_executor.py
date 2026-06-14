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

from typing import Any

from nemo_curator.backends.xenna.executor import XennaExecutor
from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, Task


class _PassthroughStage(ProcessingStage[AudioTask, AudioTask]):
    name = "passthrough"
    resources = Resources(cpus=1.0)
    batch_size = 2

    def process(self, task: AudioTask) -> AudioTask:
        return task


class _CentralizedStage(_PassthroughStage):
    name = "centralized"

    def __init__(self) -> None:
        self.batch_policy = BatchPolicy(
            buckets_sec=[0, 30, 1200],
            max_items_per_batch_by_bucket=[2, 1, 1],
            max_audio_sec_per_batch=None,
        )

    def build_prebucketed_tasks(self, tasks: list[AudioTask]) -> list[AudioTask]:
        return list(tasks)

    def scheduler_task_cost(self, task: AudioTask) -> float:
        return float(task.data.get("duration", 0.0))

    def assemble_prebucketed_task_results(
        self,
        _tasks: list[AudioTask],
        processed_tasks: list[AudioTask],
    ) -> list[AudioTask]:
        return processed_tasks


def test_xenna_executor_keeps_centralized_stage_inside_one_pipeline(monkeypatch) -> None:  # noqa: ANN001
    executor = XennaExecutor()
    stages: list[ProcessingStage[Any, Any]] = [
        _PassthroughStage(),
        _CentralizedStage(),
        _PassthroughStage(),
    ]
    initial_tasks = [AudioTask(data={"duration": 5.0})]
    calls: list[tuple[list[ProcessingStage[Any, Any]], list[Task]]] = []

    def fake_run_xenna_pipeline(
        stages_arg: list[ProcessingStage[Any, Any]],
        initial_tasks_arg: list[Task],
    ) -> list[Task]:
        calls.append((stages_arg, initial_tasks_arg))
        return initial_tasks_arg

    monkeypatch.setattr(executor, "_run_xenna_pipeline", fake_run_xenna_pipeline)

    out = executor.execute(stages, initial_tasks)

    assert out == initial_tasks
    assert calls == [(stages, initial_tasks)]
