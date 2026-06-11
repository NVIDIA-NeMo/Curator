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

import json
from dataclasses import dataclass

from nemo_curator.backends.base import FAILED_TASKS_DIR_ENV_VAR, BaseStageAdapter
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import Task
from nemo_curator.tasks.sentinels import FailedTask


@dataclass
class _FailedStage(ProcessingStage[Task, Task]):
    name: str = "failed"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: Task) -> Task:
        return FailedTask()


@dataclass
class _SimpleTask(Task[list[int]]):
    @property
    def num_items(self) -> int:
        return 0

    def validate(self) -> bool:
        return True


def _task(task_id: str = "") -> _SimpleTask:
    task = _SimpleTask(dataset_name="d", data=[])
    task.task_id = task_id
    return task


class TestBaseStageAdapter:
    def test_process_batch_writes_failed_task_marker_when_enabled(self, tmp_path, monkeypatch) -> None:
        marker_dir = tmp_path / "failed-tasks"
        monkeypatch.setenv(FAILED_TASKS_DIR_ENV_VAR, str(marker_dir))

        output = BaseStageAdapter(_FailedStage()).process_batch([_task("0_7")])

        assert output == []
        marker_files = list(marker_dir.glob("failed_task_*.json"))
        assert len(marker_files) == 1

        payload = json.loads(marker_files[0].read_text())
        assert payload["stage_name"] == "failed"
        assert payload["task_id"] == "0_7_0"
        assert payload["dataset_name"] == "failed"
        assert payload["task_type"] == "FailedTask"
        assert isinstance(payload["hostname"], str)
        assert isinstance(payload["pid"], int)
        assert isinstance(payload["created_at"], str)

    def test_process_batch_does_not_write_failed_task_marker_by_default(self, tmp_path, monkeypatch) -> None:
        marker_dir = tmp_path / "failed-tasks"
        monkeypatch.delenv(FAILED_TASKS_DIR_ENV_VAR, raising=False)

        output = BaseStageAdapter(_FailedStage()).process_batch([_task("0_7")])

        assert output == []
        assert not marker_dir.exists()
