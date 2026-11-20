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

import os
from pathlib import Path
from unittest import mock

import pytest
from ray.data import DataContext, ExecutionResources

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import FileGroupTask


class RecordBatchTypeStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Task stage (no setup): records the type of the batch it receives."""

    name = "record_batch_type"
    batch_size = 3

    def process(self, task: FileGroupTask) -> FileGroupTask:
        return task

    def process_batch(self, tasks: list[FileGroupTask]) -> list[FileGroupTask]:
        for task in tasks:
            task.data.append(type(tasks).__name__)
        return tasks


class TeardownStage(RecordBatchTypeStage):
    """Actor stage (overrides setup): writes a marker file per actor on teardown."""

    name = "teardown_stage"

    def __init__(self, marker_dir: str):
        self.marker_dir = marker_dir

    def setup(self, worker_metadata: object = None) -> None:
        self.num_rows = 0

    def process_batch(self, tasks: list[FileGroupTask]) -> list[FileGroupTask]:
        self.num_rows += len(tasks)
        return super().process_batch(tasks)

    def teardown(self) -> None:
        Path(self.marker_dir, f"teardown_{os.getpid()}").write_text(str(self.num_rows))


def _tasks(n: int) -> list[FileGroupTask]:
    return [FileGroupTask(dataset_name="ds", data=[str(i)]) for i in range(n)]


def test_stages_get_lists_and_actor_teardown_runs(tmp_path: Path) -> None:
    stages = [RecordBatchTypeStage(), TeardownStage(str(tmp_path)).with_(num_workers=2)]

    output = RayDataExecutor().execute(stages, _tasks(8))

    assert sorted(task.data[0] for task in output) == [str(i) for i in range(8)]
    assert all(task.data[1:] == ["list", "list"] for task in output)
    markers = list(tmp_path.glob("teardown_*"))
    assert len(markers) == 2  # one per actor
    assert sum(int(marker.read_text()) for marker in markers) == 8


@pytest.mark.parametrize(
    ("user_options", "expected_gpu_limit"),
    [
        ({}, 3),
        ({"resource_limits": ExecutionResources.for_limits(gpu=2)}, 2),
        ({"exclude_resources": ExecutionResources(gpu=1)}, float("inf")),
    ],
)
def test_gpu_limit_excludes_inference_server_gpus(
    user_options: dict, expected_gpu_limit: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = DataContext.get_current().execution_options
    for name, value in user_options.items():
        monkeypatch.setattr(options, name, value)
    previous = options.resource_limits
    seen_gpu_limits = []

    def record_limit(self: RayDataExecutor, tasks: list) -> object:  # noqa: ARG001
        seen_gpu_limits.append(DataContext.get_current().execution_options.resource_limits.gpu)
        msg = "stop after recording"
        raise RuntimeError(msg)

    with (
        mock.patch("nemo_curator.backends.ray_data.executor.is_inference_server_active", return_value=True),
        mock.patch("nemo_curator.backends.ray_data.executor.get_available_cpu_gpu_resources", return_value=(8, 3)),
        mock.patch.object(RayDataExecutor, "_tasks_to_dataset", record_limit),
        pytest.raises(RuntimeError, match="stop after recording"),
    ):
        RayDataExecutor().execute([RecordBatchTypeStage()], _tasks(2))

    assert seen_gpu_limits == [expected_gpu_limit]
    assert options.resource_limits == previous
