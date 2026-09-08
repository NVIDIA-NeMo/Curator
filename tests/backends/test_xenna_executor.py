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

import pytest

from nemo_curator.backends.xenna import executor as xenna_executor
from nemo_curator.backends.xenna.executor import XennaExecutor
from nemo_curator.stages.base import ProcessingStage, Resources
from nemo_curator.tasks import EmptyTask


class ConfigurableStage(ProcessingStage[EmptyTask, EmptyTask]):
    name = "configurable"
    resources = Resources(cpus=0.5)

    def __init__(
        self,
        *,
        num_workers: int | None = None,
        xenna_stage_spec: dict[str, object] | None = None,
    ) -> None:
        self._num_workers = num_workers
        self._xenna_stage_spec = xenna_stage_spec or {}

    def num_workers(self) -> int | None:
        return self._num_workers

    def xenna_stage_spec(self) -> dict[str, object]:
        return self._xenna_stage_spec

    def process(self, task: EmptyTask) -> EmptyTask:
        return task


def test_xenna_executor_uses_stage_num_workers_when_xenna_spec_has_no_worker_sizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = ConfigurableStage(num_workers=3, xenna_stage_spec={"ignore_failures": True})

    captured = _execute_and_capture_stage_spec(monkeypatch, stage)

    assert captured["num_workers"] == 3
    assert captured["num_workers_per_node"] is None
    assert captured["ignore_failures"] is True


def test_xenna_executor_accepts_num_workers_per_node_when_stage_num_workers_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = ConfigurableStage(xenna_stage_spec={"num_workers_per_node": 0.5})

    captured = _execute_and_capture_stage_spec(monkeypatch, stage)

    assert captured["num_workers"] is None
    assert captured["num_workers_per_node"] == 0.5


def test_xenna_executor_rejects_num_workers_with_num_workers_per_node(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = ConfigurableStage(num_workers=3, xenna_stage_spec={"num_workers_per_node": 0.5})

    with pytest.raises(ValueError, match=r"num_workers\(\).*num_workers_per_node"):
        _execute_and_capture_stage_spec(monkeypatch, stage)


def test_xenna_executor_rejects_num_workers_in_xenna_stage_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = ConfigurableStage(xenna_stage_spec={"num_workers": 3})

    with pytest.raises(ValueError, match="Use num_workers\\(\\) instead"):
        _execute_and_capture_stage_spec(monkeypatch, stage)


def _execute_and_capture_stage_spec(
    monkeypatch: pytest.MonkeyPatch,
    stage: ProcessingStage,
) -> dict[str, object]:
    captured: dict[str, object] = {}

    def record_stage_spec(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    def record_pipeline_spec(**_: object) -> object:
        return object()

    monkeypatch.setattr(xenna_executor.pipelines_v1, "StageSpec", record_stage_spec)
    monkeypatch.setattr(xenna_executor.pipelines_v1, "PipelineSpec", record_pipeline_spec)
    monkeypatch.setattr(xenna_executor.pipelines_v1, "run_pipeline", lambda _: [])
    monkeypatch.setattr(xenna_executor.ray, "init", lambda **_: None)
    monkeypatch.setattr(xenna_executor.ray, "shutdown", lambda: None)

    XennaExecutor().execute([stage], initial_tasks=[EmptyTask()])

    return captured
