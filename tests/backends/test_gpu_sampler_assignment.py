# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import ClassVar
from unittest import mock

from nemo_curator.backends.base import BaseStageAdapter, WorkerMetadata
from nemo_curator.backends.perf_identity import ExtendedPerfStageAdapterMixin, WorkerPerfIdentity
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import Task
from nemo_curator.utils import gpu_sampler


class _TelemetryAdapter(ExtendedPerfStageAdapterMixin, BaseStageAdapter):
    pass


class _GpuStage(ProcessingStage[Task, Task]):
    name = "gpu_stage"
    resources = Resources(gpus=1.0)
    extended_performance_metrics = True

    def process(self, task: Task) -> Task:
        self._log_metric("stage_metric", 2.0)
        return task


class _FilteringGpuStage(_GpuStage):
    def process(self, task: Task) -> None:
        self._log_metric("filtered_items", task.num_items)


@dataclass
class _Task(Task[list[int]]):
    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return True


class _FakeSampler:
    calls: ClassVar[list[dict[str, object]]] = []
    stops: ClassVar[int] = 0

    def __init__(self, **kwargs: object) -> None:
        self.calls.append(kwargs)

    def start(self) -> None:
        return None

    def window_metrics(self, _window_start: float, _window_end: float) -> dict[str, float]:
        return {"gpu_util_pct::a": 75.0}

    def stop(self) -> None:
        type(self).stops += 1


def test_actor_sampler_targets_only_assigned_gpu_uuids(monkeypatch) -> None:  # noqa: ANN001
    _FakeSampler.calls.clear()
    monkeypatch.setattr(gpu_sampler, "GpuUtilSampler", _FakeSampler)
    adapter = _TelemetryAdapter(_GpuStage())
    adapter._perf_identity = WorkerPerfIdentity(gpu_uuids=("GPU-a", "GPU-b"))

    sampler = adapter._maybe_start_gpu_sampler()

    assert isinstance(sampler, _FakeSampler)
    assert _FakeSampler.calls == [{"gpu_uuids": ("GPU-a", "GPU-b"), "sample_all_visible": False}]


def test_actor_sampler_does_not_guess_when_gpu_assignment_is_unknown(monkeypatch) -> None:  # noqa: ANN001
    _FakeSampler.calls.clear()
    monkeypatch.setattr(gpu_sampler, "GpuUtilSampler", _FakeSampler)
    adapter = _TelemetryAdapter(_GpuStage())
    adapter._perf_identity = WorkerPerfIdentity()

    assert adapter._maybe_start_gpu_sampler() is None
    assert _FakeSampler.calls == []


def test_extended_perf_attaches_invocation_identity_metrics_and_stops_sampler(monkeypatch) -> None:  # noqa: ANN001
    _FakeSampler.calls.clear()
    _FakeSampler.stops = 0
    monkeypatch.setattr(gpu_sampler, "GpuUtilSampler", _FakeSampler)
    adapter = _TelemetryAdapter(_GpuStage())
    adapter.setup(
        WorkerMetadata(
            actor_id="gpu_stage:actor-a",
            node_id="node-a",
            gpu_id="node-a:1",
            physical_address="host-a:1",
            gpu_indices=[1],
            gpu_uuids=["GPU-a"],
        )
    )

    result = adapter.process_batch([_Task(dataset_name="test", data=[1, 2])])
    perf = result[0]._stage_perf[-1]
    adapter.teardown()

    assert perf.invocation_id
    assert perf.actor_id == "gpu_stage:actor-a"
    assert perf.node_id == "node-a"
    assert perf.gpu_indices == [1]
    assert perf.gpu_uuids == ["GPU-a"]
    assert perf.custom_metrics == {"stage_metric": 2.0, "gpu_util_pct::a": 75.0}
    assert _FakeSampler.stops == 1


def test_extended_perf_publishes_fully_filtered_invocation_out_of_band() -> None:
    stage = _FilteringGpuStage()
    stage._curator_stage_id = "0002:gpu_stage"
    adapter = _TelemetryAdapter(stage)
    adapter._perf_identity = WorkerPerfIdentity(actor_id="actor")

    with mock.patch("nemo_curator.utils.stage_perf_collector.record_stage_perf", return_value=True) as publish:
        results = adapter.process_batch([_Task(dataset_name="test", data=[1, 2])])

    assert results == []
    perf = publish.call_args.args[1]
    assert perf.stage_id == "0002:gpu_stage"
    assert perf.invocation_id
    assert perf.custom_metrics["filtered_items"] == 2.0
    assert publish.call_args.kwargs == {"attached_to_output": False}
