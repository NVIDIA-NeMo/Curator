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

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

import pytest

from nemo_curator.backends import perf_telemetry
from nemo_curator.backends.base import BaseStageAdapter, WorkerMetadata
from nemo_curator.backends.perf_identity import (
    PerformanceTelemetryAdapterMixin,
    WorkerPerfIdentity,
    build_ray_perf_identity,
    build_xenna_perf_identity,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import Task
from nemo_curator.utils import gpu_sampler
from nemo_curator.utils.performance_utils import StagePerfStats


def _fake_nvml(uuids: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: len(uuids),
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetUUID=lambda index: uuids[index],
    )


def test_xenna_identity_retains_all_assigned_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    allocation = SimpleNamespace(
        node="fallback",
        gpus=[SimpleNamespace(index=0), SimpleNamespace(index=1)],
    )
    monkeypatch.setenv("POD_IP", "10.0.0.5")
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml(["GPU-a", "GPU-b"]))

    identity = build_xenna_perf_identity(
        "infer",
        worker_id="workerabcdef",
        node_id="node-0",
        allocation=allocation,
        requires_gpu=True,
    )

    assert identity.actor_id == "infer:actor-workerab"
    assert identity.gpu_id == "node-0:0"
    assert identity.physical_address == "10.0.0.5:0,1"
    assert identity.gpu_indices == (0, 1)
    assert identity.gpu_uuids == ("GPU-a", "GPU-b")


@pytest.mark.parametrize(
    ("assignment", "expected_indices", "expected_uuids"),
    [
        ([0, 1], (0, 1), ("GPU-a", "GPU-b")),
        (["GPU-a", "GPU-b"], (0, 1), ("GPU-a", "GPU-b")),
    ],
)
def test_ray_identity_maps_integer_and_uuid_assignments(
    monkeypatch: pytest.MonkeyPatch,
    assignment: list[int] | list[str],
    expected_indices: tuple[int, ...],
    expected_uuids: tuple[str, ...],
) -> None:
    context = SimpleNamespace(
        get_node_id=lambda: "nodeabcdef",
        get_actor_id=lambda: "actorabcdef",
        get_worker_id=lambda: "",
    )
    fake_ray = SimpleNamespace(
        is_initialized=lambda: True,
        get_runtime_context=lambda: context,
        get_gpu_ids=lambda: assignment,
        util=SimpleNamespace(get_node_ip_address=lambda: "10.0.0.5"),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml(["GPU-a", "GPU-b"]))

    identity = build_ray_perf_identity("infer", requires_gpu=True)

    assert identity.actor_id == "infer:actor-actorabc"
    assert identity.node_id == "node-nodeabcd"
    assert identity.physical_address == "10.0.0.5:0,1"
    assert identity.gpu_indices == expected_indices
    assert identity.gpu_uuids == expected_uuids


@dataclass
class _Task(Task[list[int]]):
    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return True


class _GpuStage(ProcessingStage[Task, Task]):
    name = "gpu_stage"
    resources = Resources(gpus=1.0)
    extended_performance_metrics = True

    def process(self, task: Task) -> Task:
        self._log_metric("stage_metric", 2.0)
        return task


class _TelemetryAdapter(PerformanceTelemetryAdapterMixin, BaseStageAdapter):
    pass


class _Sampler:
    calls: ClassVar[list[dict[str, object]]] = []
    stops = 0

    def __init__(self, **kwargs: object) -> None:
        self.calls.append(kwargs)

    def start(self) -> None:
        pass

    def window_metrics(self, _start: float, _end: float) -> dict[str, float]:
        return {"gpu_util_pct::a": 75.0}

    def stop(self) -> None:
        type(self).stops += 1


def test_backend_adapter_owns_opt_in_sampler_identity_and_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    _Sampler.calls.clear()
    _Sampler.stops = 0
    monkeypatch.setattr(gpu_sampler, "GpuUtilSampler", _Sampler)
    adapter = _TelemetryAdapter(_GpuStage())
    adapter.setup(
        WorkerMetadata(
            perf_identity=WorkerPerfIdentity(
                actor_id="gpu_stage:actor-a",
                node_id="node-a",
                gpu_indices=(1,),
                gpu_uuids=("GPU-a",),
            )
        )
    )

    result = adapter.process_batch([_Task(dataset_name="test", data=[1, 2])])
    perf = result[0]._stage_perf[-1]
    adapter.teardown()

    assert _Sampler.calls == [{"gpu_uuids": ("GPU-a",), "sample_all_visible": False}]
    assert _Sampler.stops == 1
    assert perf.actor_id == "gpu_stage:actor-a"
    assert perf.gpu_indices == [1]
    assert perf.custom_metrics == {"stage_metric": 2.0, "gpu_util_pct::a": 75.0}


class _RemoteMethod:
    def __init__(self, value: object) -> None:
        self.value = value

    def remote(self) -> object:
        return self.value


class _RemoteClass:
    options_call: ClassVar[dict[str, object]] = {}

    def options(self, **kwargs: object) -> "_RemoteClass":
        type(self).options_call = kwargs
        return self

    def remote(self, _interval: float) -> SimpleNamespace:
        return SimpleNamespace(get_node_id=_RemoteMethod("node-ref"))


def test_pipeline_sampler_uses_node_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ray = SimpleNamespace(
        remote=lambda **_kwargs: lambda _class: _RemoteClass(),
        nodes=lambda: [{"Alive": True, "NodeID": "node-id"}],
        wait=lambda refs, **_kwargs: (list(refs), []),
        get=lambda ref: ref,
    )
    monkeypatch.setattr(perf_telemetry, "ray", fake_ray)
    monkeypatch.setattr(perf_telemetry, "_node_affinity_strategy", lambda node_id: {"node_id": node_id})

    actors = perf_telemetry.start_pipeline_hardware_samplers(0.5, 5.0)

    assert len(actors) == 1
    assert _RemoteClass.options_call == {"scheduling_strategy": {"node_id": "node-id"}}


def test_pipeline_hardware_record_appends_to_shared_collector_records() -> None:
    executor = perf_telemetry.PerformanceTelemetryExecutorMixin()
    existing = StagePerfStats(stage_name="ASR")
    hardware = StagePerfStats(stage_name="pipeline_hardware_sampler")
    executor._external_perf_records = [existing]

    with mock.patch.object(executor, "_stop_pipeline_hardware_sampler", return_value=hardware):
        executor._finalize_pipeline_hardware_sampler([mock.sentinel.actor], keep_record=True)

    assert executor._external_perf_records == [existing, hardware]


def test_pipeline_sampler_covers_each_live_node(shared_ray_client: None) -> None:  # noqa: ARG001
    import ray

    expected = {str(node["NodeID"]) for node in ray.nodes() if node.get("Alive") and node.get("NodeID")}
    actors = perf_telemetry.start_pipeline_hardware_samplers(0.5, 5.0)
    try:
        actual = set(ray.get([actor.get_node_id.remote() for actor in actors]))
        assert actual == expected
    finally:
        perf_telemetry.stop_pipeline_hardware_samplers(actors, 10.0)
