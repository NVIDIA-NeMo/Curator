# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.backends.perf_identity import WorkerPerfIdentity
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import Task
from nemo_curator.utils import gpu_sampler


class _GpuStage(ProcessingStage[Task, Task]):
    name = "gpu_stage"
    resources = Resources(gpus=1.0)
    extended_performance_metrics = True

    def process(self, task: Task) -> Task:
        return task


class _FakeSampler:
    calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, **kwargs: object) -> None:
        self.calls.append(kwargs)

    def start(self) -> None:
        return None


def test_actor_sampler_targets_only_assigned_gpu_uuids(monkeypatch) -> None:  # noqa: ANN001
    _FakeSampler.calls.clear()
    monkeypatch.setattr(gpu_sampler, "GpuUtilSampler", _FakeSampler)
    adapter = BaseStageAdapter(_GpuStage())
    adapter._perf_identity = WorkerPerfIdentity(gpu_uuids=("GPU-a", "GPU-b"))

    sampler = adapter._maybe_start_gpu_sampler()

    assert isinstance(sampler, _FakeSampler)
    assert _FakeSampler.calls == [{"gpu_uuids": ("GPU-a", "GPU-b"), "sample_all_visible": False}]


def test_actor_sampler_does_not_guess_when_gpu_assignment_is_unknown(monkeypatch) -> None:  # noqa: ANN001
    _FakeSampler.calls.clear()
    monkeypatch.setattr(gpu_sampler, "GpuUtilSampler", _FakeSampler)
    adapter = BaseStageAdapter(_GpuStage())
    adapter._perf_identity = WorkerPerfIdentity()

    assert adapter._maybe_start_gpu_sampler() is None
    assert _FakeSampler.calls == []
