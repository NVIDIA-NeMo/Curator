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

from dataclasses import dataclass, field

import pytest
from pytest import MonkeyPatch

import nemo_curator.backends.base as base_module
from nemo_curator.backends.slurm_array import SlurmArrayConfig
from nemo_curator.stages.audio.metrics.performance import AudioPerformanceSummary
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import EmptyTask, FileGroupTask, Task
from nemo_curator.tasks.sentinels import FailedTask


@dataclass
class _SourceFanoutStage(ProcessingStage[Task, FileGroupTask]):
    name: str = "source"
    is_source_stage: bool = True
    partitions: list[list[str]] = field(default_factory=list)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: Task) -> list[FileGroupTask]:
        return [FileGroupTask(dataset_name="d", data=list(partition)) for partition in self.partitions]


@dataclass
class _FailedSourceStage(ProcessingStage[Task, Task]):
    name: str = "source"
    is_source_stage: bool = True

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: Task) -> Task:
        return FailedTask()


class TestBaseStageAdapter:
    def test_attaches_planned_stage_and_invocation_identity(self) -> None:
        stage = _SourceFanoutStage(partitions=[["a.parquet"]])
        stage._curator_stage_id = "000:source"

        [result] = base_module.BaseStageAdapter(stage).process_batch([EmptyTask()])

        [perf] = result._stage_perf
        assert perf.stage_id == "000:source"
        assert perf.invocation_id
        assert perf.window_end_s >= perf.window_start_s > 0
        assert "stage_id" not in dict(perf.items())
        assert "invocation_id" not in dict(perf.items())

    def test_cross_host_envelope_uses_wall_clock_not_monotonic_epoch(self, monkeypatch: MonkeyPatch) -> None:
        class _Clock:
            def __init__(self, *, monotonic_start: float) -> None:
                self._perf_values = iter((monotonic_start, monotonic_start + 10.0))
                self._wall_values = iter((1_800_000_000.0, 1_800_000_010.0))

            def perf_counter(self) -> float:
                return next(self._perf_values)

            def time(self) -> float:
                return next(self._wall_values)

        records = []
        for monotonic_start, host in ((1000.0, "host-a"), (10.0, "host-b")):
            monkeypatch.setattr(base_module, "time", _Clock(monotonic_start=monotonic_start))
            stage = _SourceFanoutStage(partitions=[[f"{host}.parquet"]])
            stage._curator_stage_id = "000:source"
            [result] = base_module.BaseStageAdapter(stage).process_batch([EmptyTask()])
            [perf] = result._stage_perf
            assert perf.process_time == 10.0
            assert perf.window_start_s == 1_800_000_000.0
            assert perf.window_end_s == 1_800_000_010.0
            perf.actor_id = host
            perf.node_id = host
            perf.physical_address = f"{host}:0"
            perf.gpu_indices = [0]
            perf.custom_metrics = {"audio_duration_s": 3600.0}
            records.append(perf)

        summary = AudioPerformanceSummary()
        summary.record_stage_perf(records)
        stage_summary = summary.build_stage_summaries()["000:source"]

        # Two simultaneous ten-second invocations on two GPUs process two audio
        # hours. The arbitrary local monotonic epochs (1000 and 10) cannot turn
        # the stage envelope into 1000 seconds.
        assert stage_summary["total_process_time_s"] == 20.0
        assert stage_summary["gpu_count"] == 2.0
        assert stage_summary["audio_hours_per_gpu_hour"] == 360.0

    def test_process_batch_delegates_slurm_array_filtering(self, monkeypatch: MonkeyPatch) -> None:
        calls = {}
        slurm_array = SlurmArrayConfig(shard_index=0, total_shards=1)

        def resolve_config(is_source_stage: bool) -> SlurmArrayConfig:
            calls["is_source_stage"] = is_source_stage
            return slurm_array

        def filter_tasks(
            tasks: list[Task],
            resolved_slurm_array: SlurmArrayConfig,
            stage_name: str,
        ) -> list[Task]:
            calls["task_count"] = len(tasks)
            calls["filter_stage_name"] = stage_name
            calls["filter_slurm_array"] = resolved_slurm_array
            return tasks[:1]

        monkeypatch.setattr(base_module, "resolve_slurm_array_config", resolve_config)
        monkeypatch.setattr(base_module, "filter_slurm_array_source_tasks", filter_tasks)

        output = base_module.BaseStageAdapter(_SourceFanoutStage(partitions=[["a.parquet"], ["b.parquet"]]))
        results = output.process_batch([EmptyTask()])

        assert calls == {
            "is_source_stage": True,
            "task_count": 2,
            "filter_stage_name": "source",
            "filter_slurm_array": slurm_array,
        }
        assert [task.data for task in results] == [["a.parquet"]]

    def test_source_stage_failed_task_raises_before_retry_bookkeeping(self, monkeypatch: MonkeyPatch) -> None:
        calls = {"resolve_config": 0, "record_failed_tasks": 0, "filter_tasks": 0}
        slurm_array = SlurmArrayConfig(shard_index=0, total_shards=1)

        def resolve_config(_is_source_stage: bool) -> SlurmArrayConfig:
            calls["resolve_config"] += 1
            return slurm_array

        def filter_tasks(
            tasks: list[Task],
            _resolved_slurm_array: SlurmArrayConfig,
            _stage_name: str,
        ) -> list[Task]:
            calls["filter_tasks"] += 1
            return tasks

        def record_failed_tasks() -> None:
            calls["record_failed_tasks"] += 1

        monkeypatch.setattr(base_module, "resolve_slurm_array_config", resolve_config)
        monkeypatch.setattr(base_module, "filter_slurm_array_source_tasks", filter_tasks)
        monkeypatch.setattr(base_module, "record_failed_tasks", record_failed_tasks)

        with pytest.raises(ValueError, match="Source stage source emitted FailedTask"):
            base_module.BaseStageAdapter(_FailedSourceStage()).process_batch([EmptyTask()])

        assert calls == {
            "resolve_config": 0,
            "record_failed_tasks": 0,
            "filter_tasks": 0,
        }
