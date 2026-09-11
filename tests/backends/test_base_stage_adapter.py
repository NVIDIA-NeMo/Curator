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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pytest import MonkeyPatch

import nemo_curator.backends.base as base_module
from nemo_curator.backends.slurm_array import SlurmArrayConfig
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, EmptyTask, FileGroupTask, Task
from nemo_curator.tasks.sentinels import FailedTask
from nemo_curator.utils import stage_perf_collector
from nemo_curator.utils.stage_perf_collector import COLLECTOR_ACTOR_ATTR


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


class _PerfStage(ProcessingStage[Task, Task]):
    name = "stage"

    def process(self, task: Task) -> Task:
        return task


class _ZeroOutputPerfStage(_PerfStage):
    name = "zero-output"

    def process(self, task: Task) -> None:
        self._log_metric("metric", 2.5)


class _FanOutPerfStage(_PerfStage):
    name = "fan-out"

    def process(self, task: Task) -> list[Task]:
        self._log_metric("metric", 2.5)
        return [task, EmptyTask()]


class TestBaseStageAdapter:
    @pytest.mark.parametrize(
        ("stage", "expected_output_count"),
        [
            pytest.param(_ZeroOutputPerfStage(), 0, id="zero-output"),
            pytest.param(_FanOutPerfStage(), 2, id="fan-out"),
        ],
    )
    def test_performance_record_is_published_once_not_copied_to_outputs(
        self,
        stage: ProcessingStage,
        expected_output_count: int,
        monkeypatch: MonkeyPatch,
    ) -> None:
        stage._curator_stage_id = f"000:{stage.name}"
        setattr(stage, COLLECTOR_ACTOR_ATTR, object())
        publish = Mock()
        monkeypatch.setattr(stage_perf_collector, "record_stage_perf", publish)

        results = base_module.BaseStageAdapter(stage).process_batch(
            [AudioTask(dataset_name="test", data={"waveform": torch.zeros(8)})]
        )

        assert len(results) == expected_output_count
        assert all(result._stage_perf == [] for result in results)
        publish.assert_called_once()
        record = publish.call_args.args[1]
        assert record["stage_id"] == f"000:{stage.name}"
        assert record["invocation_id"]
        assert record["window_end_s"] >= record["window_start_s"] > 0
        assert record["num_items_processed"] == 1
        assert record["custom_metrics"] == {"metric": 2.5}
        assert "input_data_size_mb" not in record

    def test_publication_wait_is_not_actor_idle_time(self, monkeypatch: MonkeyPatch) -> None:
        clock = SimpleNamespace(now=100.0)
        records: list[dict[str, object]] = []
        stage = _PerfStage()
        setattr(stage, COLLECTOR_ACTOR_ATTR, object())
        adapter = base_module.BaseStageAdapter(stage)

        def publish(_stage: ProcessingStage, record: dict[str, object]) -> None:
            records.append(record)
            clock.now += 10.0

        monkeypatch.setattr(base_module.time, "time", lambda: clock.now)
        monkeypatch.setattr(base_module.time, "perf_counter", lambda: clock.now)
        monkeypatch.setattr(stage_perf_collector, "record_stage_perf", publish)

        adapter.process_batch([AudioTask(dataset_name="test", data={"text": "first"})])
        clock.now += 2.0
        adapter.process_batch([AudioTask(dataset_name="test", data={"text": "second"})])

        assert records[1]["actor_idle_time"] == pytest.approx(2.0)

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
