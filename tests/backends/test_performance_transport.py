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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nemo_curator.backends.base import BaseExecutor, BaseStageAdapter
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, EmptyTask, Task
from nemo_curator.utils.stage_perf_collector import COLLECTOR_ACTOR_ATTR
from tests.utils.performance_record_store import make_performance_record_store


class _Stage(ProcessingStage[Task, Task]):
    name = "stage"

    def process(self, task: Task) -> Task:
        return task


class _MetricStage(_Stage):
    def process(self, task: Task) -> Task:
        self._log_metric("metric", 2.5)
        return task


class _Executor(BaseExecutor):
    _supports_stage_perf_collection = True

    def execute(self, stages: list[ProcessingStage], initial_tasks: list[Task] | None = None) -> None:
        return None


class _UnsupportedExecutor(BaseExecutor):
    def execute(self, stages: list[ProcessingStage], initial_tasks: list[Task] | None = None) -> None:
        return None


class _RequiredReportStage(_Stage):
    def requests_performance_records(self) -> bool:
        return True


class _ZeroOutputStage(ProcessingStage[Task, Task]):
    name = "zero-output"

    def process(self, task: Task) -> None:
        return None


class _FanOutStage(ProcessingStage[Task, Task]):
    name = "fan-out"

    def process(self, task: Task) -> list[Task]:
        return [task, EmptyTask()]


class _BatchStage(ProcessingStage[Task, Task]):
    name = "batch"

    def process(self, task: Task) -> Task:
        return task

    def process_batch(self, tasks: list[Task]) -> list[Task]:
        return tasks


@pytest.mark.parametrize(
    ("stage", "expected_output_count"),
    [
        pytest.param(_Stage(), 1, id="one-output"),
        pytest.param(_ZeroOutputStage(), 0, id="zero-output"),
        pytest.param(_FanOutStage(), 2, id="fan-out"),
        pytest.param(_BatchStage(), 1, id="explicit-process-batch"),
    ],
)
def test_adapter_publishes_once_per_invocation_without_task_duplication(
    stage: ProcessingStage,
    expected_output_count: int,
) -> None:
    stage._curator_stage_id = f"000:{stage.name}"
    setattr(stage, COLLECTOR_ACTOR_ATTR, object())

    with patch("nemo_curator.utils.stage_perf_collector.record_stage_perf", return_value=True) as publish:
        results = BaseStageAdapter(stage).process_batch([AudioTask(dataset_name="test", data={"text": "payload"})])

    assert len(results) == expected_output_count
    assert all(result._stage_perf == [] for result in results)
    publish.assert_called_once()
    record = publish.call_args.args[1]
    assert record["stage_id"] == f"000:{stage.name}"
    assert record["invocation_id"]
    assert record["window_end_s"] >= record["window_start_s"] > 0
    assert record["num_items_processed"] == 1
    assert "input_data_size_mb" not in record


def test_adapter_disabled_collection_preserves_main_task_attached_stats() -> None:
    [result] = BaseStageAdapter(_MetricStage()).process_batch(
        [AudioTask(dataset_name="test", data={"text": "payload"})]
    )

    assert len(result._stage_perf) == 1
    perf = result._stage_perf[0]
    assert perf.to_dict().keys() == {
        "stage_name",
        "process_time",
        "actor_idle_time",
        "input_data_size_mb",
        "num_items_processed",
        "custom_metrics",
    }
    assert perf.stage_name == "stage"
    assert perf.num_items_processed == 1
    assert perf.input_data_size_mb == pytest.approx(1 / 1024 / 1024)
    assert perf.custom_metrics == {"metric": 2.5}
    assert not hasattr(perf, "invocation_id")
    assert not hasattr(perf, "window_start_s")


def test_publication_wait_is_not_charged_to_the_next_actor_idle_window() -> None:
    clock = SimpleNamespace(now=100.0)
    records: list[dict[str, object]] = []
    stage = _Stage()
    setattr(stage, COLLECTOR_ACTOR_ATTR, object())
    adapter = BaseStageAdapter(stage)

    def publish(_stage: ProcessingStage, record: dict[str, object]) -> None:
        records.append(record)
        clock.now += 10.0

    with (
        patch("nemo_curator.backends.base.time.time", side_effect=lambda: clock.now),
        patch("nemo_curator.backends.base.time.perf_counter", side_effect=lambda: clock.now),
        patch("nemo_curator.utils.stage_perf_collector.record_stage_perf", side_effect=publish),
    ):
        adapter.process_batch([AudioTask(dataset_name="test", data={"text": "first"})])
        clock.now += 2.0
        adapter.process_batch([AudioTask(dataset_name="test", data={"text": "second"})])

    assert records[1]["actor_idle_time"] == pytest.approx(2.0)


def test_required_collector_start_failure_is_not_silenced() -> None:
    executor = _Executor()
    executor._set_stage_perf_collection_requested(True)
    with (
        patch(
            "nemo_curator.utils.stage_perf_collector.start_stage_perf_collector",
            side_effect=OSError("start failed"),
        ),
        pytest.raises(RuntimeError, match="Required stage performance collector failed to start"),
    ):
        executor._start_stage_perf_collector([_RequiredReportStage()])


def test_unsupported_executor_rejects_required_collection() -> None:
    executor = _UnsupportedExecutor()

    with pytest.raises(NotImplementedError, match="does not support run-scoped stage performance collection"):
        executor._set_stage_perf_collection_requested(True)


def test_executor_transfers_external_records_exactly_once() -> None:
    executor = _Executor()
    expected_record = {"stage_name": "stage", "invocation_id": "invocation-1", "process_time": 1.0}
    records = make_performance_record_store([expected_record])
    spool_path = Path(records.path)
    executor._external_perf_records = records

    transferred = executor.consume_external_perf_records()
    assert transferred is records
    assert spool_path.is_file()
    assert list(transferred) == [expected_record]

    assert len(executor.consume_external_perf_records()) == 0
    assert spool_path.is_file()

    transferred.cleanup()
    assert not spool_path.exists()
