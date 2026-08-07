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
from unittest.mock import patch

import pytest

from nemo_curator.backends.base import BaseExecutor, BaseStageAdapter
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, EmptyTask, Task
from nemo_curator.utils.performance_utils import StagePerfStats
from nemo_curator.utils.stage_perf_collector import PerformanceRecordStore


class _Stage(ProcessingStage[Task, Task]):
    name = "stage"

    def process(self, task: Task) -> Task:
        return task


class _MetricStage(_Stage):
    def process(self, task: Task) -> Task:
        self._log_metric("metric", 2.5)
        return task


class _Executor(BaseExecutor):
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


@pytest.mark.parametrize(
    ("stage", "expected_output_count"),
    [
        pytest.param(_Stage(), 1, id="one-output"),
        pytest.param(_ZeroOutputStage(), 0, id="zero-output"),
        pytest.param(_FanOutStage(), 2, id="fan-out"),
    ],
)
def test_adapter_publishes_once_per_invocation_without_task_duplication(
    stage: ProcessingStage,
    expected_output_count: int,
) -> None:
    stage._curator_stage_id = f"000:{stage.name}"
    stage._curator_stage_perf_collector_name = "collector"

    with patch("nemo_curator.utils.stage_perf_collector.record_stage_perf", return_value=True) as publish:
        results = BaseStageAdapter(stage).process_batch([EmptyTask()])

    assert len(results) == expected_output_count
    assert all(result._stage_perf == [] for result in results)
    publish.assert_called_once()
    perf = publish.call_args.args[1]
    assert perf.stage_id == f"000:{stage.name}"
    assert perf.invocation_id
    assert perf.window_end_s >= perf.window_start_s > 0


def test_adapter_disabled_collection_corrects_item_count_reported_as_byte_size() -> None:
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
    assert perf.input_data_size_mb == 0.0
    assert perf.custom_metrics == {"metric": 2.5}
    assert perf.invocation_id == ""
    assert perf.window_start_s == perf.window_end_s == 0.0


def test_required_collector_start_failure_is_not_silenced() -> None:
    with (
        patch(
            "nemo_curator.utils.stage_perf_collector.start_stage_perf_collector",
            side_effect=OSError("start failed"),
        ),
        pytest.raises(RuntimeError, match="Required stage performance collector failed to start"),
    ):
        _Executor._start_stage_perf_collector([_RequiredReportStage()])


def test_executor_transfers_external_records_exactly_once() -> None:
    executor = _Executor()
    expected_record = StagePerfStats(
        stage_name="stage",
        invocation_id="invocation-1",
        process_time=1.0,
    )
    records = PerformanceRecordStore.from_records([expected_record])
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


def test_audio_input_byte_count_is_independent_of_item_count() -> None:
    stage = _Stage()
    stage._curator_stage_perf_collector_name = "collector"
    adapter = BaseStageAdapter(stage)
    small = AudioTask(dataset_name="test", data={"text": "a"})
    large = AudioTask(dataset_name="test", data={"text": "a" * 1_000})

    with patch("nemo_curator.utils.stage_perf_collector.record_stage_perf", return_value=True) as publish:
        [small_result] = adapter.process_batch([small])
        [large_result] = adapter.process_batch([large])

    assert small_result._stage_perf == large_result._stage_perf == []
    small_perf = publish.call_args_list[0].args[1]
    large_perf = publish.call_args_list[1].args[1]
    assert small_perf.num_items_processed == large_perf.num_items_processed == 1
    assert large_perf.input_data_size_mb > small_perf.input_data_size_mb > 0
