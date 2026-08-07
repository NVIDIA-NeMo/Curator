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

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nemo_curator.backends.base import BaseExecutor, BaseStageAdapter
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, EmptyTask, Task
from nemo_curator.utils.stage_perf_collector import PerformanceRecordStore


class _Stage(ProcessingStage[Task, Task]):
    name = "stage"

    def process(self, task: Task) -> Task:
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


def test_adapter_publishes_one_collector_record_without_task_duplication() -> None:
    stage = _Stage()
    stage._curator_stage_id = "000:stage"
    stage._curator_stage_perf_collector_name = "collector"
    adapter = BaseStageAdapter(stage)

    with patch("nemo_curator.utils.stage_perf_collector.record_stage_perf", return_value=True) as publish:
        [result] = adapter.process_batch([EmptyTask()])

    assert result._stage_perf == []
    perf = publish.call_args.args[1]
    assert perf.stage_id == "000:stage"
    assert perf.invocation_id
    assert perf.window_end_s >= perf.window_start_s > 0
    publish.assert_called_once()


def test_adapter_publishes_zero_output_invocation_once() -> None:
    stage = _ZeroOutputStage()
    stage._curator_stage_id = "000:zero-output"
    stage._curator_stage_perf_collector_name = "collector"

    with patch("nemo_curator.utils.stage_perf_collector.record_stage_perf", return_value=True) as publish:
        results = BaseStageAdapter(stage).process_batch([EmptyTask()])

    assert results == []
    publish.assert_called_once()
    perf = publish.call_args.args[1]
    assert perf.stage_id == "000:zero-output"
    assert perf.invocation_id


def test_adapter_publishes_fan_out_invocation_once_without_task_duplication() -> None:
    stage = _FanOutStage()
    stage._curator_stage_id = "000:fan-out"
    stage._curator_stage_perf_collector_name = "collector"

    with patch("nemo_curator.utils.stage_perf_collector.record_stage_perf", return_value=True) as publish:
        results = BaseStageAdapter(stage).process_batch([EmptyTask()])

    assert len(results) == 2
    assert all(result._stage_perf == [] for result in results)
    publish.assert_called_once()
    perf = publish.call_args.args[1]
    assert perf.stage_id == "000:fan-out"
    assert perf.invocation_id


def test_adapter_preserves_task_attached_perf_without_collector() -> None:
    [result] = BaseStageAdapter(_Stage()).process_batch([EmptyTask()])

    assert len(result._stage_perf) == 1
    assert result._stage_perf[0].stage_name == "stage"


def test_required_collector_start_failure_is_not_silenced() -> None:
    with (
        patch(
            "nemo_curator.utils.stage_perf_collector.start_stage_perf_collector",
            side_effect=OSError("start failed"),
        ),
        pytest.raises(RuntimeError, match="Required stage performance collector failed to start"),
    ):
        _Executor._start_stage_perf_collector([_RequiredReportStage()])


def test_executor_resolves_report_request_once_for_start_and_stop() -> None:
    executor = _Executor()
    handle = MagicMock(report_required=True)
    with (
        patch(
            "nemo_curator.utils.stage_perf_collector.performance_report_requested",
            return_value=True,
        ) as requested,
        patch("nemo_curator.utils.stage_perf_collector.start_stage_perf_collector", return_value=handle),
        patch(
            "nemo_curator.utils.stage_perf_collector.stop_stage_perf_collector",
            return_value=PerformanceRecordStore(),
        ),
    ):
        collector = executor._start_stage_perf_collector([_RequiredReportStage()])
        executor._stop_stage_perf_collector(collector, [_RequiredReportStage()], keep_records=True)

    requested.assert_called_once()


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


@pytest.mark.parametrize(
    "waveform",
    [
        pytest.param(torch.zeros(16, dtype=torch.float32), id="torch"),
        pytest.param(np.zeros(16, dtype=np.float32), id="numpy"),
    ],
)
def test_audio_input_byte_count_handles_in_memory_waveforms(waveform: object) -> None:
    stage = _Stage()
    stage._curator_stage_perf_collector_name = "collector"
    task = AudioTask(dataset_name="test", data={"text": "payload", "waveform": waveform})
    envelope_bytes = len(
        json.dumps(
            {"text": "payload", "waveform": None},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    expected_bytes = envelope_bytes + 16 * 4

    with patch("nemo_curator.utils.stage_perf_collector.record_stage_perf", return_value=True) as publish:
        [result] = BaseStageAdapter(stage).process_batch([task])

    assert result._stage_perf == []
    assert task.input_data_size_bytes() == expected_bytes
    perf = publish.call_args.args[1]
    assert perf.input_data_size_mb == pytest.approx(expected_bytes / 1024 / 1024)
