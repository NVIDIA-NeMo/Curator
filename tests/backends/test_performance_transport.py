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

from unittest.mock import patch

import pytest

from nemo_curator.backends.base import BaseExecutor, BaseStageAdapter
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, EmptyTask, Task


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


def test_adapter_publishes_one_collector_record() -> None:
    stage = _Stage()
    stage._curator_stage_id = "000:stage"
    stage._curator_stage_perf_collector_name = "collector"
    adapter = BaseStageAdapter(stage)

    with patch("nemo_curator.utils.stage_perf_collector.record_stage_perf", return_value=True) as publish:
        [result] = adapter.process_batch([EmptyTask()])

    [perf] = result._stage_perf
    assert perf.stage_id == "000:stage"
    assert perf.invocation_id
    assert perf.window_end_s >= perf.window_start_s > 0
    publish.assert_called_once()
    assert publish.call_args.kwargs["attached_to_output"] is True


def test_required_collector_start_failure_is_not_silenced() -> None:
    with (
        patch(
            "nemo_curator.utils.stage_perf_collector.start_stage_perf_collector",
            side_effect=OSError("start failed"),
        ),
        pytest.raises(RuntimeError, match="Required stage performance collector failed to start"),
    ):
        _Executor._start_stage_perf_collector([_RequiredReportStage()])


def test_audio_input_byte_count_is_independent_of_item_count() -> None:
    stage = _Stage()
    stage._curator_stage_perf_collector_name = "collector"
    adapter = BaseStageAdapter(stage)
    small = AudioTask(dataset_name="test", data={"text": "a"})
    large = AudioTask(dataset_name="test", data={"text": "a" * 1_000})

    with patch("nemo_curator.utils.stage_perf_collector.record_stage_perf", return_value=True):
        [small_result] = adapter.process_batch([small])
        [large_result] = adapter.process_batch([large])

    small_perf = small_result._stage_perf[-1]
    large_perf = large_result._stage_perf[-1]
    assert small_perf.num_items_processed == large_perf.num_items_processed == 1
    assert large_perf.input_data_size_mb > small_perf.input_data_size_mb > 0
