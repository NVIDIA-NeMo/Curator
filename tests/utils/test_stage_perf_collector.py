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

from unittest.mock import MagicMock, patch

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import Task
from nemo_curator.utils.performance_utils import StagePerfStats
from nemo_curator.utils.stage_perf_collector import record_stage_perf


class _Stage(ProcessingStage):
    name = "stage"

    def process(self, task: Task) -> Task:
        return task


def test_record_stage_perf_is_noop_without_collector() -> None:
    assert record_stage_perf(_Stage(), StagePerfStats(stage_name="stage"), attached_to_output=False) is False


def test_record_stage_perf_waits_for_collector_ack() -> None:
    stage = _Stage()
    stage._curator_stage_perf_collector_name = "collector"
    collector = MagicMock()
    record_ref = collector.record.remote.return_value

    with (
        patch("nemo_curator.utils.stage_perf_collector.ray.get_actor", return_value=collector) as get_actor,
        patch("nemo_curator.utils.stage_perf_collector.ray.get") as ray_get,
    ):
        assert record_stage_perf(stage, StagePerfStats(stage_name="stage"), attached_to_output=True) is True

    get_actor.assert_called_once_with("collector")
    ray_get.assert_called_once_with(record_ref)
