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

from dataclasses import dataclass

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class _CustomMetricStage(ProcessingStage[AudioTask, AudioTask]):
    name: str = "custom_metric_stage"

    def process(self, task: AudioTask) -> AudioTask:
        msg = "_CustomMetricStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        self._log_metrics({"stage_specific_metric": 3.0})
        return tasks


def test_base_stage_adapter_attaches_stage_metrics_without_global_audio_counters() -> None:
    task = AudioTask(task_id="utt-1", data={"duration": 1.0})
    adapter = BaseStageAdapter(_CustomMetricStage())

    result = adapter.process_batch([task])

    perf = result[0]._stage_perf[-1]
    assert perf.stage_name == "custom_metric_stage"
    assert perf.custom_metrics == {"stage_specific_metric": 3.0}
    assert "input_tasks" not in perf.custom_metrics
    assert "output_tasks" not in perf.custom_metrics
    assert not hasattr(adapter, "_log_stage_invocation_metrics")
