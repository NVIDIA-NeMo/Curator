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

from benchmarking.scripts.task_metrics import sum_unique_stage_custom_metric


@dataclass
class _Perf:
    stage_name: str
    custom_metrics: dict[str, float]


@dataclass
class _Task:
    _stage_perf: list[_Perf] = field(default_factory=list)


@dataclass
class _WorkflowResult:
    pipeline_tasks: dict[str, list[_Task]]


def test_sum_unique_stage_custom_metric_counts_each_perf_record_once() -> None:
    first_perf = _Perf("KMeansStage", {"num_rows": 10})
    second_perf = _Perf("KMeansStage", {"num_rows": 20})
    workflow_result = _WorkflowResult(
        pipeline_tasks={
            "kmeans": [
                _Task([first_perf]),
                _Task([first_perf]),
                _Task([second_perf]),
            ]
        }
    )

    assert (
        sum_unique_stage_custom_metric(
            workflow_result, pipeline_name="kmeans", stage_name="KMeansStage", metric_name="num_rows"
        )
        == 30
    )


def test_sum_unique_stage_custom_metric_ignores_other_pipelines_stages_and_metrics() -> None:
    workflow_result = _WorkflowResult(
        pipeline_tasks={
            "kmeans": [_Task([_Perf("OtherStage", {"num_rows": 10}), _Perf("KMeansStage", {"other": 20})])],
            "pairwise": [_Task([_Perf("KMeansStage", {"num_rows": 30})])],
        }
    )

    assert (
        sum_unique_stage_custom_metric(
            workflow_result, pipeline_name="kmeans", stage_name="KMeansStage", metric_name="num_rows"
        )
        == 0
    )
