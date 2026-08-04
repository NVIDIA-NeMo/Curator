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

from collections.abc import Mapping, Sequence
from typing import Protocol


class _StagePerfLike(Protocol):
    stage_name: str
    custom_metrics: Mapping[str, float]


class _TaskLike(Protocol):
    _stage_perf: Sequence[_StagePerfLike]


class _WorkflowResultLike(Protocol):
    pipeline_tasks: Mapping[str, Sequence[_TaskLike]]


def sum_unique_stage_custom_metric(
    workflow_result: _WorkflowResultLike,
    *,
    pipeline_name: str,
    stage_name: str,
    metric_name: str,
) -> int:
    """Sum a custom stage metric once per unique task perf record.

    Stage perf records are attached to each output task emitted by a stage. For
    fan-out stages, the same perf record can be referenced by multiple output
    tasks, so a naive sum across output tasks can count the same stage work more
    than once.
    """
    seen_perf_ids: set[int] = set()
    total = 0.0

    for task in workflow_result.pipeline_tasks.get(pipeline_name, []):
        for perf in getattr(task, "_stage_perf", []) or []:
            if perf.stage_name != stage_name:
                continue

            perf_id = id(perf)
            if perf_id in seen_perf_ids:
                continue
            seen_perf_ids.add(perf_id)

            metric_value = perf.custom_metrics.get(metric_name)
            if metric_value is not None:
                total += metric_value

    return int(total)
