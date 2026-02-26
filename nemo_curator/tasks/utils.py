# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np

from nemo_curator.pipeline.workflow import WorkflowRunResult

from .tasks import Task


class TaskPerfUtils:
    """Utilities for aggregating stage performance metrics from tasks.

    Example output format:
    {
        "StageA": {"process_time": np.array([...]), "actor_idle_time": np.array([...]), "read_time_s": np.array([...]), ...},
        "StageB": {"process_time": np.array([...]), ...}
    }
    """

    @staticmethod
    def _normalize_pipeline_tasks(
        tasks: list[Task] | WorkflowRunResult | Mapping[str, list[Task]] | None,
    ) -> dict[str, list[Task]]:
        """Return a mapping of pipeline name -> list of tasks from various input shapes."""
        if isinstance(tasks, WorkflowRunResult):
            source: Mapping[str, Any] = tasks.pipeline_tasks
        elif isinstance(tasks, Mapping):
            if "pipeline_tasks" in tasks and isinstance(tasks["pipeline_tasks"], Mapping):
                source = tasks["pipeline_tasks"]
            else:
                source = tasks
        elif isinstance(tasks, list):
            return {"": list(tasks)}
        elif tasks is None:
            return {"": []}
        else:
            msg = (
                "tasks must be a list of Task objects, a mapping of pipeline_name -> tasks, "
                "a workflow result dict, or WorkflowRunResult instance."
            )
            raise TypeError(msg)

        normalized: dict[str, list[Task]] = {}
        for pipeline_name, pipeline_tasks in source.items():
            if pipeline_tasks is None:
                normalized[str(pipeline_name)] = []
            elif isinstance(pipeline_tasks, list):
                normalized[str(pipeline_name)] = pipeline_tasks
            elif isinstance(pipeline_tasks, Task):
                normalized[str(pipeline_name)] = [pipeline_tasks]
            else:
                normalized[str(pipeline_name)] = list(pipeline_tasks)

        return normalized or {"": []}

    @staticmethod
    def collect_stage_metrics(
        tasks: list[Task] | WorkflowRunResult | Mapping[str, list[Task]] | None,
    ) -> dict[str, dict[str, np.ndarray[float]]]:
        """Collect per-stage metric lists from tasks or workflow outputs.

        The returned mapping aggregates both built-in StagePerfStats metrics and any
        custom_stats recorded by stages.

        Args:
            tasks: Iterable of tasks, a workflow result dictionary, or WorkflowRunResult.

        Returns:
            Dict mapping stage_name -> metric_name -> list of numeric values.
        """
        stage_to_metrics: dict[str, dict[str, list[float]]] = {}

        for pipeline_tasks in TaskPerfUtils._normalize_pipeline_tasks(tasks).values():
            for task in pipeline_tasks or []:
                perfs = task._stage_perf or []
                for perf in perfs:
                    stage_name = perf.stage_name

                    if stage_name not in stage_to_metrics:
                        stage_to_metrics[stage_name] = defaultdict(list)

                    metrics_dict = stage_to_metrics[stage_name]

                    # Built-in and custom metrics, flattened via perf.items()
                    for metric_name, metric_value in perf.items():
                        metrics_dict[metric_name].append(float(metric_value))

        # Convert lists to numpy arrays per metric
        return {
            stage: {m: np.asarray(vals, dtype=float) for m, vals in metrics.items()}
            for stage, metrics in stage_to_metrics.items()
        }

    @staticmethod
    def aggregate_task_metrics(
        tasks: list[Task] | WorkflowRunResult | Mapping[str, list[Task]] | None,
        prefix: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate task metrics by computing mean/std/sum."""
        metrics: dict[str, float] = {}
        pipeline_task_map = TaskPerfUtils._normalize_pipeline_tasks(tasks)
        multiple_pipelines = len(pipeline_task_map) > 1

        for pipeline_name, pipeline_tasks in pipeline_task_map.items():
            stage_metrics = TaskPerfUtils.collect_stage_metrics(pipeline_tasks)
            if prefix:
                stage_prefix = f"{prefix}_{pipeline_name}" if pipeline_name else prefix
            elif pipeline_name and multiple_pipelines:
                stage_prefix = pipeline_name
            else:
                stage_prefix = None

            for stage_name, stage_data in stage_metrics.items():
                resolved_stage_name = stage_name if stage_prefix is None else f"{stage_prefix}_{stage_name}"
                for metric_name, values in stage_data.items():
                    for agg_name, agg_func in [("sum", np.sum), ("mean", np.mean), ("std", np.std)]:
                        metric_key = f"{resolved_stage_name}_{metric_name}_{agg_name}"
                        if len(values) > 0:
                            metrics[metric_key] = float(agg_func(values))
                        else:
                            metrics[metric_key] = 0.0
        return metrics

    @staticmethod
    def get_aggregated_stage_stat(
        tasks: list[Task] | WorkflowRunResult | Mapping[str, list[Task]] | None,
        stage_prefix: str,
        stat: str,
    ) -> float:
        """Get an aggregated stat for stages matching a name prefix.

        Sums the performance statistics from all stages whose names start with the given prefix
        across all tasks.

        Args:
            tasks: A list of Task objects, a WorkflowRunResult, or a mapping of pipeline_name -> list[Task]
            stage_prefix: Match stages whose name starts with this prefix.
            stat: The stat to extract (e.g., "num_items_processed", "process_time").

        Returns:
            The aggregated stat value, or 0.0 if no matches found.
        """
        stage_metrics = TaskPerfUtils.collect_stage_metrics(tasks)

        return sum(
            float(np.sum(metrics[stat]))
            for stage_name, metrics in stage_metrics.items()
            if stage_name.startswith(stage_prefix) and stat in metrics
        )
