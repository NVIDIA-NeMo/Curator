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
from typing import Any

import numpy as np

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
    def collect_stage_metrics(tasks: list[Task]) -> dict[str, dict[str, np.ndarray[float]]]:
        """Collect per-stage metric lists from a list of tasks.

        The returned mapping aggregates both built-in StagePerfStats metrics and any
        custom_stats recorded by stages.

        Args:
            tasks: Iterable of tasks, each having a `_stage_perf: list[StagePerfStats]` attribute.

        Returns:
            Dict mapping stage_name -> metric_name -> list of numeric values.
        """
        stage_to_metrics: dict[str, dict[str, list[float]]] = {}

        for task in tasks or []:
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
    def aggregate_task_metrics(tasks: list[Task], prefix: str | None = None) -> dict[str, Any]:
        """Aggregate task metrics by computing mean/std/sum."""
        metrics = {}
        tasks_metrics = TaskPerfUtils.collect_stage_metrics(tasks)
        # For each of the metric compute mean/std/sum and flatten the dict
        for stage_name, stage_data in tasks_metrics.items():
            for metric_name, values in stage_data.items():
                for agg_name, agg_func in [("sum", np.sum), ("mean", np.mean), ("std", np.std)]:
                    stage_key = stage_name if prefix is None else f"{prefix}_{stage_name}"
                    if len(values) > 0:
                        metrics[f"{stage_key}_{metric_name}_{agg_name}"] = float(agg_func(values))
                    else:
                        metrics[f"{stage_key}_{metric_name}_{agg_name}"] = 0.0
        return metrics
