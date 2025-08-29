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
