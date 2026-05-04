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

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.tasks import Task
from nemo_curator.utils.performance_utils import StageTimer

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.utils.performance_utils import StagePerfStats


_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def _stage_metric_logs_enabled() -> bool:
    return os.environ.get("CURATOR_LOG_STAGE_METRICS", "1").strip().lower() not in _FALSE_ENV_VALUES


def _metric_value(value: float | int) -> float | int:
    if isinstance(value, float):
        return round(value, 6)
    return value


@dataclass
class NodeInfo:
    """Generic node information for setup_on_node calls across backends.
    Simplified to match Xenna's structure.
    """

    node_id: str = ""


@dataclass
class WorkerMetadata:
    """Generic worker metadata for setup_on_node calls across backends.
    Simplified to match Xenna's structure. The allocation field can contain
    backend-specific allocation information.
    """

    worker_id: str = ""
    allocation: Any = None  # Backend-specific allocation info


class BaseExecutor(ABC):
    """Executor for a pipeline."""

    def __init__(self, config: dict[str, Any] | None = None, ignore_head_node: bool = False):
        self.config = config or {}
        if os.environ.get("CURATOR_IGNORE_RAY_HEAD_NODE", "").lower() in ("1", "true", "yes"):
            self.ignore_head_node = True
        else:
            self.ignore_head_node = ignore_head_node

    @abstractmethod
    def execute(self, stages: list["ProcessingStage"], initial_tasks: list[Task] | None = None) -> None:
        """Execute the pipeline."""


class BaseStageAdapter:
    """Adapts ProcessingStage to an execution backend, if needed."""

    def __init__(self, stage: "ProcessingStage"):
        self.stage = stage

    def _log_stage_invocation_metrics(self, stats: "StagePerfStats") -> None:
        """Emit one compact structured log line for every stage batch call."""
        payload: dict[str, float | int | str] = {
            "stage": stats.stage_name,
            "invocation_id": stats.invocation_id,
            "process_time_s": _metric_value(stats.process_time),
            "actor_idle_time_s": _metric_value(stats.actor_idle_time),
            "input_data_size_mb": _metric_value(stats.input_data_size_mb),
            "num_items_processed": stats.num_items_processed,
        }
        for key, value in sorted((stats.custom_metrics or {}).items()):
            payload[f"custom.{key}"] = _metric_value(value)
        logger.info("curator_stage_metrics {}", json.dumps(payload, sort_keys=True))

    def process_batch(self, tasks: list[Task]) -> list[Task]:
        """Process a batch of tasks.

        Args:
            tasks (list[Task]): List of tasks to process

        Returns:
            list[Task]: List of processed tasks
        """
        # Lazy initialize timer if needed
        if not hasattr(self, "_timer") or self._timer is None:
            self._timer = StageTimer(self.stage)

        # Calculate input data size for timer
        input_size = sum(task.num_items for task in tasks)
        # Initialize performance timer for this batch
        self._timer.reinit(input_size)

        with self._timer.time_process(input_size):
            # Use the batch processing logic
            results = self.stage.process_batch(tasks)

        # Log performance stats and add to result tasks
        _, stage_perf_stats = self._timer.log_stats()
        # Consume and attach any custom metrics recorded by the stage during this call
        custom_metrics = self.stage._consume_custom_metrics()
        if custom_metrics:
            stage_perf_stats.custom_metrics.update(custom_metrics)
        stage_perf_stats.custom_metrics.setdefault("input_tasks", float(len(tasks)))
        stage_perf_stats.custom_metrics.setdefault("output_tasks", float(len(results)))
        if _stage_metric_logs_enabled():
            self._log_stage_invocation_metrics(stage_perf_stats)
        for task in results:
            task.add_stage_perf(stage_perf_stats)

        return results

    def setup_on_node(self, node_info: NodeInfo | None = None, worker_metadata: WorkerMetadata | None = None) -> None:
        """Setup the stage on a node.

        Args:
            node_info (NodeInfo, optional): Information about the node
            worker_metadata (WorkerMetadata, optional): Information about the worker
        """
        # Call the underlying stage's setup_on_node method
        # Some backends may provide node/worker info, others may not
        self.stage.setup_on_node(node_info, worker_metadata)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        """Setup the stage once per actor.

        Args:
            worker_metadata (WorkerMetadata, optional): Information about the worker
        """
        self.stage.setup(worker_metadata)

    def teardown(self) -> None:
        """Teardown the stage once per actor."""
        self.stage.teardown()
