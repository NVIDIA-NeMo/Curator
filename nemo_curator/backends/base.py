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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nemo_curator.core.utils import ignore_ray_head_node
from nemo_curator.tasks import Task
from nemo_curator.utils.performance_utils import StageTimer

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage


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
        self.ignore_head_node = ignore_head_node or ignore_ray_head_node()

    @abstractmethod
    def execute(self, stages: list["ProcessingStage"], initial_tasks: list[Task] | None = None) -> None:
        """Execute the pipeline."""


class BaseStageAdapter:
    """Adapts ProcessingStage to an execution backend, if needed."""

    def __init__(self, stage: "ProcessingStage"):
        self.stage = stage
        self._checkpoint_actor = None

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
        checkpoint_path = getattr(self.stage, "_checkpoint_path", None)
        if checkpoint_path:
            from nemo_curator.utils.checkpoint import get_or_create_checkpoint_actor  # noqa: PLC0415

            self._checkpoint_actor = get_or_create_checkpoint_actor(checkpoint_path)

    def _propagate_resumability_metadata(self, input_tasks: list[Task], output_tasks: list[Task]) -> None:
        """Copy resumability_key from inputs to outputs that lack it."""
        if not input_tasks:
            return
        source_key = input_tasks[0]._metadata.get("resumability_key", "")
        if not source_key:
            return
        for task in output_tasks:
            if "resumability_key" not in task._metadata:
                task._metadata["resumability_key"] = source_key

    def _record_checkpoint_events(self, input_tasks: list[Task], output_tasks: list[Task]) -> None:
        """Detect fan-out and full-drop events; update checkpoint state accordingly."""
        if self._checkpoint_actor is None:
            return
        if not input_tasks:
            return

        is_fanout = len(output_tasks) > len(input_tasks) and len(input_tasks) == 1
        is_full_drop = len(output_tasks) == 0

        if is_fanout:
            from nemo_curator.utils.checkpoint import _resumability_uuid  # noqa: PLC0415

            import ray  # noqa: PLC0415

            parent = input_tasks[0]
            key = parent._metadata.get("resumability_key", "")
            if not key:
                return
            n = len(output_tasks)
            for i, task in enumerate(output_tasks):
                task._metadata["_resumability_uuid"] = _resumability_uuid(key, i + 1)
            # Commit increment synchronously so the recorder can't satisfy the check early.
            ray.get(self._checkpoint_actor.add_expected.remote(key, n - 1))

        elif is_full_drop:
            import ray  # noqa: PLC0415

            for task in input_tasks:
                key = task._metadata.get("resumability_key", "")
                uuid = task._metadata.get("_resumability_uuid", "")
                if key and uuid:
                    ray.get(self._checkpoint_actor.mark_completed.remote(uuid, key))

    def teardown(self) -> None:
        """Teardown the stage once per actor."""
        self.stage.teardown()
