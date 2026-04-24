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
        # Lazily initialised in setup() if stage._checkpoint_path is set.
        self._write_checkpoint_mgr: Any = None

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

        # If the stage has deterministic cached output (file-writing fan-in), use it.
        cached = self.stage.get_cached_output(tasks)
        if cached is not None:
            from loguru import logger as _logger

            _logger.info(f"Stage '{self.stage.name}': skipping — cached output found ({len(cached)} tasks)")
            return cached

        # Calculate input data size for timer
        input_size = sum(task.num_items for task in tasks)
        # Initialize performance timer for this batch
        self._timer.reinit(input_size)

        with self._timer.time_process(input_size):
            # Use the batch processing logic
            results = self.stage.process_batch(tasks)

        self._propagate_source_files(tasks, results)
        self._record_checkpoint_events(tasks, results)

        # Log performance stats and add to result tasks
        _, stage_perf_stats = self._timer.log_stats()
        # Consume and attach any custom metrics recorded by the stage during this call
        custom_metrics = self.stage._consume_custom_metrics()
        if custom_metrics:
            stage_perf_stats.custom_metrics.update(custom_metrics)
        for task in results:
            task.add_stage_perf(stage_perf_stats)

        return results

    def _propagate_source_files(self, tasks: list[Task], results: list[Task]) -> None:
        """Stamp source_files onto output tasks without mixing separate source partitions.

        Rules:
        - Single input: all outputs inherit its source_files (covers fan-out safely).
        - Multi-input, same partition: propagate the shared source_files.
        - Multi-input, fan-in (results < inputs): union is intentional (dedup stages).
        - Multi-input, mixed partitions, non-fan-in: skip to avoid corrupting partition identity.
        """
        input_source_files: set[str] = set()
        for task in tasks:
            input_source_files.update(task._metadata.get("source_files", []))

        if not input_source_files or not results:
            return

        if len(tasks) == 1:
            src = set(tasks[0]._metadata.get("source_files", []))
            for result in results:
                existing = set(result._metadata.get("source_files", []))
                result._metadata["source_files"] = sorted(existing | src)
            return

        partition_keys = [
            "|".join(sorted(t._metadata.get("source_files", []))) for t in tasks if t._metadata.get("source_files")
        ]
        unique_partitions = set(partition_keys)
        if len(unique_partitions) <= 1 or len(results) < len(tasks):
            for result in results:
                existing = set(result._metadata.get("source_files", []))
                result._metadata["source_files"] = sorted(existing | input_source_files)
        else:
            from loguru import logger as _logger

            _logger.warning(
                f"Stage '{self.stage.name}': batch contains tasks from "
                f"{len(unique_partitions)} different source partitions. "
                "source_files propagation skipped to avoid mixing partition identities. "
                "Ensure the stage copies _metadata from input to output, or set batch_size=1."
            )

    def _record_checkpoint_events(self, tasks: list[Task], results: list[Task]) -> None:
        """Write fan-out increments and filtered-task shards for checkpointing."""
        if self._write_checkpoint_mgr is None:
            return

        if len(results) > len(tasks):
            # True fan-out: only attributable when there is a single input task.
            if len(tasks) == 1:
                src_list: list[str] = tasks[0]._metadata.get("source_files", [])
                if src_list:
                    self._write_checkpoint_mgr.write_expected_increment(
                        source_key="|".join(sorted(src_list)),
                        triggering_task_id=tasks[0].task_id,
                        increment=len(results) - 1,
                    )
            else:
                from loguru import logger as _logger

                _logger.warning(
                    f"Stage '{self.stage.name}': fan-out detected from a {len(tasks)}-task batch. "
                    "Checkpoint increment tracking skipped — set batch_size=1 on fan-out "
                    "stages to enable accurate resumability for those partitions."
                )

        if not results:
            # All inputs were filtered; record each so the partition completion count is reached.
            for input_task in tasks:
                src_filtered: list[str] = input_task._metadata.get("source_files", [])
                if src_filtered:
                    self._write_checkpoint_mgr.mark_filtered(input_task.task_id, src_filtered)

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
        checkpoint_path: str | None = getattr(self.stage, "_checkpoint_path", None)
        if checkpoint_path:
            storage_options: dict[str, Any] = getattr(self.stage, "_checkpoint_storage_options", {}) or {}
            try:
                import ray

                if ray.is_initialized():
                    from nemo_curator.utils.checkpoint import _CheckpointActorProxy, get_or_create_checkpoint_actor

                    self._write_checkpoint_mgr = _CheckpointActorProxy(
                        get_or_create_checkpoint_actor(checkpoint_path, storage_options)
                    )
                else:
                    from nemo_curator.utils.checkpoint import CheckpointManager

                    self._write_checkpoint_mgr = CheckpointManager(checkpoint_path, storage_options)
            except ImportError:
                from nemo_curator.utils.checkpoint import CheckpointManager

                self._write_checkpoint_mgr = CheckpointManager(checkpoint_path, storage_options)

    def teardown(self) -> None:
        """Teardown the stage once per actor."""
        self.stage.teardown()
