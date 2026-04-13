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

"""Internal checkpoint stages injected by Pipeline.run(checkpoint_path=...).

Not part of the public API — use ``Pipeline.run(checkpoint_path=...)`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import Task

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


@dataclass
class _CheckpointFilterStage(ProcessingStage[Task, Task]):
    """Skip tasks whose source partitions are already fully complete.

    Injected automatically by :meth:`Pipeline.run` (with ``checkpoint_path``)
    after every ``is_source_stage()`` stage.  Not intended for direct use.

    A source partition is "complete" when the number of completed leaf tasks
    recorded in the checkpoint equals (or exceeds) the expected count for that
    partition.  The expected count accounts for secondary fan-outs via the
    increment-based tracking written by :class:`BaseStageAdapter`.
    """

    checkpoint_path: str
    storage_options: dict[str, Any] = field(default_factory=dict)
    name: str = "_checkpoint_filter"
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.1))

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        from nemo_curator.utils.checkpoint import CheckpointManager

        self._checkpoint_mgr = CheckpointManager(
            self.checkpoint_path, self.storage_options
        ).load()
        completed_keys = self._checkpoint_mgr.get_completed_source_keys()
        if completed_keys:
            logger.info(
                f"Checkpoint filter: {len(completed_keys)} source partition(s) already "
                "complete — those tasks will be skipped this run."
            )
        else:
            logger.info("Checkpoint filter: no previously completed partitions found — running fresh.")

    def process(self, task: Task) -> list[Task]:
        source_files: list[str] = task._metadata.get("source_files", [])
        if not source_files:
            # Can't make a checkpoint decision without source_files — pass through
            return [task]
        if self._checkpoint_mgr.is_task_completed(source_files):
            logger.debug(
                f"Checkpoint: skipping completed task {task.task_id} "
                f"(source_files={source_files[:3]}{'...' if len(source_files) > 3 else ''})"
            )
            return []
        return [task]


@dataclass
class _CheckpointRecorderStage(ProcessingStage[Task, Task]):
    """Record task completion to the checkpoint after the final pipeline stage.

    Injected automatically by :meth:`Pipeline.run` (with ``checkpoint_path``)
    as the last stage.  Not intended for direct use.

    Passes the task through unchanged; only side-effect is writing a shard file
    to ``checkpoint_path/completed/{task_id}.json``.
    """

    checkpoint_path: str
    storage_options: dict[str, Any] = field(default_factory=dict)
    name: str = "_checkpoint_recorder"
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.1))

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        from nemo_curator.utils.checkpoint import CheckpointManager

        # Write-only — no need to load existing state
        self._checkpoint_mgr = CheckpointManager(
            self.checkpoint_path, self.storage_options
        )

    def process(self, task: Task) -> Task:
        source_files: list[str] = task._metadata.get("source_files", [])
        if source_files:
            logger.debug(
                f"Checkpoint recorder: writing shard for task {task.task_id!r} "
                f"(source_files={source_files[:2]}{'...' if len(source_files) > 2 else ''})"
            )
            self._checkpoint_mgr.mark_completed(task.task_id, source_files)
        else:
            logger.warning(
                f"Checkpoint recorder: task {task.task_id!r} has no source_files in "
                "_metadata — cannot record completion. This task will be re-processed "
                "on the next resume. If this happens for every task, ensure the source "
                "stage sets _metadata['source_files'] and BaseStageAdapter propagation "
                "is running (all stages must go through XennaExecutor/RayActorPoolExecutor)."
            )
        return task
