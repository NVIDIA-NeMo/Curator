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

"""Internal checkpoint stages injected by Pipeline._with_checkpoint_stages().

These stages are never instantiated by users directly.
"""

from typing import TYPE_CHECKING

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import Task
from nemo_curator.utils.checkpoint import _checkpoint_get

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


class _CheckpointFilterStage(ProcessingStage[Task, Task]):
    """Inserted after the first source stage.

    On each run, checks whether this source partition was already fully processed
    in a previous run.  If so, drops the task so no downstream work is repeated.
    Also defensively backfills resumability_task_key from resumability_key for
    custom source stages that only set resumability_key.
    """

    name = "_checkpoint_filter"
    resources = Resources(cpus=0.1)

    def __init__(self, checkpoint_path: str) -> None:
        self._checkpoint_path = checkpoint_path
        self._actor = None

    def setup(self, _worker_metadata: "WorkerMetadata | None" = None) -> None:
        from nemo_curator.utils.checkpoint import get_or_create_checkpoint_actor

        self._actor = get_or_create_checkpoint_actor(self._checkpoint_path)

    def process(self, task: Task) -> Task | None:
        key = task._metadata.get("resumability_key", "")
        if not key:
            msg = (
                "Source stage produced a task without _metadata['resumability_key']. "
                "Source stages must set this field to a unique, stable string "
                "(e.g. '|'.join(sorted(file_group)) + '::' + str(partition_index)). "
                "See the resumability contract in nemo_curator/utils/checkpoint.py."
            )
            raise ValueError(msg)
        task._metadata.setdefault("resumability_task_key", key)

        # Register partition with expected=1 (no-op if already present from previous run).
        _checkpoint_get(self._actor.init_partition.remote(key))

        
        if _checkpoint_get(self._actor.is_task_completed.remote(key)):
            logger.info(f"Resumability: skipping already-completed partition {key!r}")
            return None

        # A previous interrupted run may have called add_expected (fan-out) without
        # writing all completions, inflating the expected count. Reset to 1 so
        # this attempt starts with a clean slate before fan-out re-registers.
        _checkpoint_get(self._actor.reset_partition.remote(key))
        

        return task


class _CheckpointRecorderStage(ProcessingStage[Task, Task]):
    """Appended as the final pipeline stage.

    Records each leaf task as completed.  Pass-through: does not modify tasks.
    """

    name = "_checkpoint_recorder"
    resources = Resources(cpus=0.1)

    def __init__(self, checkpoint_path: str) -> None:
        self._checkpoint_path = checkpoint_path
        self._actor = None

    def setup(self, _worker_metadata: "WorkerMetadata | None" = None) -> None:
        from nemo_curator.utils.checkpoint import get_or_create_checkpoint_actor

        self._actor = get_or_create_checkpoint_actor(self._checkpoint_path)

    def process(self, task: Task) -> Task:
        key = task._metadata.get("resumability_key", "")
        task_key = task._metadata.get("resumability_task_key", "")
        if not key or not task_key:
            logger.warning(
                f"Task {task.task_id!r} reached checkpoint recorder without resumability metadata "
                "(resumability_key and/or resumability_task_key missing); completion will not be recorded."
            )
            return task
        _checkpoint_get(self._actor.mark_completed.remote(task_key, key))
        return task
