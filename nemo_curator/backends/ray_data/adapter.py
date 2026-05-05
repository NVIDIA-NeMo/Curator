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

import copy
import hashlib
from collections.abc import Callable
from typing import Any

import ray
from loguru import logger
from ray.data import Dataset

from nemo_curator.backends.base import BaseStageAdapter, WorkerMetadata
from nemo_curator.backends.utils import RayStageSpecKeys, get_worker_metadata_and_node_id
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import Task

from .utils import calculate_concurrency_for_actors_for_stage, is_actor_stage


def _resumability_uuid(resumability_key: str, position: int) -> str:
    return hashlib.sha256(f"{resumability_key}::{position}".encode()).hexdigest()[:16]


class RayDataStageAdapter(BaseStageAdapter):
    """Adapts ProcessingStage to Ray Data operations.

    This adapter converts stages to work with Ray Data datasets by:
    1. Working directly with Task objects (no dictionary conversion)
    2. Using Ray Data's map_batches for parallel processing
        a. If stage has both gpus and cpus specified, then we use actors
        b. If stage.setup is overridden, then we use actors
        c. Else we use tasks
    """

    def __init__(self, stage: ProcessingStage):
        super().__init__(stage)
        self._checkpoint_actor = None

        self._batch_size = self.stage.batch_size
        if self._batch_size is None and self.stage.resources.gpus > 0:
            logger.warning(f"When using Ray Data, batch size is not set for GPU stage {self.stage}. Setting it to 1.")
            self._batch_size = 1

        # Go through all the keys in the ray_stage_spec and raise error if they are not in RayStageSpecKeys
        for key in self.stage.ray_stage_spec():
            if key not in {e.value for e in RayStageSpecKeys}:
                msg = f"Invalid key {key} in ray_stage_spec for stage {self.stage}"
                raise ValueError(msg)

    @property
    def batch_size(self) -> int | None:
        """Get the batch size for this stage."""
        return self._batch_size

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        super().setup(worker_metadata)
        checkpoint_path = getattr(self.stage, "_checkpoint_path", None)
        if checkpoint_path:
            from nemo_curator.utils.checkpoint import get_or_create_checkpoint_actor

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
            parent = input_tasks[0]
            key = parent._metadata.get("resumability_key", "")
            if not key:
                return
            n = len(output_tasks)
            # Assign stable, deterministic resumability UUIDs to each fan-out output.
            for i, task in enumerate(output_tasks):
                task._metadata["_resumability_uuid"] = _resumability_uuid(key, i + 1)
            # Synchronously commit the expected increment BEFORE returning output tasks
            # so the recorder can't satisfy the completion check prematurely.
            ray.get(self._checkpoint_actor.add_expected.remote(key, n - 1))

        elif is_full_drop:
            for task in input_tasks:
                key = task._metadata.get("resumability_key", "")
                uuid = task._metadata.get("_resumability_uuid", "")
                if key and uuid:
                    ray.get(self._checkpoint_actor.mark_completed.remote(uuid, key))

    def _process_batch_internal(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Internal method that handles the actual batch processing logic.

        Args:
            batch: Dictionary with arrays/lists representing a batch of Task objects

        Returns:
            Dictionary with arrays/lists representing processed Task objects
        """
        input_tasks: list[Task] = batch["item"]
        results: list[Task] = self.process_batch(input_tasks)
        self._propagate_resumability_metadata(input_tasks, results)
        self._record_checkpoint_events(input_tasks, results)
        return {"item": results}

    def process_dataset(self, dataset: Dataset, ignore_head_node: bool = False) -> Dataset:
        """Process a Ray Data dataset through this stage.

        Args:
            dataset (Dataset): Ray Data dataset containing Task objects

        Returns:
            Dataset: Processed Ray Data dataset
        """

        is_actor_stage_ = self.stage.ray_stage_spec().get(RayStageSpecKeys.IS_ACTOR_STAGE, is_actor_stage(self.stage))

        if is_actor_stage_:
            map_batches_fn = create_actor_from_stage(self.stage)
            concurrency_kwargs = {
                "concurrency": calculate_concurrency_for_actors_for_stage(
                    self.stage, ignore_head_node=ignore_head_node
                ),
            }
        else:
            map_batches_fn = create_task_from_stage(self.stage)
            concurrency_kwargs = {"concurrency": None}
            max_calls = self.stage.ray_stage_spec().get(RayStageSpecKeys.MAX_CALLS_PER_WORKER, None)
            if max_calls is not None:
                concurrency_kwargs["max_calls"] = max_calls

        if self.stage.resources.cpus > 0:
            concurrency_kwargs["num_cpus"] = self.stage.resources.cpus  # type: ignore[reportArgumentType]
        if self.stage.resources.gpus > 0:
            concurrency_kwargs["num_gpus"] = self.stage.resources.gpus  # type: ignore[reportArgumentType]

        # Per-stage ray_remote_args (e.g. runtime_env with different pip versions per stage).
        ray_remote_args = copy.deepcopy(self.stage.ray_stage_spec().get(RayStageSpecKeys.RAY_REMOTE_ARGS) or {})
        # If the stage declares runtime_env, forward it directly to Ray so Ray creates and
        # caches an isolated virtualenv for this stage's workers.
        if self.stage.runtime_env:
            ray_remote_args["runtime_env"] = self.stage.runtime_env

        concurrency_kwargs.update(ray_remote_args)

        # Calculate concurrency based on available resources
        logger.info(f"{self.stage.__class__.__name__} {is_actor_stage_=} with {concurrency_kwargs=}")

        processed_dataset = dataset.map_batches(map_batches_fn, batch_size=self.batch_size, **concurrency_kwargs)  # type: ignore[reportArgumentType]

        if self.stage.ray_stage_spec().get(RayStageSpecKeys.IS_FANOUT_STAGE, False):
            processed_dataset = processed_dataset.repartition(target_num_rows_per_block=1)

        return processed_dataset


def create_actor_from_stage(stage: ProcessingStage) -> type[RayDataStageAdapter]:
    """Create a StageProcessor class with the proper stage name for display."""

    class RayDataStageActorAdapter(RayDataStageAdapter):
        """Simplified stateful processor that wraps a ProcessingStage for Ray Data."""

        def __init__(self):
            """Initialize the stage processor."""
            super().__init__(stage)
            self.setup_done = False
            node_info, worker_metadata = get_worker_metadata_and_node_id()
            self.setup_on_node(node_info, worker_metadata)
            self.setup(worker_metadata)

        def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
            return self._process_batch_internal(batch)

    # Set the class name to match the stage name
    stage_name = stage.__class__.__name__ + "Actor"
    RayDataStageActorAdapter.__name__ = stage_name
    RayDataStageActorAdapter.__qualname__ = stage_name

    return RayDataStageActorAdapter


def create_task_from_stage(stage: ProcessingStage) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a named Ray Data stage adapter function.

    This creates a standalone function that wraps the stage processing logic
    with a clean name that doesn't include the class qualification.

    Args:
        stage (ProcessingStage): Processing stage to adapt

    Returns:
        Callable: A function that can be used directly with Ray Data's map_batches
    """
    # Create the adapter instance
    adapter = RayDataStageAdapter(stage)

    # Create a standalone function that wraps the adapter's processing logic
    def stage_map_fn(batch: dict[str, Any]) -> dict[str, Any]:
        """Dynamically named map function that processes a batch of Task objects."""
        return adapter._process_batch_internal(batch)

    # Set the function name to include the stage name with Task suffix
    stage_name = stage.__class__.__name__ + "Task"
    stage_map_fn.__name__ = stage_name
    stage_map_fn.__qualname__ = stage_name

    return stage_map_fn
