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
from collections.abc import Callable
from typing import Any

from loguru import logger
from ray.data import Dataset, TaskPoolStrategy

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.backends.utils import RayStageSpecKeys, get_worker_metadata_and_node_id
from nemo_curator.stages.base import ProcessingStage

from .utils import get_actor_compute_strategy_for_stage, get_configured_actor_pool_sizing_keys, is_actor_stage

CURATOR_MANAGED_MAP_BATCHES_KWARGS = {"compute", "max_calls", "num_cpus", "num_gpus"}


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

    def _process_batch_internal(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Internal method that handles the actual batch processing logic.

        Args:
            batch: Dictionary with arrays/lists representing a batch of Task objects

        Returns:
            Dictionary with arrays/lists representing processed Task objects
        """
        tasks = batch["item"]
        results = self.process_batch(tasks)
        # Return the results as Ray Data expects them
        # For Task objects, we return them in the 'item' column
        return {"item": results}

    def _build_resource_kwargs(self, ray_stage_spec: dict) -> dict[str, float]:
        """Build num_cpus/num_gpus kwargs for map_batches.

        Checks ray_stage_spec for RAY_NUM_CPUS first so stages can request a
        different CPU reservation for Ray Data (e.g. cpus=1.0 to enable stage
        fusion) without changing resources.cpus used by other executors.
        """
        kwargs: dict[str, float] = {}
        ray_num_cpus = ray_stage_spec.get(RayStageSpecKeys.RAY_NUM_CPUS)
        if ray_num_cpus is not None:
            kwargs["num_cpus"] = ray_num_cpus  # type: ignore[reportArgumentType]
        elif self.stage.resources.cpus > 0:
            kwargs["num_cpus"] = self.stage.resources.cpus  # type: ignore[reportArgumentType]
        if self.stage.resources.gpus > 0:
            kwargs["num_gpus"] = self.stage.resources.gpus  # type: ignore[reportArgumentType]
        return kwargs

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process a Ray Data dataset through this stage.

        Args:
            dataset (Dataset): Ray Data dataset containing Task objects

        Returns:
            Dataset: Processed Ray Data dataset
        """
        ray_stage_spec = self.stage.ray_stage_spec()
        stage_is_actor = ray_stage_spec.get(RayStageSpecKeys.IS_ACTOR_STAGE, is_actor_stage(self.stage))

        if stage_is_actor:
            map_batches_fn = create_actor_from_stage(self.stage)
            map_batches_kwargs = {"compute": get_actor_compute_strategy_for_stage(self.stage)}
        else:
            map_batches_fn = create_task_from_stage(self.stage)
            map_batches_kwargs = {}

            actor_pool_sizing_keys = get_configured_actor_pool_sizing_keys(ray_stage_spec)
            if actor_pool_sizing_keys:
                logger.warning(
                    f"Ignoring ray_stage_spec worker sizing keys {actor_pool_sizing_keys} "
                    f"for Ray Data task stage {self.stage.name}; these keys only apply to actor stages."
                )

            num_workers = self.stage.num_workers()
            if num_workers is not None and num_workers > 0:
                map_batches_kwargs["compute"] = TaskPoolStrategy(size=num_workers)

            max_calls = ray_stage_spec.get(RayStageSpecKeys.MAX_CALLS_PER_WORKER)
            if max_calls is not None:
                map_batches_kwargs["max_calls"] = max_calls

        map_batches_kwargs.update(self._build_resource_kwargs(ray_stage_spec))

        # Per-stage ray_remote_args (e.g. runtime_env with different pip versions per stage).
        ray_remote_args = copy.deepcopy(ray_stage_spec.get(RayStageSpecKeys.RAY_REMOTE_ARGS) or {})
        # If the stage declares runtime_env, forward it directly to Ray so Ray creates and
        # caches an isolated virtualenv for this stage's workers.
        if self.stage.runtime_env:
            ray_remote_args["runtime_env"] = self.stage.runtime_env

        colliding_ray_remote_args = sorted(CURATOR_MANAGED_MAP_BATCHES_KWARGS & ray_remote_args.keys())
        if colliding_ray_remote_args:
            msg = (
                f"ray_remote_args for Ray Data stage {self.stage.name} must not override "
                f"Curator-managed map_batches arguments {colliding_ray_remote_args}."
            )
            raise ValueError(msg)

        map_batches_kwargs.update(ray_remote_args)

        # Let Ray Data apply the selected compute strategy and resource requirements.
        logger.info(f"{self.stage.__class__.__name__} stage_is_actor={stage_is_actor} with {map_batches_kwargs=}")

        processed_dataset = dataset.map_batches(map_batches_fn, batch_size=self.batch_size, **map_batches_kwargs)  # type: ignore[reportArgumentType]

        if ray_stage_spec.get(RayStageSpecKeys.IS_FANOUT_STAGE, False):
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
