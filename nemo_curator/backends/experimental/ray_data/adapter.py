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

from collections.abc import Callable
from typing import Any

from loguru import logger
from ray.data import Dataset

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.backends.experimental.utils import RayStageSpecKeys, get_worker_metadata_and_node_id
from nemo_curator.stages.base import ProcessingStage

from .utils import calculate_concurrency_for_actors_for_stage, is_actor_stage


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

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process a Ray Data dataset through this stage.

        Args:
            dataset (Dataset): Ray Data dataset containing Task objects

        Returns:
            Dataset: Processed Ray Data dataset
        """
        # TODO: Support nvdecs / nvencs
        if self.stage.resources.gpus <= 0 and (self.stage.resources.nvdecs > 0 or self.stage.resources.nvencs > 0):
            msg = "Ray Data does not support nvdecs / nvencs. Please use gpus instead."
            raise ValueError(msg)

        is_actor_stage_ = self.stage.ray_stage_spec().get(RayStageSpecKeys.IS_ACTOR_STAGE, is_actor_stage(self.stage))

        if is_actor_stage_:
            map_batches_fn = create_actor_from_stage(self.stage)
            concurrency_kwargs = {
                "concurrency": calculate_concurrency_for_actors_for_stage(self.stage),
            }
        else:
            map_batches_fn = create_task_from_stage(self.stage)
            concurrency_kwargs = {"concurrency": None}

        if self.stage.resources.cpus > 0:
            concurrency_kwargs["num_cpus"] = self.stage.resources.cpus  # type: ignore[reportArgumentType]
        if self.stage.resources.gpus > 0:
            concurrency_kwargs["num_gpus"] = self.stage.resources.gpus  # type: ignore[reportArgumentType]

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
