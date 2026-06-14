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
from ray.data import Dataset

from nemo_curator.backends.base import (
    BaseStageAdapter,
    plan_upstream_task_batches,
    scheduler_ready_batch_tasks,
    stage_uses_centralized_batching,
    stage_uses_upstream_prebatching,
    upstream_prebatching_batch_size,
)
from nemo_curator.backends.utils import RayStageSpecKeys, get_worker_metadata_and_node_id
from nemo_curator.stages.base import ProcessingStage

from .utils import calculate_concurrency_for_actors_for_stage, coerce_batch_tasks, is_actor_stage


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
        tasks = coerce_batch_tasks(batch["item"])
        results = self.process_batch(tasks)
        # Return the results as Ray Data expects them
        # For Task objects, we return them in the 'item' column
        return {"item": results}

    def _process_preplanned_batch_internal(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Process one or more planner-emitted rows, each carrying ``list[Task]``."""
        planned_rows = coerce_batch_tasks(batch["item"])
        results = []
        for planned_row in planned_rows:
            task_batch = coerce_batch_tasks(planned_row)
            if not task_batch:
                continue
            results.extend(self.process_batch(task_batch))
        return {"item": results}

    def _process_scheduler_ready_batch_internal(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Process scheduler-ready rows without re-entering planning."""
        planned_rows = coerce_batch_tasks(batch["item"])
        results = []
        for planned_row in planned_rows:
            task_batch = scheduler_ready_batch_tasks(planned_row)
            if not task_batch:
                continue
            results.extend(self.process_scheduler_ready_batch(planned_row))
        return {"item": results}

    def _prebatch_dataset(self, dataset: Dataset) -> Dataset:
        """Convert task rows into planner-emitted rows of ``list[Task]``.

        Ray Data's regular ``map_batches(batch_size=N)`` has a fixed row count.
        The prebatch planner emits one object row per variable-size planned
        batch, then the actual stage map uses ``batch_size=1`` to preserve that
        plan. Planning runs over policy-sized windows to avoid a global
        repartition bottleneck for large waveform-bearing datasets.
        """
        stage = self.stage

        def prebatch_map_fn(batch: dict[str, Any]) -> dict[str, Any]:
            tasks = coerce_batch_tasks(batch["item"])
            return {"item": plan_upstream_task_batches(stage, tasks)}

        planner_batch_size = upstream_prebatching_batch_size(stage, self.batch_size)
        return dataset.map_batches(prebatch_map_fn, batch_size=planner_batch_size)

    def process_scheduler_ready_dataset(self, dataset: Dataset, ignore_head_node: bool = False) -> Dataset:
        """Process a dataset whose rows are ``SchedulerReadyTaskBatch`` objects.

        This is a compatibility/future-native scheduler hook. The default
        ``RayDataExecutor`` no longer routes centralized stages through a
        separate scheduler-ready dataset, because that forced a driver
        materialization barrier. Current executor flow keeps centralized
        planning inside the stage worker's Ray Data window.
        """
        return self._process_dataset(
            dataset,
            ignore_head_node=ignore_head_node,
            scheduler_ready_batches=True,
        )

    def process_dataset(self, dataset: Dataset, ignore_head_node: bool = False) -> Dataset:
        """Process a Ray Data dataset through this stage.

        Args:
            dataset (Dataset): Ray Data dataset containing Task objects

        Returns:
            Dataset: Processed Ray Data dataset
        """
        return self._process_dataset(dataset, ignore_head_node=ignore_head_node)

    def _process_dataset(
        self,
        dataset: Dataset,
        *,
        ignore_head_node: bool = False,
        scheduler_ready_batches: bool = False,
    ) -> Dataset:
        is_actor_stage_ = self.stage.ray_stage_spec().get(RayStageSpecKeys.IS_ACTOR_STAGE, is_actor_stage(self.stage))

        use_centralized_batching = stage_uses_centralized_batching(self.stage)
        use_preplanned_batches = (
            stage_uses_upstream_prebatching(self.stage) and not use_centralized_batching and not scheduler_ready_batches
        )
        if use_preplanned_batches:
            dataset = self._prebatch_dataset(dataset)

        map_batches_fn, concurrency_kwargs = self._map_batches_fn_and_kwargs(
            is_actor_stage=is_actor_stage_,
            ignore_head_node=ignore_head_node,
            preplanned_batches=use_preplanned_batches,
            scheduler_ready_batches=scheduler_ready_batches,
        )

        # Calculate concurrency based on available resources
        logger.info(f"{self.stage.__class__.__name__} {is_actor_stage_=} with {concurrency_kwargs=}")

        map_batch_size = self._map_batch_size(
            scheduler_ready_batches=scheduler_ready_batches,
            preplanned_batches=use_preplanned_batches,
            centralized_batches=use_centralized_batching,
        )
        processed_dataset = dataset.map_batches(map_batches_fn, batch_size=map_batch_size, **concurrency_kwargs)  # type: ignore[reportArgumentType]

        if self.stage.ray_stage_spec().get(RayStageSpecKeys.IS_FANOUT_STAGE, False):
            processed_dataset = processed_dataset.repartition(target_num_rows_per_block=1)

        return processed_dataset

    def _map_batch_size(
        self,
        *,
        scheduler_ready_batches: bool,
        preplanned_batches: bool,
        centralized_batches: bool,
    ) -> int | None:
        """Choose the Ray Data row window for this stage."""
        if scheduler_ready_batches or preplanned_batches:
            return 1
        if centralized_batches:
            return upstream_prebatching_batch_size(self.stage, self.batch_size)
        return self.batch_size

    def _map_batches_fn_and_kwargs(
        self,
        *,
        is_actor_stage: bool,
        ignore_head_node: bool,
        preplanned_batches: bool,
        scheduler_ready_batches: bool,
    ) -> tuple[Any, dict[str, Any]]:
        if is_actor_stage:
            map_batches_fn = create_actor_from_stage(
                self.stage,
                preplanned_batches=preplanned_batches,
                scheduler_ready_batches=scheduler_ready_batches,
            )
            concurrency_kwargs = {
                "concurrency": calculate_concurrency_for_actors_for_stage(
                    self.stage, ignore_head_node=ignore_head_node
                ),
            }
        else:
            map_batches_fn = create_task_from_stage(
                self.stage,
                preplanned_batches=preplanned_batches,
                scheduler_ready_batches=scheduler_ready_batches,
            )
            concurrency_kwargs = {"concurrency": None}
            max_calls = self.stage.ray_stage_spec().get(RayStageSpecKeys.MAX_CALLS_PER_WORKER, None)
            if max_calls is not None:
                concurrency_kwargs["max_calls"] = max_calls

        if self.stage.resources.cpus > 0:
            concurrency_kwargs["num_cpus"] = self.stage.resources.cpus  # type: ignore[reportArgumentType]
        if self.stage.resources.gpus > 0:
            concurrency_kwargs["num_gpus"] = self.stage.resources.gpus  # type: ignore[reportArgumentType]

        ray_remote_args = copy.deepcopy(self.stage.ray_stage_spec().get(RayStageSpecKeys.RAY_REMOTE_ARGS) or {})
        if self.stage.runtime_env:
            ray_remote_args["runtime_env"] = self.stage.runtime_env

        concurrency_kwargs.update(ray_remote_args)
        return map_batches_fn, concurrency_kwargs


def create_actor_from_stage(
    stage: ProcessingStage,
    *,
    preplanned_batches: bool = False,
    scheduler_ready_batches: bool = False,
) -> type[RayDataStageAdapter]:
    """Create a StageProcessor class with the proper stage name for display."""

    class RayDataStageActorAdapter(RayDataStageAdapter):
        """Simplified stateful processor that wraps a ProcessingStage for Ray Data."""

        def __init__(self):
            """Initialize the stage processor."""
            super().__init__(stage)
            self.setup_done = False
            requires_gpu = bool(getattr(getattr(stage, "resources", None), "requires_gpu", False))
            node_info, worker_metadata = get_worker_metadata_and_node_id(
                str(stage.name),
                requires_gpu=requires_gpu,
            )
            self.setup_on_node(node_info, worker_metadata)
            self.setup(worker_metadata)

        def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
            if scheduler_ready_batches:
                return self._process_scheduler_ready_batch_internal(batch)
            if preplanned_batches:
                return self._process_preplanned_batch_internal(batch)
            return self._process_batch_internal(batch)

    # Set the class name to match the stage name
    stage_name = stage.__class__.__name__ + "Actor"
    RayDataStageActorAdapter.__name__ = stage_name
    RayDataStageActorAdapter.__qualname__ = stage_name

    return RayDataStageActorAdapter


def create_task_from_stage(
    stage: ProcessingStage,
    *,
    preplanned_batches: bool = False,
    scheduler_ready_batches: bool = False,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
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
        if scheduler_ready_batches:
            return adapter._process_scheduler_ready_batch_internal(batch)
        if preplanned_batches:
            return adapter._process_preplanned_batch_internal(batch)
        return adapter._process_batch_internal(batch)

    # Set the function name to include the stage name with Task suffix
    stage_name = stage.__class__.__name__ + "Task"
    stage_map_fn.__name__ = stage_name
    stage_map_fn.__qualname__ = stage_name

    return stage_map_fn
