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

import ray
from loguru import logger
from ray.actor import ActorHandle
from ray.data import ActorPoolStrategy, Dataset, TaskPoolStrategy

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.backends.utils import (
    RayStageSpecKeys,
    get_configured_actor_pool_sizing_keys,
    get_num_workers_for_nodes,
    get_stage_num_workers_per_node,
    get_worker_metadata_and_node_id,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.utils.ray_utils import get_alive_ray_node_count

from .utils import get_actor_compute_strategy_for_stage, is_ray_data_actor_stage

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

    def __init__(
        self,
        stage: ProcessingStage,
        ignore_head_node: bool = False,
        downstream_batch_size: int | None = None,
        initial_workers: int | None = None,
        teardown_tracker: ActorHandle | None = None,
    ):
        super().__init__(stage)
        self.ignore_head_node = ignore_head_node
        # Batch size of the stage consuming this stage's output; sizes fanout output blocks.
        self.downstream_batch_size = downstream_batch_size
        # Start size of an autoscaling actor pool when the stage does not set INITIAL_WORKERS.
        self.initial_workers = initial_workers
        self.teardown_tracker = teardown_tracker

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
        # Ray Data hands over the column as a NumPy object array; stages expect a list.
        tasks = list(batch["item"])
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

    def _total_pool_size(self) -> int | None:
        """Return the cluster-wide pool size implied by num_workers_per_node()."""
        num_workers_per_node = get_stage_num_workers_per_node(self.stage)
        if num_workers_per_node is None:
            return None

        node_count = get_alive_ray_node_count(ignore_head_node=self.ignore_head_node)
        if node_count <= 0:
            msg = f"No alive Ray nodes available for num_workers_per_node sizing on stage {self.stage.name}."
            raise ValueError(msg)
        return get_num_workers_for_nodes(num_workers_per_node, node_count, self.stage.name)

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process a Ray Data dataset through this stage.

        Args:
            dataset (Dataset): Ray Data dataset containing Task objects

        Returns:
            Dataset: Processed Ray Data dataset
        """
        ray_stage_spec = self.stage.ray_stage_spec()
        stage_is_actor = is_ray_data_actor_stage(self.stage)
        total_pool_size = self._total_pool_size()

        if stage_is_actor:
            map_batches_fn = create_actor_from_stage(self.stage, self.teardown_tracker)
            compute = (
                ActorPoolStrategy(size=total_pool_size)
                if total_pool_size is not None
                else get_actor_compute_strategy_for_stage(self.stage, self.initial_workers)
            )
            map_batches_kwargs = {"compute": compute}
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
            pool_size = total_pool_size if total_pool_size is not None else num_workers
            if pool_size is not None and pool_size > 0:
                map_batches_kwargs["compute"] = TaskPoolStrategy(size=pool_size)

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

        if total_pool_size is not None:
            map_batches_kwargs["scheduling_strategy"] = "SPREAD"

        map_batches_kwargs.update(ray_remote_args)

        # Let Ray Data apply the selected compute strategy and resource requirements.
        logger.info(f"{self.stage.__class__.__name__} stage_is_actor={stage_is_actor} with {map_batches_kwargs=}")

        processed_dataset = dataset.map_batches(map_batches_fn, batch_size=self.batch_size, **map_batches_kwargs)  # type: ignore[reportArgumentType]

        if ray_stage_spec.get(RayStageSpecKeys.IS_FANOUT_STAGE, False):
            # A fanout emits all its output Tasks in one block; split it so the next stage can run
            # them in parallel. Size blocks to one downstream batch: the next map_batches bundles
            # blocks up to its batch_size anyway, so this keeps its parallelism while avoiding
            # one tiny block (and its scheduling/object-store overhead) per Task.
            rows_per_block = max(1, self.downstream_batch_size or 1)
            processed_dataset = processed_dataset.repartition(target_num_rows_per_block=rows_per_block)

        return processed_dataset


@ray.remote(num_cpus=0)
class TeardownTracker:
    """Counts actors that finished ``setup`` and ``teardown`` so the driver can wait for teardown."""

    def __init__(self):
        self.pending = 0

    def started(self) -> None:
        self.pending += 1

    def finished(self) -> None:
        self.pending -= 1

    def num_pending(self) -> int:
        return self.pending


def create_actor_from_stage(
    stage: ProcessingStage, teardown_tracker: ActorHandle | None = None
) -> type[RayDataStageAdapter]:
    """Create a StageProcessor class with the proper stage name for display.

    With ``teardown_tracker``, the actor calls ``stage.teardown()`` when Ray Data releases it
    and reports to the tracker, so the executor can wait for teardown before ``ray.shutdown()``.
    """

    class RayDataStageActorAdapter(RayDataStageAdapter):
        """Simplified stateful processor that wraps a ProcessingStage for Ray Data."""

        def __init__(self):
            """Initialize the stage processor."""
            super().__init__(stage)
            self.setup_done = False
            node_info, worker_metadata = get_worker_metadata_and_node_id()
            self.setup_on_node(node_info, worker_metadata)
            self.setup(worker_metadata)
            if teardown_tracker is not None:
                ray.get(teardown_tracker.started.remote())
                self.setup_done = True

        def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
            return self._process_batch_internal(batch)

        def __del__(self) -> None:
            # Ray Data deletes the UDF when it releases the actor; this is its only teardown hook.
            if not getattr(self, "setup_done", False):
                return
            self.setup_done = False
            try:
                self.teardown()
            except Exception:  # noqa: BLE001
                logger.exception(f"Teardown failed for stage {stage.name}")
            finally:
                ray.get(teardown_tracker.finished.remote())

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
