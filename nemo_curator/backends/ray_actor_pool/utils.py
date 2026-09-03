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

import math
import time
from typing import TYPE_CHECKING

import ray
from loguru import logger

from nemo_curator.backends.utils import get_available_cpu_gpu_resources

if TYPE_CHECKING:
    from ray.actor import ActorClass

    from nemo_curator.stages.base import ProcessingStage

    from .adapter import RayActorPoolStageAdapter
    from .raft_adapter import RayActorPoolRAFTAdapter

_LARGE_INT = 2**31 - 1


def get_available_actor_pool_resources(
    reserved_cpus: float = 0.0,
    reserved_gpus: float = 0.0,
    ignore_head_node: bool = False,
) -> tuple[float, float]:
    """Return currently available CPU/GPU resources after executor reservations."""
    available_cpus, available_gpus = get_available_cpu_gpu_resources(ignore_head_node=ignore_head_node)
    return (
        max(0.0, available_cpus - reserved_cpus),
        max(0.0, available_gpus - reserved_gpus),
    )


def update_resource_baseline(
    resource_baseline: tuple[float, float],
    reserved_cpus: float = 0.0,
    reserved_gpus: float = 0.0,
    ignore_head_node: bool = False,
) -> tuple[float, float]:
    """Update a CPU/GPU high-water mark from current availability."""
    current_available = get_available_actor_pool_resources(reserved_cpus, reserved_gpus, ignore_head_node)
    return (
        max(resource_baseline[0], current_available[0]),
        max(resource_baseline[1], current_available[1]),
    )


def calculate_optimal_actors_for_stage(
    stage: "ProcessingStage",
    num_tasks: int,
    reserved_cpus: float = 0.0,
    reserved_gpus: float = 0.0,
    ignore_head_node: bool = False,
) -> int:
    """Calculate optimal number of actors for a stage."""
    available_cpus, available_gpus = get_available_cpu_gpu_resources(ignore_head_node=ignore_head_node)
    # Reserve resources for system overhead
    available_cpus = max(0, available_cpus - reserved_cpus)
    available_gpus = max(0, available_gpus - reserved_gpus)

    return calculate_optimal_actors_for_resources(stage, num_tasks, (available_cpus, available_gpus))


def calculate_optimal_actors_for_stage_with_wait(  # noqa: PLR0913
    stage: "ProcessingStage",
    num_tasks: int,
    resource_baseline: tuple[float, float],
    *,
    reserved_cpus: float = 0.0,
    reserved_gpus: float = 0.0,
    ignore_head_node: bool = False,
    timeout: float = 5.0,
    interval: float = 0.2,
) -> int:
    """Wait for enough resources to create the baseline-sized actor pool."""
    if timeout < 0 or interval <= 0:
        msg = "resource_wait_timeout_s must be non-negative and resource_wait_interval_s must be positive"
        raise ValueError(msg)

    try:
        intended_num_actors = calculate_optimal_actors_for_resources(
            stage,
            num_tasks,
            resource_baseline,
        )
    except ValueError:
        intended_num_actors = 1

    required_cpus = intended_num_actors * stage.resources.cpus
    required_gpus = intended_num_actors * stage.resources.gpus
    deadline = time.monotonic() + timeout

    while True:
        available_resources = get_available_actor_pool_resources(reserved_cpus, reserved_gpus, ignore_head_node)
        if available_resources[0] >= required_cpus and available_resources[1] >= required_gpus:
            return intended_num_actors

        if time.monotonic() >= deadline:
            details = (
                f"required CPUs={required_cpus}, GPUs={required_gpus}; "
                f"available CPUs={available_resources[0]}, GPUs={available_resources[1]}"
            )
            msg = (
                f"Timed out after {timeout}s waiting for the intended {intended_num_actors}-actor pool "
                f"for {stage.name}: {details}."
            )
            raise TimeoutError(msg)

        logger.info(
            f"      Waiting for resources for {stage.name}: CPUs={available_resources[0]}/{required_cpus}, "
            f"GPUs={available_resources[1]}/{required_gpus}"
        )
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))


def calculate_optimal_actors_for_resources(
    stage: "ProcessingStage",
    num_tasks: int,
    available_resources: tuple[float, float],
) -> int:
    """Calculate an actor count from a supplied CPU/GPU availability snapshot."""
    available_cpus, available_gpus = available_resources

    # Calculate max actors based on CPU constraints
    max_actors_cpu = int(available_cpus // stage.resources.cpus) if stage.resources.cpus > 0 else _LARGE_INT

    # Calculate max actors based on GPU constraints
    max_actors_gpu = int(available_gpus // stage.resources.gpus) if stage.resources.gpus > 0 else _LARGE_INT

    # Take the minimum constraint
    max_actors_resources = min(max_actors_cpu, max_actors_gpu)

    logger.info(f"    Resource calculation: CPU limit={max_actors_cpu}, GPU limit={max_actors_gpu}")
    logger.info(f"    Available: {available_cpus} CPUs, {available_gpus} GPUs")
    logger.info(f"    Stage requirements: {stage.resources.cpus} CPUs, {stage.resources.gpus} GPUs")

    if max_actors_resources == 0:
        msg = f"No resources available for stage {stage.name}."
        raise ValueError(msg)

    num_workers = stage.num_workers()
    if num_workers is not None and num_workers > 0:
        if num_workers > max_actors_resources:
            msg = (
                f"Stage {stage.name} requires {num_workers} actors from num_workers(), "
                f"but only {max_actors_resources} fit with available resources. "
                f"Capping actor count to {max_actors_resources}."
            )
            logger.warning(msg)
            return max_actors_resources
        return num_workers

    number_of_batches = (
        math.ceil(num_tasks / stage.batch_size) if stage.batch_size is not None and stage.batch_size > 0 else num_tasks
    )
    # Don't create more actors than batches of work
    optimal_actors = min(number_of_batches, max_actors_resources)

    # Ensure at least 1 actor if we have tasks
    return max(1, optimal_actors) if num_tasks > 0 else 0


def create_named_ray_actor_pool_stage_adapter(
    stage: "ProcessingStage",
    cls: type["RayActorPoolStageAdapter"] | type["RayActorPoolRAFTAdapter"],
) -> "ActorClass[RayActorPoolStageAdapter | RayActorPoolRAFTAdapter]":
    """Create a named RayActorPoolStageAdapter or RayActorPoolRAFTAdapter.

    This function creates a dynamic subclass of the given adapter class,
    named after the stage's class name. This ensures that when Ray calls
    type(adapter).__name__, it returns the original stage's class name rather
    than 'RayActorPoolStageAdapter' or 'RayActorPoolRAFTAdapter'.

    Args:
        stage (ProcessingStage): ProcessingStage to adapt
        cls (type): The adapter class to inherit from

    Returns:
        ActorClass: A ray.remote decorated class that can be used to create actors
    """
    # Get the original stage's class name
    original_class_name = type(stage).__name__

    # Create a dynamic subclass with the original name
    DynamicAdapter = type(  # noqa: N806
        original_class_name,  # Use the original stage's name
        (cls,),  # Inherit from the adapter class
        {
            "__module__": cls.__module__,  # Keep the same module
        },
    )

    # Return the ray.remote decorated class
    return ray.remote(DynamicAdapter)
