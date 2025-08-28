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
from typing import TYPE_CHECKING

import ray
from loguru import logger

from nemo_curator.backends.experimental.utils import get_available_cpu_gpu_resources

if TYPE_CHECKING:
    from ray.actor import ActorClass

    from nemo_curator.stages.base import ProcessingStage

    from .adapter import RayActorPoolStageAdapter
    from .raft_adapter import RayActorPoolRAFTAdapter

_LARGE_INT = 2**31 - 1


def calculate_optimal_actors_for_stage(
    stage: "ProcessingStage",
    num_tasks: int,
    reserved_cpus: float = 0.0,
    reserved_gpus: float = 0.0,
) -> int:
    """Calculate optimal number of actors for a stage."""
    # Get available resources (not total cluster resources)
    available_cpus, available_gpus = get_available_cpu_gpu_resources()
    # Reserve resources for system overhead
    available_cpus = max(0, available_cpus - reserved_cpus)
    available_gpus = max(0, available_gpus - reserved_gpus)

    # Calculate max actors based on CPU constraints
    max_actors_cpu = int(available_cpus // stage.resources.cpus) if stage.resources.cpus > 0 else _LARGE_INT

    # Calculate max actors based on GPU constraints
    max_actors_gpu = int(available_gpus // stage.resources.gpus) if stage.resources.gpus > 0 else _LARGE_INT

    # Take the minimum constraint
    max_actors_resources = min(max_actors_cpu, max_actors_gpu)

    # Ensure we don't create more actors than configured maximum
    max_actors_resources = min(max_actors_resources, stage.num_workers() or _LARGE_INT)

    if max_actors_resources == 0:
        msg = f"No resources available for stage {stage.name}."
        raise ValueError(msg)

    number_of_batches = (
        math.ceil(num_tasks / stage.batch_size) if stage.batch_size is not None and stage.batch_size > 0 else num_tasks
    )
    # Don't create more actors than batches of work
    optimal_actors = min(number_of_batches, max_actors_resources)

    # Ensure at least 1 actor if we have tasks
    optimal_actors = max(1, optimal_actors) if num_tasks > 0 else 0

    logger.info(f"    Resource calculation: CPU limit={max_actors_cpu}, GPU limit={max_actors_gpu}")
    logger.info(f"    Available: {available_cpus} CPUs, {available_gpus} GPUs")
    logger.info(f"    Stage requirements: {stage.resources.cpus} CPUs, {stage.resources.gpus} GPUs")

    return optimal_actors


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
