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

from nemo_curator.backends.utils import get_available_cpu_gpu_resources
from nemo_curator.stages.base import ProcessingStage


def compute_joint_initial_allocation(
    actor_stages: list[ProcessingStage],
    available_cpus: float,
    available_gpus: float,
    stage_speeds: dict[str, float] | None = None,
    cpu_allocation_percentage: float = 0.85,
) -> dict[str, int]:
    """Compute initial replica counts for all actor stages jointly.

    If stage_speeds is provided for every actor stage (items/s per actor, from Xenna
    profiling), uses a throughput-balanced heuristic: sets N_i proportional to
    1/speed_i so all stages process items at the same rate, then scales up as far as
    the CPU and GPU budget allows.

    For CPU-only pipelines (no GPU actors), cpu_allocation_percentage reserves a
    fraction of CPUs for task-mode stages so actor reservations do not starve
    upstream/downstream task pools.

    If any actor stage is missing a speed, falls back to N=1 for all stages (safe:
    Ray Data autoscales upward from min=1 as resources free up).

    Returns:
        dict mapping stage class name -> initial replica count (>= 1)
    """
    if not actor_stages:
        return {}

    names = [s.__class__.__name__ for s in actor_stages]
    if stage_speeds is not None and all(n in stage_speeds for n in names):
        return _throughput_balanced_allocation(
            actor_stages, available_cpus, available_gpus, stage_speeds, cpu_allocation_percentage
        )
    return dict.fromkeys(names, 1)


def _throughput_balanced_allocation(
    actor_stages: list[ProcessingStage],
    available_cpus: float,
    available_gpus: float,
    stage_speeds: dict[str, float],
    cpu_allocation_percentage: float,
) -> dict[str, int]:
    """Scale all stages proportionally to speed so they run at the same rate.

    Finds the largest integer scale K such that N_i = ceil(bottleneck_speed / speed_i * K)
    fits within the CPU/GPU budget for every stage simultaneously.
    """
    names = [s.__class__.__name__ for s in actor_stages]
    speeds = [stage_speeds[n] for n in names]
    bottleneck = min(speeds)
    ratios = [bottleneck / spd for spd in speeds]

    # Always reserve cpu_allocation_percentage for task-mode stages (VideoReader, etc.).
    # Even GPU-bottlenecked pipelines have CPU-intensive task stages that need headroom.
    avail_cpus = available_cpus * cpu_allocation_percentage

    # Continuous upper bound for K (before applying ceil to individual counts)
    cpu_weight = sum(r * s.resources.cpus for r, s in zip(ratios, actor_stages, strict=True))
    gpu_weight = sum(r * s.resources.gpus for r, s in zip(ratios, actor_stages, strict=True))
    k_cpu = int(avail_cpus / cpu_weight) if cpu_weight > 0 else 1
    k_gpu = int(available_gpus / gpu_weight) if gpu_weight > 0 else k_cpu
    k_max = max(1, min(k_cpu, k_gpu))

    # Descend from k_max until the actual (ceil'd) counts fit the budget.
    # Ceils can push usage above the continuous bound, so we check iteratively.
    for k in range(k_max, 0, -1):
        counts = [max(1, math.ceil(r * k)) for r in ratios]
        cpu_used = sum(c * s.resources.cpus for c, s in zip(counts, actor_stages, strict=True))
        gpu_used = sum(c * s.resources.gpus for c, s in zip(counts, actor_stages, strict=True))
        if cpu_used <= avail_cpus and (available_gpus == 0 or gpu_used <= available_gpus):
            return dict(zip(names, counts, strict=True))

    return dict.fromkeys(names, 1)


def calculate_concurrency_for_actors_for_stage(
    stage: ProcessingStage,
    ignore_head_node: bool = False,
    initial_replicas: int | None = None,
) -> tuple[int, int] | int:
    """
    Calculate concurrency if we want to spin up actors based on available resources and stage requirements.

    Args:
        initial_replicas: If provided (from compute_joint_initial_allocation), use this as
            the minimum (initial) replica count instead of 1. This mirrors Xenna's behaviour
            of pre-allocating all workers upfront based on a joint resource budget.

    Returns:
        int | tuple[int, int]: Number of actors to use
            int: Number of workers to use
            tuple[int, int]: tuple of (initial, max) actors
    """
    # If explicitly set, use the specified number of workers
    num_workers = stage.num_workers()
    if num_workers is not None and num_workers > 0:
        return max(1, num_workers)

    # Get available resources from Ray
    available_cpus, available_gpus = get_available_cpu_gpu_resources(
        init_and_shutdown=False, ignore_head_node=ignore_head_node
    )
    # Calculate based on CPU and GPU requirements
    max_cpu_actors = float("inf")
    max_gpu_actors = float("inf")

    # CPU constraint
    if stage.resources.cpus > 0:
        max_cpu_actors = available_cpus // stage.resources.cpus

    # GPU constraint
    if stage.resources.gpus > 0:
        max_gpu_actors = available_gpus // stage.resources.gpus

    # Take the minimum of CPU and GPU constraints
    max_actors = int(min(max_cpu_actors, max_gpu_actors))
    min_actors = initial_replicas if initial_replicas is not None else 1
    return (min_actors, max_actors)


def is_actor_stage(stage: ProcessingStage) -> bool:
    """Check if the stage is an actor stage."""
    overridden_setup = type(stage).setup is not ProcessingStage.setup
    has_gpu_and_cpu = (stage.resources.gpus > 0) and (stage.resources.cpus > 0)
    return overridden_setup or has_gpu_and_cpu
