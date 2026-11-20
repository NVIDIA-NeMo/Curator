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

from ray.data import ActorPoolStrategy

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage


def get_actor_compute_strategy_for_stage(
    stage: ProcessingStage, default_initial_workers: int | None = None
) -> ActorPoolStrategy:
    """Get the Ray Data actor-pool compute strategy for a processing stage.

    Explicit stage ``num_workers`` requests a fixed-size actor pool. Otherwise,
    actor stages use Ray Data's autoscaling pool and can optionally override
    min/max/initial workers through ``ray_stage_spec``. ``default_initial_workers``
    is the start size used when ``INITIAL_WORKERS`` is not set (capped at ``MAX_WORKERS``).
    """
    num_workers = stage.num_workers()
    if num_workers is not None and num_workers > 0:
        return ActorPoolStrategy(size=num_workers)

    ray_stage_spec = stage.ray_stage_spec()
    min_size = ray_stage_spec.get(RayStageSpecKeys.MIN_WORKERS, 1)
    max_size = ray_stage_spec.get(RayStageSpecKeys.MAX_WORKERS)
    initial_size = ray_stage_spec.get(RayStageSpecKeys.INITIAL_WORKERS)
    if initial_size is None and default_initial_workers is not None:
        initial_size = max(default_initial_workers, min_size)
        if max_size is not None:
            initial_size = min(initial_size, max_size)

    try:
        return ActorPoolStrategy(min_size=min_size, max_size=max_size, initial_size=initial_size)
    except ValueError as e:
        msg = f"Invalid Ray Data actor pool sizing for stage {stage.name}: {e}"
        raise ValueError(msg) from e


def is_actor_stage(stage: ProcessingStage) -> bool:
    """Check if the stage is an actor stage."""
    overridden_setup = type(stage).setup is not ProcessingStage.setup
    has_gpu_and_cpu = (stage.resources.gpus > 0) and (stage.resources.cpus > 0)
    return overridden_setup or has_gpu_and_cpu


def is_ray_data_actor_stage(stage: ProcessingStage) -> bool:
    """Whether Ray Data runs the stage in actors, honouring the ``IS_ACTOR_STAGE`` override."""
    return stage.ray_stage_spec().get(RayStageSpecKeys.IS_ACTOR_STAGE, is_actor_stage(stage))


def get_gpu_actor_initial_workers(stages: list[ProcessingStage], available_gpus: float) -> list[int | None]:
    """Start size for each stage's actor pool: GPU actor stages split the free GPUs evenly.

    Without this, Ray Data starts every autoscaling pool with one actor and adds at most one
    per scheduling decision, so GPU stages run on a single GPU for a long time. Stages that
    are not GPU actor stages, or that set ``num_workers``/``num_workers_per_node``, get ``None``.
    """
    gpu_actor_stages = [
        i for i, stage in enumerate(stages) if stage.resources.gpus > 0 and is_ray_data_actor_stage(stage)
    ]
    initial_workers: list[int | None] = [None] * len(stages)
    if not gpu_actor_stages:
        return initial_workers
    gpus_per_stage = available_gpus / len(gpu_actor_stages)
    for i in gpu_actor_stages:
        stage = stages[i]
        if stage.num_workers() is not None or stage.num_workers_per_node() is not None:
            continue
        initial_workers[i] = max(1, int(gpus_per_stage // stage.resources.gpus))
    return initial_workers
