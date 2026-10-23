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


def get_actor_compute_strategy_for_stage(stage: ProcessingStage) -> ActorPoolStrategy:
    """Get the Ray Data actor-pool compute strategy for a processing stage.

    Explicit stage ``num_workers`` requests a fixed-size actor pool. Otherwise,
    actor stages use Ray Data's autoscaling pool and can optionally override
    min/max/initial workers through ``ray_stage_spec``.
    """
    num_workers = stage.num_workers()
    if num_workers is not None and num_workers > 0:
        return ActorPoolStrategy(size=num_workers)

    ray_stage_spec = stage.ray_stage_spec()
    min_size = ray_stage_spec.get(RayStageSpecKeys.MIN_WORKERS, 1)
    max_size = ray_stage_spec.get(RayStageSpecKeys.MAX_WORKERS)
    initial_size = ray_stage_spec.get(RayStageSpecKeys.INITIAL_WORKERS)

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
