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

import time
from enum import Enum

import ray
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage


class RayStageSpecKeys(str, Enum):
    """String enum of different flags that define keys inside ray_stage_spec."""

    IS_ACTOR_STAGE = "is_actor_stage"
    IS_FANOUT_STAGE = "is_fanout_stage"
    IS_RAFT_ACTOR = "is_raft_actor"
    IS_LSH_STAGE = "is_lsh_stage"
    IS_SHUFFLE_STAGE = "is_shuffle_stage"


def get_worker_metadata_and_node_id() -> tuple[NodeInfo, WorkerMetadata]:
    """Get the worker metadata and node id from the runtime context."""
    ray_context = ray.get_runtime_context()
    return NodeInfo(node_id=ray_context.get_node_id()), WorkerMetadata(worker_id=ray_context.get_worker_id())


def get_available_cpu_gpu_resources(init_and_shudown: bool = False) -> tuple[int, int]:
    """Get available CPU and GPU resources from Ray."""
    if init_and_shudown:
        ray.init(ignore_reinit_error=True)
    time.sleep(0.2)  # ray.available_resources() returns might have a lag
    available_resources = ray.available_resources()
    if init_and_shudown:
        ray.shutdown()
    return (available_resources.get("CPU", 0), available_resources.get("GPU", 0))


@ray.remote
def _setup_stage_on_node(stage: ProcessingStage, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:
    """Ray remote function to execute setup_on_node for a stage."""
    stage.setup_on_node(node_info, worker_metadata)


def execute_setup_on_node(stages: list[ProcessingStage]) -> None:
    """Execute setup on node for a stage."""
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray_tasks = []
    for node in ray.nodes():
        node_id = node["NodeID"]
        node_info = NodeInfo(node_id=node_id)
        worker_metadata = WorkerMetadata(worker_id="", allocation=None)
        logger.info(f"Executing setup on node {node_id} for {len(stages)} stages")
        for stage in stages:
            # Create NodeInfo and WorkerMetadata for this node

            ray_tasks.append(
                _setup_stage_on_node.options(
                    num_cpus=stage.resources.cpus if stage.resources is not None else 1,
                    num_gpus=stage.resources.gpus if stage.resources is not None else 0,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
                ).remote(stage, node_info, worker_metadata)
            )
    ray.get(ray_tasks)
