import time
from enum import Enum

import ray
from loguru import logger

from ray_curator.backends.base import NodeInfo, WorkerMetadata
from ray_curator.stages.base import ProcessingStage


class RayStageSpecKeys(str, Enum):
    """String enum of different flags that define keys inside ray_stage_spec."""

    IS_ACTOR_STAGE = "is_actor_stage"
    IS_FANOUT_STAGE = "is_fanout_stage"
    IS_RAFT_ACTOR = "is_raft_actor"


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
                    num_cpus=stage.resources.cpus or 1,
                    num_gpus=stage.resources.gpus or 0,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
                ).remote(stage, node_info, worker_metadata)
            )
    ray.get(ray_tasks)
