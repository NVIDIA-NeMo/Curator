from enum import Enum

import ray

from ray_curator.backends.base import NodeInfo, WorkerMetadata


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
    available_resources = ray.available_resources()
    if init_and_shudown:
        ray.shutdown()
    return (available_resources.get("CPU", 0), available_resources.get("GPU", 0))
