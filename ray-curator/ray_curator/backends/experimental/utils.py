import ray

from ray_curator.backends.base import NodeInfo, WorkerMetadata


def get_worker_metadata_and_node_id() -> tuple[NodeInfo, WorkerMetadata]:
    """Get the worker metadata and node id from the runtime context."""
    ray_context = ray.get_runtime_context()
    return NodeInfo(node_id=ray_context.get_node_id()), WorkerMetadata(worker_id=ray_context.get_worker_id())
