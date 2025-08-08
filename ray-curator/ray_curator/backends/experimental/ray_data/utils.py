from enum import Enum

import ray
from loguru import logger

from ray_curator.backends.base import NodeInfo, WorkerMetadata
from ray_curator.stages.base import ProcessingStage


class RayStageSpecKeys(str, Enum):
    """String enum of different flags that define keys inside ray_stage_spec."""

    IS_ACTOR_STAGE = "is_actor_stage"
    IS_FANOUT_STAGE = "is_fanout_stage"


def get_available_cpu_gpu_resources() -> tuple[int, int]:
    """Get available CPU and GPU resources from Ray."""
    available_resources = ray.available_resources()
    return (available_resources.get("CPU", 0), available_resources.get("GPU", 0))


def calculate_concurrency_for_actors_for_stage(stage: ProcessingStage) -> tuple[int, int] | int:
    """
    Calculate concurrency if we want to spin up actors based on available resources and stage requirements.

    Returns:
        int | tuple[int, int]: Number of actors to use
            int: Number of workers to use
            tuple[int, int]: tuple of min / max actors to use and number of workers to use
    """
    # If explicitly set, use the specified number of workers
    num_workers = stage.num_workers()
    if num_workers is not None and num_workers > 0:
        return max(1, num_workers)

    # Get available resources from Ray
    available_cpus, available_gpus = get_available_cpu_gpu_resources()
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
    max_actors = min(max_cpu_actors, max_gpu_actors)
    return (1, int(max_actors))


def is_actor_stage(stage: ProcessingStage) -> bool:
    """Check if the stage is an actor stage."""
    overridden_setup = type(stage).setup is not ProcessingStage.setup
    has_gpu_and_cpu = (stage.resources.gpus > 0) and (stage.resources.cpus > 0)
    return overridden_setup or has_gpu_and_cpu


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
                    num_cpus=1,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
                ).remote(stage, node_info, worker_metadata)
            )
    ray.get(ray_tasks)
