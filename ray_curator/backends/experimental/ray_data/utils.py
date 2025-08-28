from ray_curator.backends.experimental.utils import get_available_cpu_gpu_resources
from ray_curator.stages.base import ProcessingStage


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
    available_cpus, available_gpus = get_available_cpu_gpu_resources(init_and_shudown=False)
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
