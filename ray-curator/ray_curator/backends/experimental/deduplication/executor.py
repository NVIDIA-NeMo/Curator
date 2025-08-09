# Copyright (c) 2025, NVIDIA CORPORATION.
"""Deduplication executor that handles shuffle stages with actor pools."""

from typing import TYPE_CHECKING, Any

import ray
from loguru import logger
from ray.util.actor_pool import ActorPool

from ray_curator.backends.base import BaseExecutor
from ray_curator.backends.experimental.ray_data.utils import execute_setup_on_node
from ray_curator.backends.utils import register_loguru_serializer
from ray_curator.stages.deduplication.fuzzy.lsh.stage import LSHProcessingStage
from ray_curator.tasks import EmptyTask, Task

from .adapter import RayActorPoolStageAdapter
from .shuffle_adapter import ShuffleStageAdapter

if TYPE_CHECKING:
    from ray_curator.stages.base import ProcessingStage


_LARGE_INT = 2**31 - 1


class DeduplicationExecutor(BaseExecutor):
    """Executor that handles deduplication stages using Ray actor pools.

    This executor provides special handling for shuffle-based stages (like LSH)
    by creating actor pools and managing the shuffle workflow.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the executor.

        Args:
            config: Optional configuration dict
        """
        super().__init__(config)

    def execute(self, stages: list["ProcessingStage"], initial_tasks: list[Task] | None = None) -> list[Task]:
        """Execute the pipeline stages using ActorPool.
        Args:
            stages: List of processing stages to execute
            initial_tasks: Initial tasks to process (can be None for empty start)
        Returns:
            List of final processed tasks
        """
        if not stages:
            return []

        try:
            # Initialize Ray and register loguru serializer
            register_loguru_serializer()
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, runtime_env=self.config.get("runtime_env", None))

            # Execute setup on node for all stages BEFORE processing begins
            execute_setup_on_node(stages)
            logger.info(
                f"Setup on node complete for all stages. Starting Ray Actor Pool pipeline with {len(stages)} stages"
            )

            # Initialize with initial tasks
            current_tasks = initial_tasks or [EmptyTask]
            # Process through each stage with ActorPool
            for i, stage in enumerate(stages):
                logger.info(f"\nProcessing stage {i + 1}/{len(stages)}: {stage}")
                logger.info(f"  Input tasks: {len(current_tasks)}")

                if not current_tasks:
                    msg = f"{stage} - No tasks to process, can't continue"
                    raise ValueError(msg)  # noqa: TRY301

                # Create actor pool for this stage
                num_actors = self._calculate_optimal_actors(
                    stage,
                    len(current_tasks),
                    reserved_cpus=self.config.get("reserved_cpus", 0.0),
                    reserved_gpus=self.config.get("reserved_gpus", 0.0),
                )
                logger.info(
                    f" {stage} - Creating {num_actors} actors (CPUs: {stage.resources.cpus}, GPUs: {stage.resources.gpus})"
                )

                # Check if this is a LSHProcessingStage and create appropriate actor pool
                if isinstance(stage, LSHProcessingStage):
                    logger.info(f"  Creating RapidsMPFShuffling Actor Pool for stage: {stage.name}")
                    actors = self._create_rapidsmpf_actors(stage, num_actors, len(current_tasks))
                    current_tasks = self._process_stage_with_rapidsmpf_actors(actors, stage, current_tasks)
                else:
                    actor_pool = self._create_actor_pool(stage, num_actors)
                    # Process tasks through this stage using ActorPool
                    current_tasks = self._process_stage_with_pool(actor_pool, stage, current_tasks)

                # Clean up actor pool
                self._cleanup_actor_pool(actor_pool)

                logger.info(f"Output tasks: {len(current_tasks)}")

        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            raise
        else:
            # Return final results directly - no need for ray.get()
            final_results = current_tasks if current_tasks else []
            logger.info(f"\nPipeline completed. Final results: {len(final_results)} tasks")

            return final_results
        finally:
            # Clean up all Ray resources including named actors
            logger.info("Shutting down Ray to clean up all resources...")
            ray.shutdown()

    def _calculate_optimal_actors(
        self,
        stage: "ProcessingStage",
        num_tasks: int,
        reserved_cpus: float = 0.0,
        reserved_gpus: float = 0.0,
    ) -> int:
        """Calculate optimal number of actors for a stage."""
        # Get available resources (not total cluster resources)
        available_resources = ray.available_resources()
        available_cpus = available_resources.get("CPU", 0)
        available_gpus = available_resources.get("GPU", 0)
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

        # Don't create more actors than tasks
        optimal_actors = min(num_tasks, max_actors_resources)

        # Ensure at least 1 actor if we have tasks
        optimal_actors = max(1, optimal_actors) if num_tasks > 0 else 0

        logger.info(f"    Resource calculation: CPU limit={max_actors_cpu}, GPU limit={max_actors_gpu}")
        logger.info(f"    Available: {available_cpus} CPUs, {available_gpus} GPUs")
        logger.info(f"    Stage requirements: {stage.resources.cpus} CPUs, {stage.resources.gpus} GPUs")

        return optimal_actors

    def _create_actor_pool(self, stage: "ProcessingStage", num_actors: int) -> ActorPool:
        """Create an ActorPool for a specific stage."""
        actors = []
        for _ in range(num_actors):
            actor = RayActorPoolStageAdapter.options(
                num_cpus=stage.resources.cpus,
                num_gpus=stage.resources.gpus,
            ).remote(stage)
            actors.append(actor)

        return ActorPool(actors)

    def _create_rapidsmpf_actors(
        self, stage: "ProcessingStage", num_actors: int, num_tasks: int
    ) -> list[ray.actor.ActorHandle]:
        """Create a RapidsMPFShuffling Actors and setup UCXX communication for a specific stage."""
        logger.info(f"    Initializing RapidsMPFShuffling actor pool with {num_actors} actors")

        # Create Shuffling actors using the specialized RapidsMPFShuffling adapter
        actors = []
        for actor_idx in range(num_actors):
            actor = ShuffleStageAdapter.options(
                num_cpus=stage.resources.cpus,
                num_gpus=stage.resources.gpus,
                name=f"{stage.name}-Worker_{actor_idx}",
            ).remote(stage=stage, rank=actor_idx, nranks=num_actors, num_input_tasks=num_tasks)
            actors.append(actor)

        # Setup UCXX communication
        logger.info("    Setting up UCXX communication...")

        # initialize the first actor as the root remotely in the cluster
        root_address_bytes = ray.get(actors[0].setup_root.remote())

        # setup the workers in the cluster including root
        ray.get([actor.setup_worker.remote(root_address_bytes) for actor in actors])

        logger.info("    UCXX setup complete")

        return actors

    def _process_stage_with_rapidsmpf_actors(
        self, actors: list[ray.actor.ActorHandle], _stage: "ProcessingStage", tasks: list[Task]
    ) -> list[Task]:
        """Process Shuffle through the actors.
        Args:
            actors: The actors to use for processing
            _stage: The processing stage (for logging/context, unused)
            tasks: List of Task objects to process
        Returns:
            List of processed Task objects
        """

        actor_pool = ActorPool(actors)
        stage_batch_size: int = ray.get(actors[0].get_batch_size.remote())
        task_batches = self._generate_task_batches(tasks, stage_batch_size)

        # Step 1: Insert tasks into shuffler
        _ = list(actor_pool.map_unordered(lambda actor, batch: actor.read_and_insert.remote(batch), task_batches))

        # Step 2: Signal to all actors that insertion is complete
        _ = ray.get([actor.insert_finished.remote() for actor in actors])

        # Step 3: Extract written results
        all_results = []
        extracted_tasks = ray.get([actor.extract_and_write.remote() for actor in actors])
        for extracted_task in extracted_tasks:
            all_results.extend(extracted_task)
        return all_results

    def _generate_task_batches(self, tasks: list[Task], batch_size: int) -> list[list[Task]]:
        """Generate task batches from a list of tasks.
        Args:
            tasks: List of Task objects to process
            batch_size: The size of the batch
        Returns:
            List of task batches
        """
        task_batches = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            task_batches.append(batch)
        return task_batches

    def _process_stage_with_pool(
        self, actor_pool: ActorPool, _stage: "ProcessingStage", tasks: list[Task]
    ) -> list[Task]:
        """Process tasks through the actor pool.
        Args:
            actor_pool: The ActorPool to use for processing
            _stage: The processing stage (for logging/context, unused)
            tasks: List of Task objects to process
        Returns:
            List of processed Task objects
        """
        stage_batch_size: int = ray.get(actor_pool._idle_actors[0].get_batch_size.remote())
        task_batches = self._generate_task_batches(tasks, stage_batch_size)

        # Process each task and flatten the results since each task can produce multiple output tasks
        all_results = []
        for result_batch in actor_pool.map_unordered(
            lambda actor, batch: actor.process_batch.remote(batch), task_batches
        ):
            # result_batch is a list of tasks from processing a single input task
            all_results.extend(result_batch)
        return all_results

    def _cleanup_actor_pool(self, actor_pool: ActorPool) -> None:
        """Clean up actors in the pool."""

        # Get all actors from the pool
        all_actors = list(actor_pool._idle_actors) + [actor for actor, _ in actor_pool._future_to_actor.items()]

        for i, actor in enumerate(all_actors):
            try:
                ray.get(actor.teardown.remote())
                ray.kill(actor)
            except (ray.exceptions.RayActorError, ray.exceptions.RaySystemError) as e:  # noqa: PERF203
                logger.warning(f"      Warning: Error cleaning up actor {i}: {e}")
