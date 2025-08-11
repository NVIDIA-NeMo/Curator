import os
import uuid
from typing import TYPE_CHECKING

import ray
from loguru import logger
from ray.util.actor_pool import ActorPool

from ray_curator.backends.base import BaseExecutor
from ray_curator.backends.experimental.utils import RayStageSpecKeys, execute_setup_on_node
from ray_curator.backends.utils import register_loguru_serializer
from ray_curator.tasks import EmptyTask, Task

from .adapter import RayActorPoolStageAdapter
from .raft_adapter import RayActorPoolRAFTAdapter
from .shuffle_adapter import ShuffleStageAdapter
from .utils import calculate_optimal_actors_for_stage, create_named_ray_actor_pool_stage_adapter

if TYPE_CHECKING:
    from ray_curator.stages.base import ProcessingStage
    from ray_curator.stages.deduplication.fuzzy.lsh.stage import LSHStage

_LARGE_INT = 2**31 - 1


class RayActorPoolExecutor(BaseExecutor):
    """Ray-based executor using ActorPool for better resource management.

    This executor:
    1. Creates a pool of actors per stage using Ray's ActorPool
    2. Uses map_unordered for better load balancing and fault tolerance
    3. Lets Ray handle object ownership and garbage collection automatically
    4. Provides better backpressure management through ActorPool
    """

    def __init__(self, config: dict | None = None):
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

        session_id = uuid.uuid4().bytes

        try:
            # Initialize Ray and register loguru serializer
            register_loguru_serializer()
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

                if stage.name == "LSHStage":
                    current_tasks = self._execute_lsh_stage(stage, current_tasks)
                else:
                    # Create actor pool for this stage
                    num_actors = calculate_optimal_actors_for_stage(
                        stage,
                        len(current_tasks),
                        reserved_cpus=self.config.get("reserved_cpus", 0.0),
                        reserved_gpus=self.config.get("reserved_gpus", 0.0),
                    )
                    logger.info(
                        f" {stage} - Creating {num_actors} actors (CPUs: {stage.resources.cpus}, GPUs: {stage.resources.gpus})"
                    )

                    # Check if this is a RAFT stage and create appropriate actor pool
                    if stage.ray_stage_spec().get(RayStageSpecKeys.IS_RAFT_ACTOR, False):
                        logger.info(f"  Creating RAFT actor pool for stage: {stage.name}")
                        actor_pool = self._create_raft_actor_pool(stage, num_actors, session_id)
                    else:
                        actor_pool = self._create_actor_pool(stage, num_actors)

                    # Process tasks through this stage using ActorPool
                    current_tasks = self._process_stage_with_pool(actor_pool, stage, current_tasks)

                    # Clean up actor pool
                    self._cleanup_actor_pool(actor_pool)

                    logger.info(f"  Output tasks: {len(current_tasks)}")

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

    def _create_actor_pool(self, stage: "ProcessingStage", num_actors: int) -> ActorPool:
        """Create an ActorPool for a specific stage."""
        actors = []
        for i in range(num_actors):
            actor = (
                create_named_ray_actor_pool_stage_adapter(stage, RayActorPoolStageAdapter)
                .options(
                    num_cpus=stage.resources.cpus,
                    num_gpus=stage.resources.gpus,
                    name=f"{stage.name}-{i}",
                )
                .remote(stage)
            )
            actors.append(actor)

        return ActorPool(actors)

    def _create_raft_actor_pool(self, stage: "ProcessingStage", num_actors: int, session_id: bytes) -> ActorPool:
        """Create a RAFT ActorPool for a specific stage."""
        logger.info(f"    Initializing RAFT actor pool with {num_actors} actors")

        # Create RAFT actors using the specialized RAFT adapter
        actors = []
        for actor_idx in range(num_actors):
            actor = (
                create_named_ray_actor_pool_stage_adapter(stage, RayActorPoolRAFTAdapter)
                .options(
                    num_cpus=stage.resources.cpus,
                    num_gpus=stage.resources.gpus,
                    name=f"{stage.name}-{actor_idx}",
                )
                .remote(
                    stage=stage,
                    index=actor_idx,
                    pool_size=num_actors,
                    session_id=session_id,
                    actor_name_prefix=stage.name,
                )
            )
            actors.append(actor)

        # Setup RAFT communication
        logger.info("    Setting up RAFT communication...")

        # Get the root actor (index 0) and broadcast root unique ID
        root_actor = actors[0]
        ray.get(root_actor.broadcast_root_unique_id.remote())

        # Setup all actors (including root)
        setup_futures = [actor.setup.remote() for actor in actors]
        ray.get(setup_futures)

        logger.info("    RAFT setup complete")

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

    def _process_lsh_stage_with_rapidsmpf_actors(
        self,
        actors: list[ray.actor.ActorHandle],
        _stage: "ProcessingStage",
        tasks: list[Task],
        band_range: tuple[int, int],
    ) -> list[Task]:
        """Process Shuffle through the actors.
        Args:
            actors: The actors to use for processing
            _stage: The processing stage (for logging/context, unused)
            tasks: List of Task objects to process
            band_range: Band range for LSH shuffle
        Returns:
            List of processed Task objects
        """

        actor_pool = ActorPool(actors)
        stage_batch_size: int = ray.get(actors[0].get_batch_size.remote())
        task_batches = self._generate_task_batches(tasks, stage_batch_size)

        # Step 1: Insert tasks into shuffler
        _ = list(
            actor_pool.map_unordered(
                lambda actor, batch: actor.read_and_insert.remote(tasks=batch, band_range=band_range), task_batches
            )
        )

        # Step 2: Signal to all actors that insertion is complete
        _ = ray.get([actor.insert_finished.remote() for actor in actors])

        # Step 3: Extract written results
        all_results = []
        extracted_tasks = ray.get([actor.extract_and_write.remote() for actor in actors])
        for extracted_task in extracted_tasks:
            all_results.extend(extracted_task)
        return all_results

    def _cleanup_actors(self, actors: list[ray.actor.ActorHandle]) -> None:
        """Clean up a list of actors."""
        for i, actor in enumerate(actors):
            try:
                ray.get(actor.teardown.remote())
                ray.kill(actor)
            except (ray.exceptions.RayActorError, ray.exceptions.RaySystemError) as e:  # noqa: PERF203
                logger.warning(f"      Warning: Error cleaning up actor {i}: {e}")

    def _cleanup_actor_pool(self, actor_pool: ActorPool) -> None:
        """Clean up actors in the pool."""

        # Get all actors from the pool
        all_actors = list(actor_pool._idle_actors) + [actor for actor, _ in actor_pool._future_to_actor.items()]

        self._cleanup_actors(all_actors)

    def _execute_lsh_stage(self, stage: "LSHStage", input_tasks: list[Task]) -> list[Task]:
        """Execute an LSH stage with band iteration.

        Args:
            stage: The LSH stage to execute
            input_tasks: Input tasks to process

        Returns:
            List of output tasks from all band iterations
        """
        all_lsh_outputs = []
        original_input = input_tasks.copy()

        for band_range in stage.get_band_iterations():
            logger.info(f"  Processing band range: {band_range[0]}-{band_range[1]}")

            # Create output path for this band range
            output_path = os.path.join(stage.output_dir, f"{stage.name}", f"band_{band_range[0]}-band_{band_range[1]}")
            os.makedirs(output_path, exist_ok=True)

            # Update actor kwargs with band-specific output path
            stage.actor_kwargs["output_path"] = output_path

            # Create actors for this band range
            num_actors = calculate_optimal_actors_for_stage(
                stage,
                len(original_input),
                reserved_cpus=self.config.get("reserved_cpus", 0.0),
                reserved_gpus=self.config.get("reserved_gpus", 0.0),
            )
            logger.info(
                f" {stage} - Creating {num_actors} actors (CPUs: {stage.resources.cpus}, GPUs: {stage.resources.gpus})"
            )

            logger.info(f"  Creating RapidsMPFShuffling Actor Pool for stage: {stage.name}")
            actors = self._create_rapidsmpf_actors(stage, num_actors, len(original_input))

            # Process with original input and band range
            outputs = self._process_lsh_stage_with_rapidsmpf_actors(actors, stage, original_input, band_range)
            all_lsh_outputs.extend(outputs)

            # Clean up actors
            self._cleanup_actors(actors)

        logger.info(f"  LSH processing complete. Output tasks: {len(all_lsh_outputs)}")
        return all_lsh_outputs
