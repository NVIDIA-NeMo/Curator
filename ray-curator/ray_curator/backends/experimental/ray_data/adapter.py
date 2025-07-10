"""Ray Data adapter for processing stages."""

from typing import Any

import ray
from loguru import logger
from ray.data import Dataset

from ray_curator.backends.base import BaseStageAdapter
from ray_curator.backends.experimental.utils import get_worker_metadata_and_node_id
from ray_curator.stages.base import ProcessingStage


class RayDataStageAdapter(BaseStageAdapter):
    """Adapts ProcessingStage to Ray Data operations.

    This adapter converts stages to work with Ray Data datasets by:
    1. Working directly with Task objects (no dictionary conversion)
    2. Using Ray Data's map_batches for parallel processing
    3. Handling single and batch processing modes
    4. Using stateful transforms (classes) for efficient setup handling
    """

    def __init__(self, stage: ProcessingStage):
        super().__init__(stage)

        self._batch_size = self.stage.batch_size
        if self._batch_size is None and self.stage.resources.gpus > 0:
            logger.warning(f"When using Ray Data, batch size is not set for GPU stage {self.stage}. Setting it to 1.")
            self._batch_size = 1

    @property
    def batch_size(self) -> int | None:
        """Get the batch size for this stage."""
        return self._batch_size

    def _calculate_concurrency(self) -> tuple[int, int] | int:
        """Calculate concurrency based on available resources and stage requirements.

        Returns:
            int: Number of actors to use
        """
        # If explicitly set, use the specified number of workers
        if self.stage.num_workers is not None:
            return max(1, self.stage.num_workers)

        # Get available resources from Ray
        available_resources = ray.available_resources()
        # Calculate based on CPU and GPU requirements
        max_cpu_actors = float("inf")
        max_gpu_actors = float("inf")

        # CPU constraint
        if self.stage.resources.cpus > 0:
            available_cpus = available_resources.get("CPU", 0)
            max_cpu_actors = int(available_cpus / self.stage.resources.cpus)

        # GPU constraint
        if self.stage.resources.gpus > 0:
            available_gpus = available_resources.get("GPU", 0)
            max_gpu_actors = int(available_gpus / self.stage.resources.gpus)

        # Take the minimum of CPU and GPU constraints
        max_actors = min(max_cpu_actors, max_gpu_actors)

        # Ensure minimum is 1 and maximum is reasonable
        max_actors = max(1, min(int(max_actors), 8))  # Cap at 8 for safety

        return (1, max_actors)

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process a Ray Data dataset through this stage.

        Args:
            dataset (Dataset): Ray Data dataset containing Task objects

        Returns:
            Dataset: Processed Ray Data dataset
        """
        # TODO: Support nvdecs / nvencs
        if self.stage.resources.gpus <= 0 and (self.stage.resources.nvdecs > 0 or self.stage.resources.nvencs > 0):
            msg = "Ray Data does not support nvdecs / nvencs. Please use gpus instead."
            raise ValueError(msg)

        # Create a stage processor class with the proper name
        stage_processor_class = create_stage_processor_class(self.stage)

        # Calculate concurrency based on available resources
        concurrency = self._calculate_concurrency()
        logger.info(f"Concurrency for {self.stage}: {concurrency}")

        processed_dataset = dataset.map_batches(
            stage_processor_class,
            concurrency=concurrency,
            batch_size=self.batch_size,
            num_cpus=self.stage.resources.cpus,
            num_gpus=self.stage.resources.gpus,
        )

        if self.stage.ray_stage_spec.get("is_fanout_stage", False):
            processed_dataset = processed_dataset.repartition(target_num_rows_per_block=1)

        return processed_dataset


def create_stage_processor_class(stage: ProcessingStage) -> type[BaseStageAdapter]:
    """Create a StageProcessor class with the proper stage name for display."""

    class StageProcessor(BaseStageAdapter):
        """Simplified stateful processor that wraps a ProcessingStage for Ray Data."""

        def __init__(self):
            """Initialize the stage processor."""
            super().__init__(stage)
            self.setup_done = False
            node_info, worker_metadata = get_worker_metadata_and_node_id()
            self.setup_on_node(node_info, worker_metadata)
            self.setup(worker_metadata)

        def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
            """Process a batch of Task objects."""
            # Extract tasks from Ray Data batch format
            tasks = batch["item"]

            # Use the inherited process_batch method from BaseStageAdapter
            # This handles timing, performance stats, and actual processing
            results = self.process_batch(tasks)

            # Return the results in Ray Data format
            return {"item": results}

    # Set the class name to match the stage name
    StageProcessor.__name__ = stage.__class__.__name__
    StageProcessor.__qualname__ = stage.__class__.__name__

    return StageProcessor
