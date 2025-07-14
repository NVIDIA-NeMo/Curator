"""Ray Data adapter for processing stages."""

from typing import Any

from loguru import logger
from ray.data import Dataset

from ray_curator.backends.base import BaseStageAdapter
from ray_curator.stages.base import ProcessingStage

from .setup_utils import setup_stage


class RayDataStageAdapter(BaseStageAdapter):
    """Adapts ProcessingStage to Ray Data operations.

    This adapter converts stages to work with Ray Data datasets by:
    1. Working directly with Task objects (no dictionary conversion)
    2. Using Ray Data's map_batches for parallel processing
    3. Handling single and batch processing modes
    4. TODO: Supporting setup() and setup_on_node() calls like other backends
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

    def _setup_if_needed(self) -> None:
        """Setup the stage if it hasn't been setup yet.

        This method ensures setup happens exactly once per Ray Data worker.
        TODO: We need to improve this implementation to better support setup_on_node.
        """
        if not hasattr(self, "_setup_done") or not getattr(self, "_setup_done", False):
            # Mark setup as done first to avoid recursion
            self._setup_done = True

            # Use the setup utilities for coordinated setup
            setup_stage(stage=self.stage, setup_fn=self.setup, setup_on_node_fn=self.setup_on_node)

    def _process_batch_internal(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Internal method that handles the actual batch processing logic.

        Args:
            batch: Dictionary with arrays/lists representing a batch of Task objects

        Returns:
            Dictionary with arrays/lists representing processed Task objects
        """
        # Ensure setup is called before processing
        self._setup_if_needed()

        tasks = batch["item"]
        results = self.process_batch(tasks)

        # Return the results as Ray Data expects them
        # For Task objects, we return them in the 'item' column
        return {"item": results}

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

        processed_dataset = dataset.map_batches(
            create_named_ray_data_stage_adapter(self.stage).map_batch_fn,
            batch_size=self.batch_size,
            num_cpus=self.stage.resources.cpus,
            num_gpus=self.stage.resources.gpus,
        )

        if self.stage.ray_stage_spec.get("is_fanout_stage", False):
            processed_dataset = processed_dataset.repartition(target_num_rows_per_block=1)

        return processed_dataset

    def map_batch_fn(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Map function that processes a batch of Task objects.
        This gets overridden by create_named_ray_data_stage_adapter
        """
        return self._process_batch_internal(batch)


def create_named_ray_data_stage_adapter(stage: ProcessingStage) -> RayDataStageAdapter:
    """Create a named Ray Data stage adapter.

    This creates an adapter instance and assigns a dynamically named map function
    that reflects the stage name, similar to the _create_named_map_function logic.

    Args:
        stage (ProcessingStage): Processing stage to adapt

    Returns:
        RayDataStageAdapter: Ray Data stage adapter with dynamically named map function
    """
    # Create the adapter instance
    adapter = RayDataStageAdapter(stage)

    # Get the stage name for the function
    stage_name = stage.__class__.__name__

    # Create a dynamically named map function
    def stage_map_fn(batch: dict[str, Any]) -> dict[str, Any]:
        """Dynamically named map function that processes a batch of Task objects."""
        return adapter._process_batch_internal(batch)

    # Set the function name to include the stage name
    stage_map_fn.__name__ = f"{stage_name}"
    stage_map_fn.__qualname__ = f"{stage_name}"

    # Assign the dynamically named function to the adapter
    adapter.map_batch_fn = stage_map_fn

    return adapter
