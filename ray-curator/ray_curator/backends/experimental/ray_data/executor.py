"""Ray Data executor for pipeline execution."""

from typing import TYPE_CHECKING, Any

import ray
from loguru import logger
from ray.data import Dataset

from ray_curator.backends.base import BaseExecutor
from ray_curator.tasks import EmptyTask, Task

from .adapter import RayDataStageAdapter

if TYPE_CHECKING:
    from ray_curator.stages.base import ProcessingStage


class RayDataExecutor(BaseExecutor):
    """Ray Data-based executor for pipeline execution.

    This executor:
    1. Converts initial tasks to Ray Data dataset
    2. Applies each stage as a Ray Data transformation
    3. Leverages Ray Data's automatic parallelization and resource management
    4. Returns final results as a list of tasks
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        logger.warning("Ray Data executor is experimental and might not work as expected.")

    def execute(self, stages: list["ProcessingStage"], initial_tasks: list[Task] | None = None) -> list[Task]:
        """Execute the pipeline stages using Ray Data.

        Args:
            stages (list[ProcessingStage]): List of processing stages to execute
            initial_tasks (list[Task], optional): Initial tasks to process (can be None for empty start)

        Returns:
            list[Task]: List of final processed tasks
        """
        if not stages:
            return []

        logger.info(f"Starting Ray Data pipeline with {len(stages)} stages")

        # Initialize with initial tasks if provided, otherwise start with EmptyTask
        tasks: list[Task] = initial_tasks if initial_tasks else [EmptyTask]

        # Convert tasks to dataset
        current_dataset = self._tasks_to_dataset(tasks)
        logger.info(f"Initial dataset size: {current_dataset.count()}")

        try:
            # Process through each stage
            for i, stage in enumerate(stages):
                # TODO: add pipeline level config for verbosity
                logger.info(f"Processing stage {i + 1}/{len(stages)}: {stage}")
                logger.info(f"  CPU cores: {stage.resources.cpus}, GPU ratio: {stage.resources.gpus}")

                # Create adapter for this stage
                adapter = RayDataStageAdapter(stage)

                # Apply stage transformation
                current_dataset = adapter.process_dataset(current_dataset)

                # Log progress
                logger.info(f"  Stage {i + 1} completed")

        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            raise

        else:
            # Convert final dataset back to tasks
            # TODO: add pipeline configuration to check if user wants to return last stages output to driver
            logger.info(f"Converting final dataset to tasks -> {current_dataset=}")
            final_tasks = self._dataset_to_tasks(current_dataset)
            logger.info(f"Pipeline completed. Final results: {len(final_tasks)} tasks")

            return final_tasks

    def _tasks_to_dataset(self, tasks: list[Task]) -> Dataset:
        """Convert list of tasks to Ray Data dataset.

        Args:
            tasks: List of Task objects

        Returns:
            Ray Data dataset containing Task objects directly
        """
        # Create Ray Data dataset directly from Task objects
        return ray.data.from_items(tasks)

    def _dataset_to_tasks(self, dataset: Dataset) -> list[Task]:
        """Convert Ray Data dataset back to list of tasks.

        Args:
            dataset: Ray Data dataset containing Task objects

        Returns:
            List of Task objects
        """
        # Get all items from dataset
        items = dataset.take_all()

        # Handle the fact that Ray Data might return different formats
        tasks = []
        for item in items:
            tasks.append(item["item"])
        return tasks
