# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any

import ray
from loguru import logger
from ray.data import DataContext, Dataset

from nemo_curator.backends.base import BaseExecutor
from nemo_curator.backends.experimental.utils import execute_setup_on_node
from nemo_curator.backends.utils import register_loguru_serializer
from nemo_curator.tasks import EmptyTask, Task

from .adapter import RayDataStageAdapter

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage


class RayDataExecutor(BaseExecutor):
    """Ray Data-based executor for pipeline execution.

    This executor:
    1. Executes setup on all nodes for all stages
    2. Converts initial tasks to Ray Data dataset
    3. Applies each stage as a Ray Data transformation (as a task or actor in map_batches)
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

        register_loguru_serializer()
        # This prevents verbose logging from Ray Data about serialization of the dataclass
        DataContext.get_current().enable_fallback_to_arrow_object_ext_type = True
        # Initialize with initial tasks if provided, otherwise start with EmptyTask
        tasks: list[Task] = initial_tasks if initial_tasks else [EmptyTask]
        output_tasks: list[Task] = []
        try:
            # Initialize ray and explicitly set NOSET to empty
            # This ensures if Xenna was used before which was setting NOSET, we end up overriding it.
            ray.init(
                ignore_reinit_error=True, runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": ""}}
            )

            # Convert tasks to dataset
            current_dataset = self._tasks_to_dataset(tasks)

            # Execute setup on node for all stages
            execute_setup_on_node(stages)
            logger.info(f"Setup on node complete for all stages. Starting Ray Data pipeline with {len(stages)} stages")

            # Process through each stage
            for i, stage in enumerate(stages):
                # TODO: add pipeline level config for verbosity
                logger.info(f"Processing stage {i + 1}/{len(stages)}: {stage}")
                logger.info(f"  CPU cores: {stage.resources.cpus}, GPU ratio: {stage.resources.gpus}")

                # Create adapter for this stage
                adapter = RayDataStageAdapter(stage)

                # Apply stage transformation
                current_dataset = adapter.process_dataset(current_dataset)
        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            raise
        else:
            # Convert final dataset back to tasks
            # TODO: add pipeline configuration to check if user wants to return last stages output to driver
            output_tasks = self._dataset_to_tasks(current_dataset)
            logger.info(f"Pipeline completed. Final results: {len(output_tasks)} tasks")
        finally:
            # This ensures we unset all the env vars set above during initialize and kill the pending actors.
            ray.shutdown()
        return output_tasks

    def _tasks_to_dataset(self, tasks: list[Task]) -> Dataset:
        """Convert list of tasks to Ray Data dataset.

        Args:
            tasks: List of Task objects

        Returns:
            Ray Data dataset containing Task objects directly
        """
        # Create Ray Data dataset directly from Task objects
        return ray.data.from_items(tasks, override_num_blocks=len(tasks))

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
