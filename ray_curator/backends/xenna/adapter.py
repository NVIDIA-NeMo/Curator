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

from cosmos_xenna.pipelines import v1 as pipelines_v1
from cosmos_xenna.ray_utils.resources import NodeInfo as XennaNodeInfo
from cosmos_xenna.ray_utils.resources import Resources as XennaResources
from cosmos_xenna.ray_utils.resources import WorkerMetadata as XennaWorkerMetadata
from loguru import logger

from nemo_curator.backends.base import BaseStageAdapter, NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import Task


class XennaStageAdapter(BaseStageAdapter, pipelines_v1.Stage):
    """Adapts ProcessingStage to Xenna.
    Args:
        stage: ProcessingStage to adapt
    """

    def __init__(
        self,
        processing_stage: ProcessingStage,
    ):
        # Initialize the base adapter with the processing stage
        super().__init__(processing_stage)
        self.processing_stage = processing_stage

    @property
    def required_resources(self) -> XennaResources:
        """Get the resources required for this stage."""
        logger.info(f"Resources: {self.processing_stage.resources}")
        return XennaResources(
            cpus=self.processing_stage.resources.cpus,
            gpus=self.processing_stage.resources.gpus,
            nvdecs=self.processing_stage.resources.nvdecs,
            nvencs=self.processing_stage.resources.nvencs,
            entire_gpu=self.processing_stage.resources.entire_gpu,
        )

    @property
    def stage_batch_size(self) -> int:
        """Get the batch size for this stage."""
        batch_size = self.processing_stage.batch_size
        return batch_size if batch_size is not None else 1

    @property
    def env_info(self) -> pipelines_v1.RuntimeEnv | None:
        """Runtime environment for this stage."""
        # Can be customized per stage if needed
        return None

    def process_data(self, tasks: list[Task]) -> list[Task] | None:
        """Process batch of tasks with automatic performance tracking.
        Args:
            tasks: List of tasks to process
        Returns:
            List of processed tasks or None
        """
        # Use the base stage's monitoring capability
        return self.process_batch(tasks)

    def setup_on_node(self, node_info: XennaNodeInfo, worker_metadata: XennaWorkerMetadata) -> None:
        """Setup the stage on a node - Xenna-specific signature.
        This method is called by Xenna with its specific types. We convert them
        to our generic types and delegate to the base adapter.
        Args:
            node_info: Xenna's NodeInfo object
            worker_metadata: Xenna's WorkerMetadata object
        """
        # Convert Xenna's types to our generic types (simplified)
        generic_node_info = NodeInfo(node_id=node_info.node_id)
        generic_worker_metadata = WorkerMetadata(
            worker_id=worker_metadata.worker_id,
            allocation=worker_metadata.allocation,  # Keep the original allocation object
        )
        super().setup_on_node(generic_node_info, generic_worker_metadata)

    def setup(self, worker_metadata: XennaWorkerMetadata) -> None:
        """Setup the stage per worker - Xenna-specific signature.
        This method is called by Xenna with its specific types. We convert them
        to our generic types and delegate to the base adapter.
        Args:
            worker_metadata: Xenna's WorkerMetadata object
        """
        # Convert Xenna's WorkerMetadata to our generic type
        generic_worker_metadata = WorkerMetadata(
            worker_id=worker_metadata.worker_id,
            allocation=worker_metadata.allocation,  # Keep the original allocation object
        )

        super().setup(generic_worker_metadata)


def create_named_xenna_stage_adapter(stage: ProcessingStage) -> XennaStageAdapter:
    """When we run a pipeline in Xenna, since we wrap using XennaStageAdapter,
    the stage name is shown as XennaStageAdapter. This is not what we want.
    So we create a dynamic subclass with the original stage's name.
    This ensures that when Xenna calls type(adapter).__name__, it returns the
    original stage's class name rather than 'XennaStageAdapter'.
    Args:
        stage (ProcessingStage): ProcessingStage to adapt

    Returns:
        XennaStageAdapter: XennaStageAdapter instance with the wrapped stage's class name
    """
    # Get the original stage's class name
    original_class_name = type(stage).__name__

    # Create a dynamic subclass with the original name
    DynamicAdapter = type(  # noqa: N806
        original_class_name,  # Use the original stage's name
        (XennaStageAdapter,),  # Inherit from XennaStageAdapter
        {
            "__module__": XennaStageAdapter.__module__,  # Keep the same module
        },
    )

    # Create and return an instance of the dynamic adapter
    return DynamicAdapter(stage)
