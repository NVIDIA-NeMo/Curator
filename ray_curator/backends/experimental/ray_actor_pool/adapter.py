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

from loguru import logger

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.backends.experimental.utils import get_worker_metadata_and_node_id
from nemo_curator.stages.base import ProcessingStage


class RayActorPoolStageAdapter(BaseStageAdapter):
    """Adapts ProcessingStage to Ray actors for use with ActorPool.

    This adapter is designed to work with Ray's ActorPool for better
    resource management and load balancing.
    """

    def __init__(self, stage: ProcessingStage):
        super().__init__(stage)

        # Get runtime context for worker metadata
        node_info, worker_metadata = get_worker_metadata_and_node_id()

        # Create WorkerMetadata with actor information
        self.worker_metadata = worker_metadata
        self.node_info = node_info

        # Setup the stage when the actor is created
        self.stage.setup(worker_metadata)

        self._batch_size = self.stage.batch_size
        if self._batch_size is None:
            logger.warning(f"batch size not set for stage {self.stage}. Setting it to 1.")
            self._batch_size = 1

    def get_batch_size(self) -> int:
        """Get the batch size for this stage."""
        return self._batch_size

    def setup_on_node(self) -> None:
        """Setup method for Ray actors.

        Note: This method is not used in the current implementation since we use
        the Ray Data pattern of calling setup_on_node before actor creation.
        """
        super().setup_on_node(self.node_info, self.worker_metadata)
