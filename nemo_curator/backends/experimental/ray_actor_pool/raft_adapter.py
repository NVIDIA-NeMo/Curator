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

from typing import TYPE_CHECKING

import ray
from loguru import logger

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.backends.experimental.utils import get_worker_metadata_and_node_id
from nemo_curator.stages.base import ProcessingStage

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


class RayActorPoolRAFTAdapter(BaseStageAdapter):
    """RAFT Actor adapter for Ray Actor Pool backend.

    This adapter extends RayActorPoolStageAdapter and adds RAFT capabilities
    to enable distributed processing with RAFT communication.
    """

    def __init__(
        self, stage: ProcessingStage, index: int, pool_size: int, session_id: bytes, actor_name_prefix: str = "RAFT"
    ):
        """Initialize the RAFT adapter.

        Args:
            stage: The processing stage to wrap
            index: The index of this actor in the pool
            pool_size: Total number of actors in the pool
            session_id: Unique session identifier
            actor_name_prefix: Prefix for actor names
        """
        # Initialize the base stage adapter first
        super().__init__(stage)

        # Get runtime context for worker metadata (copied from RayActorPoolStageAdapter)
        node_info, worker_metadata = get_worker_metadata_and_node_id()

        # Create WorkerMetadata with actor information
        self.worker_metadata = worker_metadata
        self.node_info = node_info

        self._batch_size = self.stage.batch_size
        if self._batch_size is None:
            logger.warning(f"batch size not set for stage {self.stage}. Setting it to 1.")
            self._batch_size = 1

        # Initialize RAFT-specific attributes
        self._index = index
        self._actor_name_prefix = actor_name_prefix
        self._name = f"{self._actor_name_prefix}Actor-{self._index}"
        self._pool_size = pool_size
        self._is_root = not index
        self.session_id = session_id

        # Initialize RAFT communication
        from nemo_curator.backends.internal.raft.ray_comms import Comms

        self.cb = Comms(verbose=True, nccl_root_location="ray-actor")
        self.cb.init()
        self.unique_id = self.cb.uniqueId
        self.root_unique_id = self.unique_id if self._is_root else None

        logger.debug(f"Initialized RAFT adapter {self._name} (index={index}, is_root={self._is_root})")

    def get_batch_size(self) -> int:
        """Get the batch size for this stage."""
        return self._batch_size

    def setup_on_node(self) -> None:
        """Setup method for Ray actors.

        Note: This method is not used in the current implementation since we use
        the Ray Data pattern of calling setup_on_node before actor creation.
        """
        super().setup_on_node(self.node_info, self.worker_metadata)

    def broadcast_root_unique_id(self) -> None:
        """Broadcast the root unique ID to all actors.

        This method should only be called by the root actor.
        """
        if self._is_root:
            actor_handles = [
                ray.get_actor(name=f"{self._actor_name_prefix}Actor-{i}", namespace=None)
                for i in range(1, self._pool_size)
            ]
            futures = [actor.set_root_unique_id.remote(self.root_unique_id) for actor in actor_handles]

            # Block until all futures complete
            ray.get(futures)
        else:
            msg = "This method should only be called by the root"
            raise RuntimeError(msg)

    def set_root_unique_id(self, root_unique_id: int) -> None:
        """Set the root unique ID.

        Parameters
        ----------
        root_unique_id : int
            The root unique ID.
        """
        logger.debug(f"{self._name}: set_root_unique_id")
        if self.root_unique_id is None:
            self.root_unique_id = root_unique_id

    def _setup_nccl(self) -> None:
        """Setup NCCL communicator."""
        from raft_dask.common.nccl import nccl

        self._nccl = nccl()
        self._nccl.init(self._pool_size, self.root_unique_id, self._index)

    def _setup_raft(self) -> None:
        """Setup RAFT."""
        from pylibraft.common.handle import Handle
        from raft_dask.common.comms_utils import inject_comms_on_handle_coll_only

        self._raft_handle = Handle(n_streams=0)
        inject_comms_on_handle_coll_only(self._raft_handle, self._nccl, self._pool_size, self._index, verbose=True)

    def setup(self, worker_metadata: "WorkerMetadata | None" = None) -> None:
        """Setup the RAFT actor.

        This method should be called after the root unique ID has been broadcast.
        """
        if self.root_unique_id is None:
            msg = "The unique ID of root is not set. Make sure `broadcast_root_unique_id` "
            "runs on the root before calling this method."
            raise RuntimeError(msg)

        try:
            self._setup_nccl()
            self._setup_raft()
            # Set the RAFT handle on the stage so it can access it
            self.stage._raft_handle = self._raft_handle
            self.stage._actor_pool_size = self._pool_size
            # This calls the stage's setup method
            super().setup(worker_metadata)
        except Exception as e:
            logger.error(f"An error occurred while setting up {self._name}: {e}.")
            raise

    def teardown(self) -> None:
        super().teardown()
