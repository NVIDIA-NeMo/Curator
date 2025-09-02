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

import math
from typing import TYPE_CHECKING

import ray
from loguru import logger

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.backends.experimental.utils import RayStageSpecKeys, get_worker_metadata_and_node_id
from nemo_curator.tasks import FileGroupTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.stages.deduplication.fuzzy.lsh.stage import LSHStage
    from nemo_curator.stages.shuffler.stage import ShuffleStage


# TODO: Remove once UCX memory usage with GPU staging buffers is fixed.
@ray.remote(runtime_env={"env_vars": {"UCX_RNDV_FRAG_MEM_TYPES": "host"}})
class ShuffleStageAdapter(BaseStageAdapter):
    """Ray actor that wraps a shuffle stage and its actor.

    This adapter manages the lifecycle of a shuffle actor (like LSHActor)
    and provides a uniform interface for the executor.
    """

    def __init__(
        self,
        stage: "ShuffleStage | LSHStage",
        rank: int,
        nranks: int,
        num_input_tasks: int | None = None,
    ):
        """Initialize the adapter.

        Args:
            stage: The shuffle stage to wrap
            rank: This actor's rank in the group
            nranks: Total number of actors in the group
            session_id: Unique session identifier
            input_nparts: Total input partitions
        """
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

        # Auto set total_nparts if not set
        stage_class_kwargs = self.stage.actor_kwargs.copy()
        if stage_class_kwargs.get("total_nparts") is None:
            if num_input_tasks is None:
                err_msg = "Shuffle total_nparts could not be set automatically. Please set it manually during stage initialization."
                raise ValueError(err_msg)
            if self.stage.ray_stage_spec().get(RayStageSpecKeys.IS_LSH_STAGE, False):
                # Rounds down to the nearest power of 2.
                # Emperically this improves shuffle performance for LSH without significantly increasing the risk of OOMs.
                self.output_nparts = max(1, 2 ** math.floor(math.log2(num_input_tasks)))
            else:
                self.output_nparts = max(1, num_input_tasks)
        else:
            self.output_nparts = stage_class_kwargs.get("total_nparts")

        stage_class_kwargs.update(
            {
                "nranks": nranks,
                "total_nparts": self.output_nparts,
            }
        )
        self.root_address = None

        self.stage._actor_obj = self.stage.actor_class(**stage_class_kwargs)

        logger.debug(f"Initialized ShuffleStageAdapter actor for rank {rank}/{nranks}")

    def get_batch_size(self) -> int:
        """Get the batch size for this stage."""
        return self._batch_size

    def setup_on_node(self) -> None:
        """
        Note: This method is not used in the current implementation since we use
        the Ray Data pattern of calling setup_on_node before actor creation.
        """
        super().setup_on_node(self.node_info, self.worker_metadata)

    def setup(self, root_address: bytes, worker_metadata: "WorkerMetadata | None" = None) -> None:
        """Setup shuffle workers and stage"""
        self.setup_worker(root_address)
        # call the stage's setup method
        super().setup(worker_metadata)

    def setup_root(self) -> None:
        """Setup the root actor."""
        _, self.root_address = self.stage._actor_obj.setup_root()
        return self.root_address

    def setup_worker(self, root_address: bytes) -> None:
        """Setup UCXX communication."""
        if self.root_address is None:
            self.root_address = root_address
        elif self.root_address != root_address:
            err_msg = f"Root address mismatch during worker setup: {self.root_address} != {root_address}"
            raise RuntimeError(err_msg)
        self.stage._actor_obj.setup_worker(self.root_address)

    def read_and_insert(
        self, tasks: list[FileGroupTask], band_range: tuple[int, int] | None = None
    ) -> list[FileGroupTask]:
        """Read and insert tasks into the shuffler."""
        insert_kwargs = {"band_range": band_range} if band_range is not None else {}
        results = []
        for task in tasks:
            results.append(self.stage.read_and_insert(task, **insert_kwargs))
        return results

    def insert_finished(self) -> None:
        """Finish the insertion phase and trigger shuffle."""
        self.stage.insert_finished()

    def extract_and_write(self) -> list[FileGroupTask]:
        """Extract shuffled data and write to output files."""
        return self.stage.extract_and_write()

    def teardown(self) -> None:
        """Clean up resources."""
        self.stage.teardown()
