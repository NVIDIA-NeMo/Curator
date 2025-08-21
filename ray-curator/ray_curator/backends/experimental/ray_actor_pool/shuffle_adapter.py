# Copyright (c) 2025, NVIDIA CORPORATION.
"""Shuffle adapter for deduplication executor."""

import math
from typing import TYPE_CHECKING

import ray
from loguru import logger

from ray_curator.backends.base import BaseStageAdapter
from ray_curator.backends.experimental.utils import get_worker_metadata_and_node_id
from ray_curator.tasks import FileGroupTask

if TYPE_CHECKING:
    from ray_curator.stages.deduplication.fuzzy.lsh.stage import LSHStage


@ray.remote
class ShuffleStageAdapter(BaseStageAdapter):
    """Ray actor that wraps a shuffle stage and its actor.

    This adapter manages the lifecycle of a shuffle actor (like LSHActor)
    and provides a uniform interface for the executor.
    """

    def __init__(
        self,
        stage: "LSHStage",
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

            self.output_nparts = max(1, 2 ** math.floor(math.log2(num_input_tasks)))
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

    def setup(self) -> None:
        """Note: This method is unused"""
        err_msg = "ShuffleStageAdapter.setup is not used in the current implementation"
        raise NotImplementedError(err_msg)

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

    def read_and_insert(self, tasks: list[FileGroupTask], band_range: tuple[int, int]) -> list[FileGroupTask]:
        """Read and insert tasks into the shuffler."""
        results = []
        for task in tasks:
            results.append(self.stage.read_and_insert(task, band_range))
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
