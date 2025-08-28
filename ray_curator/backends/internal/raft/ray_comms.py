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

import uuid

from loguru import logger
from raft_dask.common.nccl import nccl


class Comms:
    """
    Initializes and manages underlying NCCL comms handles across the a pool of
    Ray actors. It is expected that `init()` will be called explicitly. It is
    recommended to also call `destroy()` when the comms are no longer needed so
    the underlying resources can be cleaned up. This class is not meant to be
    thread-safe.
    """

    valid_nccl_placements = "ray-actor"

    def __init__(
        self,
        verbose: bool = False,
        nccl_root_location: str = "ray-actor",
    ) -> None:
        """
        Args:
            verbose (bool): Print verbose logging. Defaults to False.
            nccl_root_location (str): Indicates where the NCCL's root node should be located.
                ['client', 'worker', 'scheduler', 'ray-actor']. Defaults to "ray-actor".
        """

        self.nccl_root_location = nccl_root_location.lower()
        if self.nccl_root_location not in Comms.valid_nccl_placements:
            msg = f"nccl_root_location must be one of: {Comms.valid_nccl_placements}"
            raise ValueError(msg)

        self.sessionId = uuid.uuid4().bytes

        self.nccl_initialized = False

        self.verbose = verbose

        if verbose:
            logger.debug("Initializing comms!")

    def __del__(self) -> None:
        if self.nccl_initialized:
            self.nccl_initialized = False

    def create_nccl_uniqueid(self) -> None:
        self.uniqueId = nccl.get_unique_id()

    def init(self) -> None:
        """
        Initializes the underlying comms.
        """
        self.create_nccl_uniqueid()

        self.nccl_initialized = True

        if self.verbose:
            logger.debug("Initialization complete.")
