# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

from nemo_curator.backends.base import BaseStageAdapter, WorkerMetadata
from nemo_curator.backends.ray_actor_pool.raft_adapter import RayActorPoolRAFTAdapter
from nemo_curator.backends.ray_actor_pool.shuffle_adapter import ShuffleStageAdapter


def test_raft_setup_uses_constructor_worker_metadata() -> None:
    adapter = object.__new__(RayActorPoolRAFTAdapter)
    adapter.worker_metadata = WorkerMetadata(worker_id="raft-worker")
    adapter.root_unique_id = 1
    adapter._pool_size = 1
    adapter._index = 0
    adapter._name = "raft"
    adapter._raft_handle = object()
    adapter.stage = mock.MagicMock()

    with (
        mock.patch.object(adapter, "_setup_nccl"),
        mock.patch.object(adapter, "_setup_raft"),
        mock.patch.object(BaseStageAdapter, "setup") as base_setup,
    ):
        adapter.setup()

    base_setup.assert_called_once_with(adapter.worker_metadata)


def test_shuffle_setup_and_teardown_use_base_lifecycle() -> None:
    modified_class = ShuffleStageAdapter.__ray_metadata__.modified_class
    adapter = object.__new__(modified_class)
    adapter.worker_metadata = WorkerMetadata(worker_id="shuffle-worker")
    adapter.stage = mock.MagicMock()

    with (
        mock.patch.object(adapter, "setup_worker"),
        mock.patch.object(BaseStageAdapter, "setup") as base_setup,
    ):
        adapter.setup(b"root")

    base_setup.assert_called_once_with(adapter.worker_metadata)

    with mock.patch.object(BaseStageAdapter, "teardown") as base_teardown:
        adapter.teardown()

    base_teardown.assert_called_once_with()
