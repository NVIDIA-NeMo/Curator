# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import ray

from nemo_curator.utils import ray_utils
from nemo_curator.utils.ray_utils import get_head_node_id, run_on_each_node, submit_on_each_node


@contextmanager
def _reset_head_node_cache() -> Iterator[None]:
    original = ray_utils._HEAD_NODE_ID_CACHE
    ray_utils._HEAD_NODE_ID_CACHE = None
    try:
        yield
    finally:
        ray_utils._HEAD_NODE_ID_CACHE = original


@pytest.fixture
def reset_head_node_cache() -> Iterator[None]:
    with _reset_head_node_cache():
        yield


@ray.remote
def _node_id_remote() -> str:
    return ray.get_runtime_context().get_node_id()


class TestRunOnEachNode:
    def test_returns_one_result_per_alive_node(self, shared_ray_client: None) -> None:
        """Default call schedules ``remote_fn`` once per alive node and returns each landing node's id."""
        results = run_on_each_node(_node_id_remote)
        alive_ids = {n["NodeID"] for n in ray.nodes() if n.get("Alive")}
        assert set(results) == alive_ids
        assert len(results) == len(alive_ids)

    def test_ignore_head_node_skips_head(
        self,
        shared_ray_client: None,
        reset_head_node_cache: None,
    ) -> None:
        """``ignore_head_node=True`` removes the head node from the schedule set."""
        results = run_on_each_node(_node_id_remote, ignore_head_node=True)
        head_id = get_head_node_id()
        assert head_id is not None
        assert head_id not in set(results)
        expected = {n["NodeID"] for n in ray.nodes() if n.get("Alive") and n["NodeID"] != head_id}
        assert set(results) == expected

    def test_submit_returns_unresolved_refs(self, shared_ray_client: None) -> None:
        """``submit_on_each_node`` returns ObjectRefs (not values) so callers can batch awaits."""
        refs = submit_on_each_node(_node_id_remote)
        assert all(isinstance(r, ray.ObjectRef) for r in refs)
        alive_ids = {n["NodeID"] for n in ray.nodes() if n.get("Alive")}
        assert set(ray.get(refs)) == alive_ids
