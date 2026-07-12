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
from dataclasses import dataclass, field

import pytest

from nemo_curator.utils import ray_utils
from nemo_curator.utils.ray_utils import get_head_node_id, run_on_each_node, submit_on_each_node

_HEAD_NODE_ID = "a" * 56
_WORKER_NODE_ID = "b" * 56
_DEAD_NODE_ID = "c" * 56

_CLUSTER_NODES = [
    {"NodeID": _HEAD_NODE_ID, "Alive": True, "Resources": {"node:__internal_head__": 1}},
    {"NodeID": _WORKER_NODE_ID, "Alive": True, "Resources": {}},
    {"NodeID": _DEAD_NODE_ID, "Alive": False, "Resources": {}},
]


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


@pytest.fixture(autouse=True)
def mock_ray_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ray_utils.ray, "nodes", lambda: _CLUSTER_NODES)


@dataclass
class _FakeRemoteFunction:
    submissions: list[dict[str, object]] = field(default_factory=list)
    _options: dict[str, object] = field(default_factory=dict)

    def options(self, **kwargs: object) -> "_FakeRemoteFunction":
        self._options = kwargs
        return self

    def remote(self, *args: object) -> dict[str, object]:
        ref = {"args": args, "options": self._options}
        self.submissions.append(ref)
        return ref


class TestRunOnEachNode:
    def test_returns_one_result_per_alive_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default call schedules ``remote_fn`` once per alive node and returns each landing node's id."""
        remote_fn = _FakeRemoteFunction()
        monkeypatch.setattr(ray_utils.ray, "get", lambda refs: refs)

        results = run_on_each_node(remote_fn, "payload")

        assert results == remote_fn.submissions
        assert [r["args"] for r in results] == [("payload",), ("payload",)]
        scheduled_ids = [
            r["options"]["scheduling_strategy"].node_id for r in remote_fn.submissions
        ]
        assert scheduled_ids == [_HEAD_NODE_ID, _WORKER_NODE_ID]
        assert all(not r["options"]["scheduling_strategy"].soft for r in remote_fn.submissions)

    def test_ignore_head_node_skips_head(
        self,
        reset_head_node_cache: None,
    ) -> None:
        """``ignore_head_node=True`` removes the head node from the schedule set."""
        remote_fn = _FakeRemoteFunction()
        results = submit_on_each_node(remote_fn, ignore_head_node=True)

        head_id = get_head_node_id()
        assert head_id is not None
        assert results == remote_fn.submissions
        assert len(results) == 1
        strategy = results[0]["options"]["scheduling_strategy"]
        assert strategy.node_id == _WORKER_NODE_ID

    def test_submit_returns_unresolved_refs(self) -> None:
        """``submit_on_each_node`` returns ObjectRefs (not values) so callers can batch awaits."""
        remote_fn = _FakeRemoteFunction()

        refs = submit_on_each_node(remote_fn)

        assert refs == remote_fn.submissions
        assert len(refs) == 2
        assert [r["options"]["num_cpus"] for r in refs] == [0, 0]
        assert [r["options"]["num_gpus"] for r in refs] == [0, 0]
