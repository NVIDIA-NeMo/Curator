# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from collections.abc import Callable, Iterable

from nemo_curator.utils import pipeline_hardware_sampler


class _RemoteMethod:
    def __init__(self, function):  # noqa: ANN001
        self._function = function

    def remote(self) -> object:
        return self._function()


class _FakeActor:
    def __init__(self, node_id: str) -> None:
        self.node_id = _RemoteMethod(lambda: f"ref:{node_id}")


class _FakeRemoteClass:
    def __init__(self) -> None:
        self.options_calls: list[dict[str, object]] = []

    def options(self, **kwargs: object) -> "_FakeRemoteClass":
        self.options_calls.append(kwargs)
        return self

    def remote(self, _interval_s: float) -> _FakeActor:
        strategy = self.options_calls[-1]["scheduling_strategy"]
        assert isinstance(strategy, dict)
        return _FakeActor(str(strategy["node_id"]))


class _FakeRay:
    def __init__(self, remote_class: _FakeRemoteClass) -> None:
        self._remote_class = remote_class

    def remote(self, **_kwargs: object) -> Callable[[type[object]], _FakeRemoteClass]:
        return lambda _actor_class: self._remote_class

    @staticmethod
    def nodes() -> list[dict[str, object]]:
        return [
            {
                "Alive": True,
                "NodeID": "hex-node-id",
                "Resources": {"CPU": 8.0, "node:10.0.0.1": 1.0},
            }
        ]

    @staticmethod
    def wait(
        refs: Iterable[object],
        *,
        num_returns: int,
        timeout: float,  # noqa: ARG004
    ) -> tuple[list[object], list[object]]:
        return list(refs)[:num_returns], []

    @staticmethod
    def get(ref: object) -> str:
        return str(ref).removeprefix("ref:")

    @staticmethod
    def kill(_actor, *, no_restart: bool) -> None:  # noqa: ANN001, ARG004
        return None


def test_sampler_placement_uses_node_id_not_node_resource_shape(monkeypatch) -> None:  # noqa: ANN001
    remote_class = _FakeRemoteClass()
    monkeypatch.setitem(sys.modules, "ray", _FakeRay(remote_class))
    monkeypatch.setattr(
        pipeline_hardware_sampler,
        "_node_affinity_strategy",
        lambda node_id: {"node_id": node_id, "soft": False},
    )

    actors = pipeline_hardware_sampler.start_pipeline_hardware_samplers()

    assert len(actors) == 1
    assert remote_class.options_calls == [{"scheduling_strategy": {"node_id": "hex-node-id", "soft": False}}]


def test_sampler_actor_is_placed_on_each_live_ray_node(shared_ray_client: None) -> None:  # noqa: ARG001
    import ray

    expected_node_ids = {str(node["NodeID"]) for node in ray.nodes() if node.get("Alive") and node.get("NodeID")}
    actors = pipeline_hardware_sampler.start_pipeline_hardware_samplers()
    try:
        actual_node_ids = set(ray.get([actor.node_id.remote() for actor in actors]))
        assert actual_node_ids == expected_node_ids
    finally:
        pipeline_hardware_sampler.stop_pipeline_hardware_samplers(actors)
