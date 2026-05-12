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

"""Cluster-wide Ray helpers shared across backends and inference-server code."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ray
from loguru import logger
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

if TYPE_CHECKING:
    from ray.remote_function import RemoteFunction


_HEAD_NODE_ID_CACHE: str | None = None


def is_head_node(node: dict[str, Any]) -> bool:
    """Check if a Ray node dict represents the cluster head."""
    return "node:__internal_head__" in node.get("Resources", {})


def get_head_node_id() -> str | None:
    """Return the cluster head node ID, lazily computed and cached.

    Returns ``None`` if no head node is present in the cluster.
    """
    global _HEAD_NODE_ID_CACHE  # noqa: PLW0603

    if _HEAD_NODE_ID_CACHE is not None:
        return _HEAD_NODE_ID_CACHE

    for node in ray.nodes():
        if is_head_node(node):
            _HEAD_NODE_ID_CACHE = node["NodeID"]
            return _HEAD_NODE_ID_CACHE

    return None


def submit_on_each_node(
    remote_fn: RemoteFunction,
    *args,
    ignore_head_node: bool = False,
    num_cpus: float = 0,
    num_gpus: float = 0,
) -> list[Any]:
    """Submit ``remote_fn(*args)`` once per alive Ray node and return the ObjectRefs.

    Each invocation is pinned to its node via ``NodeAffinitySchedulingStrategy(soft=False)``,
    so the function runs on (and only on) the targeted node. Dead nodes are skipped; the
    head node is also skipped when ``ignore_head_node`` is True. The caller is responsible
    for awaiting the returned refs (typically via ``ray.get``); use this when batching
    multiple fan-outs into a single await preserves parallelism.
    """
    head_node_id = get_head_node_id() if ignore_head_node else None
    refs = []
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        node_id = node["NodeID"]
        if ignore_head_node and node_id == head_node_id:
            logger.info(f"Skipping head node {node_id}")
            continue
        refs.append(
            remote_fn.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            ).remote(*args)
        )
    return refs


def run_on_each_node(
    remote_fn: RemoteFunction,
    *args,
    ignore_head_node: bool = False,
    num_cpus: float = 0,
    num_gpus: float = 0,
) -> list[Any]:
    """Submit ``remote_fn(*args)`` once per alive Ray node and return results in submission order.

    Convenience wrapper that submits via :func:`submit_on_each_node` and awaits the
    refs with a single ``ray.get``. For fan-outs across multiple submissions where
    parallelism matters, call :func:`submit_on_each_node` directly and ``ray.get``
    the combined ref list once.
    """
    return ray.get(
        submit_on_each_node(
            remote_fn,
            *args,
            ignore_head_node=ignore_head_node,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        )
    )
