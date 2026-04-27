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

"""Ray-placement-group construction and bundle operations.

Covers two concerns:

1. **Planning** -- turning a TP size + cluster topology into a
   ``ReplicaBundleSpec`` (single-node ``STRICT_PACK`` or multi-node
   ``STRICT_SPREAD`` with an equal per-node split).
2. **Construction + bundle-scoped operations** -- ``build_pg`` /
   ``build_replica_pg`` create detached, named PGs and wait until ready;
   ``get_bundle_node_ip`` / ``get_free_port_in_bundle`` discover
   where a bundle actually landed; ``remove_named_pgs_with_prefix``
   reaps orphans left by a prior driver session.

Subprocess lifecycle (actors, graceful stop, CUDA/env propagation)
lives in ``subprocess_mgr``. Backend-specific PGs (e.g. the Dynamo
etcd+NATS+frontend bundle) live in the backend's own subpackage.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import ray
from loguru import logger

from nemo_curator.core.serve.constants import PLACEMENT_GROUP_READY_TIMEOUT_S, WORKER_NODE_LABEL
from nemo_curator.core.utils import ignore_ray_head_node

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup


# ---------------------------------------------------------------------------
# Replica bundle-shape planner
# ---------------------------------------------------------------------------


@dataclass
class ReplicaBundleSpec:
    """Bundle shape + strategy for a single model replica.

    Attributes:
        bundles: List of resource dicts, one per bundle (e.g.
            ``[{"CPU": 1, "GPU": 4}, {"CPU": 1, "GPU": 4}]`` for TP=8 across 2 nodes).
        strategy: ``"STRICT_PACK"`` for single-bundle replicas (irrelevant but
            explicit); ``"STRICT_SPREAD"`` for multi-bundle (forces distinct nodes).
        nnodes: Number of bundles == number of distinct nodes required.
        per_node_gpus: GPUs each bundle requires (== ``tp_size // nnodes``).
        bundle_label_selector: Per-bundle node-label constraints; ``None`` when
            head-node exclusion is off.
    """

    bundles: list[dict[str, float]]
    strategy: Literal["STRICT_PACK", "STRICT_SPREAD"]
    nnodes: int
    per_node_gpus: int
    bundle_label_selector: list[dict[str, str]] | None = None

    @property
    def is_multi_node(self) -> bool:
        return self.nnodes > 1

    @property
    def total_gpus(self) -> int:
        return self.nnodes * self.per_node_gpus


def _get_gpu_topology(
    head_node_id: str | None = None,
    nodes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return per-node GPU topology: ``[{"node_id", "num_gpus", "is_head"}, ...]``.

    Uses total node resources, not current availability -- topology shape is a
    static property of the cluster. Ray's PG scheduler handles dynamic capacity.

    Args:
        head_node_id: Ray node ID to tag as head in output (for
            ``CURATOR_IGNORE_RAY_HEAD_NODE`` filtering). Defaults to the
            node bearing the ``node:__internal_head__`` resource marker;
            falls back to the driver's own node id if no marker is found
            (matches the behaviour used by ``backends/utils.py``).
        nodes: Pre-fetched ``ray.nodes()`` to avoid a redundant call.
    """
    if head_node_id is None:
        from nemo_curator.backends.utils import get_head_node_id

        head_node_id = get_head_node_id() or ray.get_runtime_context().get_node_id()

    topology: list[dict[str, Any]] = []
    for node in nodes or ray.nodes():
        if not node.get("Alive", False):
            continue
        resources = node.get("Resources", {})
        num_gpus = int(resources.get("GPU", 0))
        if num_gpus == 0:
            continue
        node_id = node["NodeID"]
        topology.append({"node_id": node_id, "num_gpus": num_gpus, "is_head": node_id == head_node_id})
    return topology


def plan_replica_bundle_shape(
    tp_size: int,
    *,
    head_node_id: str | None = None,
    _topology: list[dict[str, Any]] | None = None,
    _nodes: list[dict[str, Any]] | None = None,
) -> ReplicaBundleSpec:
    """Pick the bundle shape for one replica given current cluster topology.

    Single-node: if any node has ``>= tp_size`` GPUs, return one bundle of
    size ``tp_size`` with ``STRICT_PACK``.

    Multi-node: find the smallest ``nnodes`` such that
    ``tp_size % nnodes == 0`` and at least ``nnodes`` nodes have
    ``>= tp_size / nnodes`` GPUs each. Return ``nnodes`` equal bundles with
    ``STRICT_SPREAD``. vLLM requires an even per-node split (1+3 for TP=4
    fails with a CUDA device ordinal error), so asymmetric splits are never
    considered.

    When ``CURATOR_IGNORE_RAY_HEAD_NODE`` is set, the head node is filtered
    out of topology and every bundle gets
    ``[{"ray.io/node-type": "worker"}]`` as a label selector.
    """
    if tp_size < 1:
        msg = f"tp_size must be >= 1, got {tp_size}"
        raise ValueError(msg)

    topology = _topology if _topology is not None else _get_gpu_topology(head_node_id, nodes=_nodes)
    if not topology:
        msg = "No GPU nodes found in the Ray cluster."
        raise RuntimeError(msg)

    ignore_head = ignore_ray_head_node()
    if ignore_head:
        topology = [n for n in topology if not n["is_head"]]
        if not topology:
            msg = "CURATOR_IGNORE_RAY_HEAD_NODE is set but no non-head GPU nodes are available."
            raise RuntimeError(msg)

    label_selector_one = [WORKER_NODE_LABEL] if ignore_head else None

    # Single-node preference: cheapest placement, no cross-node NCCL.
    max_per_node = max(n["num_gpus"] for n in topology)
    if max_per_node >= tp_size:
        return ReplicaBundleSpec(
            bundles=[{"CPU": 1, "GPU": tp_size}],
            strategy="STRICT_PACK",
            nnodes=1,
            per_node_gpus=tp_size,
            bundle_label_selector=label_selector_one,
        )

    # Multi-node even split: smallest nnodes dividing tp_size with sufficient
    # per-node topology. vLLM's distributed executor requires identical
    # local_world_size per node.
    for nnodes in range(2, len(topology) + 1):
        if tp_size % nnodes != 0:
            continue
        per_node = tp_size // nnodes
        eligible = sum(1 for n in topology if n["num_gpus"] >= per_node)
        if eligible >= nnodes:
            selector = [WORKER_NODE_LABEL] * nnodes if ignore_head else None
            return ReplicaBundleSpec(
                bundles=[{"CPU": 1, "GPU": per_node}] * nnodes,
                strategy="STRICT_SPREAD",
                nnodes=nnodes,
                per_node_gpus=per_node,
                bundle_label_selector=selector,
            )

    msg = (
        f"Cannot place TP={tp_size}: need an even split across nodes but no "
        f"valid combination found across {len(topology)} eligible node(s)."
    )
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Placement group construction
# ---------------------------------------------------------------------------


def build_pg(
    bundles: list[dict[str, float]],
    strategy: str,
    *,
    name: str,
    bundle_label_selector: list[dict[str, str]] | None,
    ready_timeout_s: float,
) -> PlacementGroup:
    """Create a detached, named PG and wait until ready; clean up on failure."""
    from ray.util.placement_group import placement_group

    pg_kwargs: dict[str, Any] = {
        "bundles": bundles,
        "strategy": strategy,
        "name": name,
        "lifetime": "detached",
    }
    if bundle_label_selector is not None:
        pg_kwargs["bundle_label_selector"] = bundle_label_selector

    pg = placement_group(**pg_kwargs)
    try:
        ray.get(pg.ready(), timeout=ready_timeout_s)
    except Exception:
        with contextlib.suppress(Exception):
            ray.util.remove_placement_group(pg)
        raise
    return pg


def build_replica_pg(
    spec: ReplicaBundleSpec,
    *,
    name: str,
    ready_timeout_s: float = PLACEMENT_GROUP_READY_TIMEOUT_S,
) -> PlacementGroup:
    """Create a detached, named PG for one replica and wait until ready.

    PG is created with ``lifetime="detached"`` so it survives driver
    disconnects between ``server.start()``, ``pipeline.run()``, and
    ``server.stop()``. The caller-supplied ``name`` is used for orphan
    cleanup via ``remove_named_pgs_with_prefix``.
    """
    return build_pg(
        spec.bundles,
        spec.strategy,
        name=name,
        bundle_label_selector=spec.bundle_label_selector,
        ready_timeout_s=ready_timeout_s,
    )


# ---------------------------------------------------------------------------
# Remote discovery (port + IP) via PG bundles
# ---------------------------------------------------------------------------


def _run_in_bundle(pg: PlacementGroup, bundle_index: int, remote_fn: Any, *args: Any) -> Any:  # noqa: ANN401
    """Schedule *remote_fn* into ``pg``'s bundle *bundle_index* and return the result."""
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    return ray.get(
        remote_fn.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=bundle_index
            ),
        ).remote(*args),
    )


@ray.remote(num_cpus=0)
def _remote_get_free_port(start: int, get_next: bool) -> int:
    from nemo_curator.core.utils import get_free_port as _local_get_free_port

    return _local_get_free_port(start, get_next)


@ray.remote(num_cpus=0)
def _remote_get_node_ip() -> str:
    return ray.util.get_node_ip_address()


def get_free_port_in_bundle(
    pg: PlacementGroup, bundle_index: int, start_port: int, get_next_free_port: bool = True
) -> int:
    """Find a free port on the node hosting ``pg``'s bundle *bundle_index*.

    The remote task is scheduled into the target bundle via
    ``PlacementGroupSchedulingStrategy``, so port availability is checked on
    the same node where the consuming actor will bind.
    """
    return _run_in_bundle(pg, bundle_index, _remote_get_free_port, start_port, get_next_free_port)


def get_bundle_node_ip(pg: PlacementGroup, bundle_index: int) -> str:
    """Return the routable IP of the node hosting ``pg``'s bundle *bundle_index*.

    Used to resolve the master-addr for multi-node TP after ``pg.ready()``:
    the rank-0 actor will schedule into this same bundle, so its peers can
    connect to this IP.
    """
    return _run_in_bundle(pg, bundle_index, _remote_get_node_ip)


# ---------------------------------------------------------------------------
# Orphan cleanup
# ---------------------------------------------------------------------------


def remove_named_pgs_with_prefix(prefix: str) -> int:
    """Remove all placement groups in the current namespace whose name starts with *prefix*.

    Requires a live Ray connection on the current driver. Intended for orphan
    cleanup after a driver restart: since PGs are namespace-scoped and named,
    a reconnecting driver (with matching ``namespace=``) can find and reap
    leftover state from a prior session. Removing a PG forcibly kills all
    actors scheduled into it, releasing the reserved resources.

    Returns the number of PGs removed.
    """
    from ray.util.placement_group import placement_group_table

    try:
        table = placement_group_table()
    except Exception:  # noqa: BLE001
        logger.debug("Could not list placement groups for orphan cleanup")
        return 0

    # placement_group_table may return {pg_id: meta} or a list of metas;
    # normalize to an iterable of metas.
    metas = table.values() if isinstance(table, dict) else list(table)

    removed = 0
    for meta in metas:
        name = meta.get("name") or ""
        state = meta.get("state", "")
        if not name.startswith(prefix):
            continue
        if state == "REMOVED":
            continue
        pg_id = meta.get("placement_group_id") or meta.get("id")
        if pg_id is None:
            continue
        try:
            pg = ray.util.get_placement_group(name)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Could not look up orphaned PG {name!r}: {exc}")
            continue
        with contextlib.suppress(Exception):
            ray.util.remove_placement_group(pg)
            removed += 1
            logger.debug(f"Removed orphaned placement group: {name}")
    if removed:
        logger.info(f"Removed {removed} orphaned placement group(s) with prefix '{prefix}'")
    return removed
