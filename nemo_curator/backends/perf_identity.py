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

"""Backend-specific perf identity labels.

Each executor backend resolves ``(actor_id, node_id, gpu_id)`` once at worker
setup from **that backend's own APIs only**. ``BaseStageAdapter`` copies the
values stamped on ``WorkerMetadata`` — no cross-backend fallback chain.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


def _format_gpu_label(node_label: str, gpu_index: object) -> str:
    idx_str = str(gpu_index).strip()
    if not idx_str:
        return ""
    return f"{node_label}:{idx_str}" if node_label else idx_str


def _format_actor_label(stage_name: str, worker_or_actor_id: str) -> str:
    wid = (worker_or_actor_id or "").strip()
    if not wid:
        return stage_name
    return f"{stage_name}:actor-{wid[:8]}"


def build_xenna_perf_identity(
    stage_name: str,
    *,
    worker_id: str,
    node_id: str,
    allocation: object | None,
    requires_gpu: bool,
) -> tuple[str, str, str]:
    """Identity from Xenna ``WorkerMetadata`` + ``NodeInfo`` (allocation-first GPU).

    GPU index comes **only** from ``allocation.gpus[0].index``. Node label comes
    from Xenna's ``node_id``, then the MPI rank env (launcher-provided fact on
    multi-node jobs), then ``allocation.node``.
    """
    node_label = (node_id or "").strip()
    if not node_label:
        rank = os.environ.get("OMPI_COMM_WORLD_RANK")
        if rank not in (None, ""):
            node_label = f"node-{rank}"
        elif allocation is not None:
            node_label = str(getattr(allocation, "node", "") or "").strip()

    actor_label = _format_actor_label(stage_name, worker_id)

    gpu_label = ""
    if requires_gpu and allocation is not None:
        gpus = getattr(allocation, "gpus", None) or []
        if gpus:
            idx = getattr(gpus[0], "index", None)
            if idx is not None:
                gpu_label = _format_gpu_label(node_label, idx)

    return actor_label, node_label, gpu_label


def _ray_node_label(ctx: object) -> str:
    try:
        node_hex = getattr(ctx, "get_node_id", lambda: "")()
        if node_hex:
            return f"node-{str(node_hex)[:8]}"
    except Exception:  # noqa: BLE001
        return ""


def _ray_worker_short_id(ctx: object) -> str:
    short_id = ""
    try:
        short_id = (getattr(ctx, "get_actor_id", lambda: "")() or "") if hasattr(ctx, "get_actor_id") else ""
    except Exception:  # noqa: BLE001
        short_id = ""
    if short_id:
        return short_id
    try:
        return getattr(ctx, "get_worker_id", lambda: "")() or ""
    except Exception:  # noqa: BLE001
        return ""


def _ray_gpu_label(node_label: str, requires_gpu: bool) -> str:
    if not requires_gpu:
        return ""
    try:
        import ray

        gpu_ids = ray.get_gpu_ids()
        if gpu_ids:
            return _format_gpu_label(node_label, gpu_ids[0])
    except Exception:  # noqa: BLE001
        return ""


def build_ray_perf_identity(
    stage_name: str,
    *,
    requires_gpu: bool,
) -> tuple[str, str, str]:
    """Identity from Ray runtime context (Ray Data / Ray Actor Pool).

    GPU index comes **only** from ``ray.get_gpu_ids()`` when the stage requests
    a GPU. No ``CUDA_VISIBLE_DEVICES`` parsing. Only resolves inside a Ray
    worker process (returns blanks on the driver).
    """
    # Ray sets worker-scoped env vars; the driver has no assigned GPU/worker id.
    if os.environ.get("RAY_WORKER_ID") is None:
        return "", "", ""

    try:
        import ray

        ctx = ray.get_runtime_context()
    except Exception:  # noqa: BLE001
        return "", "", ""

    node_label = _ray_node_label(ctx)
    actor_label = _format_actor_label(stage_name, _ray_worker_short_id(ctx))
    gpu_label = _ray_gpu_label(node_label, requires_gpu)
    return actor_label, node_label, gpu_label


def read_worker_metadata_identity(
    stage_name: str,
    worker_metadata: WorkerMetadata | None,
) -> tuple[str, str, str]:
    """Return perf labels previously stamped on ``WorkerMetadata`` by the backend."""
    if worker_metadata is None:
        return "", "", ""
    actor_id = (worker_metadata.actor_id or "").strip()
    node_id = (worker_metadata.node_id or "").strip()
    gpu_id = (worker_metadata.gpu_id or "").strip()
    if not (actor_id or node_id or gpu_id):
        return "", "", ""
    return actor_id or stage_name, node_id, gpu_id
