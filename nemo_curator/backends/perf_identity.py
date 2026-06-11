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

Each executor backend resolves ``WorkerPerfIdentity`` once at worker setup from
**that backend's own APIs only**. ``BaseStageAdapter`` copies the values stamped
on ``WorkerMetadata`` — no cross-backend fallback chain.

``gpu_id`` remains the stable within-run join key for ``per_gpu`` aggregation.
``physical_address`` / ``pod_ip`` / ``hostname`` / ``gpu_indices`` are additive
cluster-location metadata for debugging and cross-artifact correlation.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.utils.performance_utils import StagePerfStats


@dataclass(frozen=True)
class WorkerPerfIdentity:
    """Perf identity resolved once per worker at backend setup."""

    actor_id: str = ""
    node_id: str = ""
    gpu_id: str = ""
    physical_address: str = ""
    pod_ip: str = ""
    hostname: str = ""
    gpu_indices: tuple[int, ...] = ()
    gpu_uuids: tuple[str, ...] = ()


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


def _resolve_hostname() -> str:
    try:
        return (socket.gethostname() or "").strip()
    except OSError:
        return ""


def _resolve_pod_ip() -> str:
    for key in ("POD_IP", "STATUS_POD_IP"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return ""


def _resolve_host_ip() -> str:
    pod_ip = _resolve_pod_ip()
    if pod_ip:
        return pod_ip
    try:
        import ray

        return (ray.util.get_node_ip_address() or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _allocation_gpu_indices(allocation: object | None, requires_gpu: bool) -> tuple[int, ...]:
    if not requires_gpu or allocation is None:
        return ()
    gpus = getattr(allocation, "gpus", None) or []
    indices: list[int] = []
    for gpu in gpus:
        idx = getattr(gpu, "index", None)
        if idx is not None:
            indices.append(int(idx))
    return tuple(indices)


def _collect_gpu_uuids(gpu_indices: tuple[int, ...]) -> tuple[str, ...]:
    if not gpu_indices:
        return ()
    uuids: list[str] = []
    try:
        import torch

        if not torch.cuda.is_available():
            return ()
        for idx in gpu_indices:
            props = torch.cuda.get_device_properties(idx)
            uuid = str(getattr(props, "uuid", "") or "").strip()
            if uuid:
                uuids.append(uuid)
    except Exception:  # noqa: BLE001
        return ()
    return tuple(uuids)


def _format_physical_address(host_ip: str, gpu_indices: tuple[int, ...]) -> str:
    if not host_ip or not gpu_indices:
        return ""
    idx_part = ",".join(str(idx) for idx in gpu_indices)
    return f"{host_ip}:{idx_part}"


def build_xenna_perf_identity(
    stage_name: str,
    *,
    worker_id: str,
    node_id: str,
    allocation: object | None,
    requires_gpu: bool,
) -> WorkerPerfIdentity:
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
    gpu_indices: tuple[int, ...] = ()
    if requires_gpu and allocation is not None:
        gpu_indices = _allocation_gpu_indices(allocation, requires_gpu=True)
        if gpu_indices:
            gpu_label = _format_gpu_label(node_label, gpu_indices[0])

    host_ip = _resolve_host_ip()
    physical_address = _format_physical_address(host_ip, gpu_indices)
    gpu_uuids = _collect_gpu_uuids(gpu_indices) if gpu_indices else ()

    return WorkerPerfIdentity(
        actor_id=actor_label,
        node_id=node_label,
        gpu_id=gpu_label,
        physical_address=physical_address,
        pod_ip=_resolve_pod_ip(),
        hostname=_resolve_hostname(),
        gpu_indices=gpu_indices,
        gpu_uuids=gpu_uuids,
    )


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


def _ray_gpu_indices(requires_gpu: bool) -> tuple[int, ...]:
    if not requires_gpu:
        return ()
    try:
        import ray

        gpu_ids = ray.get_gpu_ids()
        if gpu_ids:
            return tuple(int(gpu_id) for gpu_id in gpu_ids)
    except Exception:  # noqa: BLE001
        return ()
    return ()


def _ray_gpu_label(node_label: str, requires_gpu: bool) -> str:
    gpu_indices = _ray_gpu_indices(requires_gpu)
    if gpu_indices:
        return _format_gpu_label(node_label, gpu_indices[0])
    return ""


def build_ray_perf_identity(
    stage_name: str,
    *,
    requires_gpu: bool,
) -> WorkerPerfIdentity:
    """Identity from Ray runtime context (Ray Data / Ray Actor Pool).

    GPU index comes **only** from ``ray.get_gpu_ids()`` when the stage requests
    a GPU. No ``CUDA_VISIBLE_DEVICES`` parsing. Only resolves inside a Ray
    worker process (returns blanks on the driver).
    """
    blank = WorkerPerfIdentity()
    if os.environ.get("RAY_WORKER_ID") is None:
        return blank

    try:
        import ray

        ctx = ray.get_runtime_context()
    except Exception:  # noqa: BLE001
        return blank

    node_label = _ray_node_label(ctx)
    actor_label = _format_actor_label(stage_name, _ray_worker_short_id(ctx))
    gpu_indices = _ray_gpu_indices(requires_gpu)
    gpu_label = _format_gpu_label(node_label, gpu_indices[0]) if gpu_indices else ""
    host_ip = ""
    try:
        import ray

        host_ip = (ray.util.get_node_ip_address() or "").strip()
    except Exception:  # noqa: BLE001
        host_ip = ""
    physical_address = _format_physical_address(host_ip, gpu_indices)
    gpu_uuids = _collect_gpu_uuids(gpu_indices) if gpu_indices else ()

    return WorkerPerfIdentity(
        actor_id=actor_label,
        node_id=node_label,
        gpu_id=gpu_label,
        physical_address=physical_address,
        pod_ip=_resolve_pod_ip(),
        hostname=_resolve_hostname(),
        gpu_indices=gpu_indices,
        gpu_uuids=gpu_uuids,
    )


def read_worker_metadata_identity(
    stage_name: str,
    worker_metadata: WorkerMetadata | None,
) -> WorkerPerfIdentity:
    """Return perf labels previously stamped on ``WorkerMetadata`` by the backend."""
    if worker_metadata is None:
        return WorkerPerfIdentity()
    actor_id = (worker_metadata.actor_id or "").strip()
    node_id = (worker_metadata.node_id or "").strip()
    gpu_id = (worker_metadata.gpu_id or "").strip()
    if not (actor_id or node_id or gpu_id):
        return WorkerPerfIdentity()
    return WorkerPerfIdentity(
        actor_id=actor_id or stage_name,
        node_id=node_id,
        gpu_id=gpu_id,
        physical_address=(worker_metadata.physical_address or "").strip(),
        pod_ip=(worker_metadata.pod_ip or "").strip(),
        hostname=(worker_metadata.hostname or "").strip(),
        gpu_indices=tuple(worker_metadata.gpu_indices or ()),
        gpu_uuids=tuple(worker_metadata.gpu_uuids or ()),
    )


def stamp_worker_metadata(worker_metadata: WorkerMetadata, identity: WorkerPerfIdentity) -> None:
    """Copy a resolved identity onto generic ``WorkerMetadata``."""
    worker_metadata.actor_id = identity.actor_id
    worker_metadata.node_id = identity.node_id
    worker_metadata.gpu_id = identity.gpu_id
    worker_metadata.physical_address = identity.physical_address
    worker_metadata.pod_ip = identity.pod_ip
    worker_metadata.hostname = identity.hostname
    worker_metadata.gpu_indices = list(identity.gpu_indices)
    worker_metadata.gpu_uuids = list(identity.gpu_uuids)


def apply_worker_perf_identity(stage_perf_stats: StagePerfStats, identity: WorkerPerfIdentity) -> None:
    """Copy resolved worker identity onto a ``StagePerfStats`` record."""
    stage_perf_stats.actor_id = identity.actor_id
    stage_perf_stats.node_id = identity.node_id
    stage_perf_stats.gpu_id = identity.gpu_id
    stage_perf_stats.physical_address = identity.physical_address
    stage_perf_stats.pod_ip = identity.pod_ip
    stage_perf_stats.hostname = identity.hostname
    stage_perf_stats.gpu_indices = list(identity.gpu_indices)
    stage_perf_stats.gpu_uuids = list(identity.gpu_uuids)
