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
# ruff: noqa: S110, S112, SIM105

"""Backend-specific perf identity labels.

Each backend resolves ``WorkerPerfIdentity`` once at worker setup from its own
APIs; ``BaseStageAdapter`` copies the values stamped on ``WorkerMetadata``.

Metrics aggregate by ``actor_id``. ``physical_address`` (``<host>:<gpu_indices>``)
is the canonical, backend-independent GPU identifier on each GPU actor's block;
the remaining fields are additive cluster-location metadata for debugging.
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


def _visible_gpu_ordinals(gpu_indices: tuple[int, ...], visible_count: int) -> list[int]:
    """Translate physical CUDA indices to torch's *visible* ordinals.

    ``gpu_indices`` are physical ids but torch enumerates only the devices in
    ``CUDA_VISIBLE_DEVICES`` as ordinals ``0..visible_count-1``. Map via the env
    when it lists integer ids; under per-worker isolation every visible ordinal
    belongs to this worker.
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not env:
        # No mask -> all GPUs visible -> physical index == torch ordinal.
        return [i for i in gpu_indices if 0 <= i < visible_count]
    phys_to_ordinal: dict[int, int] = {}
    for ordinal, token in enumerate(t.strip() for t in env.split(",") if t.strip()):
        try:
            phys_to_ordinal[int(token)] = ordinal
        except ValueError:
            phys_to_ordinal = {}  # UUID-style mask -> no positional int mapping
            break
    mapped = [phys_to_ordinal[i] for i in gpu_indices if i in phys_to_ordinal]
    if mapped:
        return mapped
    # Isolated worker (or unmappable ids): the visible set *is* this worker's.
    return list(range(visible_count))


def _collect_gpu_uuids(gpu_indices: tuple[int, ...]) -> tuple[str, ...]:
    if not gpu_indices:
        return ()
    try:
        import torch

        if not torch.cuda.is_available():
            return ()
        visible_count = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        return ()
    uuids: list[str] = []
    # Per-ordinal guard: one bad index must not wipe the rest.
    for ordinal in _visible_gpu_ordinals(gpu_indices, visible_count):
        try:
            uuid = str(getattr(torch.cuda.get_device_properties(ordinal), "uuid", "") or "").strip()
        except Exception:  # noqa: BLE001
            continue
        if uuid:
            uuids.append(uuid)
    return tuple(uuids)


def _format_physical_address(host_token: str, gpu_indices: tuple[int, ...]) -> str:
    """Canonical physical GPU address: ``<host>:<idx[,idx...]>``.

    ``host_token`` degrades to ``node`` so a GPU worker always gets a non-empty,
    backend-independent identifier. Returns ``""`` only when it holds no GPUs.
    """
    if not gpu_indices:
        return ""
    host = (host_token or "").strip() or "node"
    idx_part = ",".join(str(idx) for idx in gpu_indices)
    return f"{host}:{idx_part}"


def build_xenna_perf_identity(
    stage_name: str,
    *,
    worker_id: str,
    node_id: str,
    allocation: object | None,
    requires_gpu: bool,
) -> WorkerPerfIdentity:
    """Identity from Xenna ``WorkerMetadata`` + ``NodeInfo`` (allocation-first GPU).

    GPU index comes only from ``allocation.gpus[0].index``; node label falls back
    ``node_id`` -> MPI rank env -> ``allocation.node``.
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

    hostname = _resolve_hostname()
    physical_address = _format_physical_address(_resolve_host_ip() or hostname or node_label, gpu_indices)
    gpu_uuids = _collect_gpu_uuids(gpu_indices) if gpu_indices else ()

    return WorkerPerfIdentity(
        actor_id=actor_label,
        node_id=node_label,
        gpu_id=gpu_label,
        physical_address=physical_address,
        pod_ip=_resolve_pod_ip(),
        hostname=hostname,
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


def _gpu_assignment_tokens(values: object) -> tuple[str, ...]:
    """Return non-empty GPU assignment tokens from Ray/env values."""
    if values is None:
        return ()
    if isinstance(values, str):
        iterable = values.split(",")
    else:
        try:
            iterable = list(values)  # type: ignore[arg-type]
        except TypeError:
            iterable = [values]
    return tuple(token for token in (str(value).strip() for value in iterable) if token)


def _parse_int_indices(values: object) -> tuple[int, ...]:
    """Best-effort int-index parse; silently drops non-integer (e.g. UUID) ids."""
    out: list[int] = []
    for value in _gpu_assignment_tokens(values):
        try:
            out.append(int(str(value).strip()))
        except (TypeError, ValueError):
            continue  # UUID-style assignment -> no positional index
    return tuple(out)


def _normalize_gpu_uuid(value: object) -> str:
    try:
        text = value.decode() if isinstance(value, bytes) else str(value)
    except Exception:  # noqa: BLE001
        text = ""
    text = text.strip().lower()
    return text.removeprefix("gpu-")


def _uuid_gpu_assignment(tokens: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Map Ray/CUDA UUID assignment tokens back to physical GPU indices with NVML."""
    wanted = tuple((token, _normalize_gpu_uuid(token)) for token in tokens if _normalize_gpu_uuid(token))
    if not wanted:
        return (), ()
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001
        return (), ()

    matches: dict[str, int] = {}
    try:
        for index in range(int(pynvml.nvmlDeviceGetCount())):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                uuid = _normalize_gpu_uuid(pynvml.nvmlDeviceGetUUID(handle))
            except Exception:  # noqa: BLE001
                continue
            if uuid:
                matches[uuid] = index
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass

    indices: list[int] = []
    uuids: list[str] = []
    for token, normalized in wanted:
        if normalized in matches:
            indices.append(matches[normalized])
            uuids.append(token)
    return tuple(indices), tuple(uuids)


def _gpu_assignment_from_tokens(tokens: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[str, ...]]:
    indices = _parse_int_indices(tokens)
    if indices:
        return indices, _collect_gpu_uuids(indices)
    return _uuid_gpu_assignment(tokens)


def _ray_gpu_assignment(requires_gpu: bool) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if not requires_gpu:
        return (), ()
    try:
        import ray

        tokens = _gpu_assignment_tokens(ray.get_gpu_ids())
    except Exception:  # noqa: BLE001
        tokens = ()

    indices, uuids = _gpu_assignment_from_tokens(tokens)
    if indices:
        return indices, uuids

    # Ray may leave CUDA_VISIBLE_DEVICES set to this worker's assigned slice
    # when get_gpu_ids() is empty. Support both integer and UUID masks.
    env_tokens = _gpu_assignment_tokens(os.environ.get("CUDA_VISIBLE_DEVICES"))
    return _gpu_assignment_from_tokens(env_tokens)


def _ray_gpu_indices(requires_gpu: bool) -> tuple[int, ...]:
    return _ray_gpu_assignment(requires_gpu)[0]


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

    GPU assignment comes from ``ray.get_gpu_ids()``, falling back to
    ``CUDA_VISIBLE_DEVICES`` when Ray returns no ids. Supports both integer and
    UUID assignments so Ray Data actors still start GPU utilization sampling.
    """
    blank = WorkerPerfIdentity()

    try:
        import ray

        if hasattr(ray, "is_initialized") and not ray.is_initialized():
            return blank
        ctx = ray.get_runtime_context()
    except Exception:  # noqa: BLE001
        return blank

    worker_id = _ray_worker_short_id(ctx)
    node_label = _ray_node_label(ctx)
    if not (worker_id or node_label):
        return blank

    actor_label = _format_actor_label(stage_name, worker_id)
    gpu_indices, gpu_uuids = _ray_gpu_assignment(requires_gpu)
    gpu_label = _format_gpu_label(node_label, gpu_indices[0]) if gpu_indices else ""
    host_ip = ""
    try:
        import ray

        host_ip = (ray.util.get_node_ip_address() or "").strip()
    except Exception:  # noqa: BLE001
        host_ip = ""
    hostname = _resolve_hostname()
    physical_address = _format_physical_address(host_ip or hostname or node_label, gpu_indices)
    if gpu_indices and not gpu_uuids:
        gpu_uuids = _collect_gpu_uuids(gpu_indices)

    return WorkerPerfIdentity(
        actor_id=actor_label,
        node_id=node_label,
        gpu_id=gpu_label,
        physical_address=physical_address,
        pod_ip=_resolve_pod_ip(),
        hostname=hostname,
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
