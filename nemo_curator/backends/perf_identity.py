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

"""Backend worker identity and opt-in stage telemetry."""

from __future__ import annotations

import os
import socket
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.utils.performance_utils import norm_gpu_uuid

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.tasks import Task
    from nemo_curator.utils.performance_utils import StagePerfStats


@dataclass(frozen=True)
class WorkerPerfIdentity:
    """Backend-resolved identity for one stage worker."""

    actor_id: str = ""
    node_id: str = ""
    gpu_id: str = ""
    physical_address: str = ""
    pod_ip: str = ""
    hostname: str = ""
    gpu_indices: tuple[int, ...] = ()
    gpu_uuids: tuple[str, ...] = ()


def _text(value: object) -> str:
    return (value.decode() if isinstance(value, bytes) else str(value)).strip()


def _tokens(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = values.split(",")
    else:
        try:
            values = list(values)  # type: ignore[arg-type]
        except TypeError:
            values = [values]
    return tuple(token for value in values if (token := _text(value)))


def _nvml_inventory() -> list[tuple[int, str]]:
    """Return physical NVML indices and UUIDs, or an empty fail-open result."""
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001
        return []
    try:
        devices = []
        for index in range(int(pynvml.nvmlDeviceGetCount())):
            with suppress(Exception):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                devices.append((index, _text(pynvml.nvmlDeviceGetUUID(handle))))
        return devices
    finally:
        with suppress(Exception):
            pynvml.nvmlShutdown()


def _gpu_assignment(values: object) -> tuple[tuple[int, ...], tuple[str, ...]]:
    tokens = _tokens(values)
    if not tokens:
        return (), ()
    inventory = _nvml_inventory()
    try:
        indices = tuple(int(token) for token in tokens)
    except ValueError:
        by_uuid = {norm_gpu_uuid(uuid): (index, uuid) for index, uuid in inventory}
        matches = [by_uuid[norm_gpu_uuid(token)] for token in tokens if norm_gpu_uuid(token) in by_uuid]
        return tuple(index for index, _ in matches), tuple(
            token for token in tokens if norm_gpu_uuid(token) in by_uuid
        )
    by_index = dict(inventory)
    uuids = tuple(by_index[index] for index in indices if index in by_index)
    return indices, uuids if len(uuids) == len(indices) else ()


def _hostname() -> str:
    try:
        return (socket.gethostname() or "").strip()
    except OSError:
        return ""


def _pod_ip() -> str:
    return next(
        (value for key in ("POD_IP", "STATUS_POD_IP") if (value := (os.environ.get(key) or "").strip())),
        "",
    )


def _runtime_ip() -> str:
    if pod_ip := _pod_ip():
        return pod_ip
    try:
        import ray

        return (ray.util.get_node_ip_address() or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _build_identity(
    stage_name: str,
    worker_id: str,
    node_id: str,
    gpu_indices: tuple[int, ...],
    gpu_uuids: tuple[str, ...],
) -> WorkerPerfIdentity:
    hostname = _hostname()
    address_host = _runtime_ip() or hostname or node_id or "node"
    indices = ",".join(map(str, gpu_indices))
    return WorkerPerfIdentity(
        actor_id=f"{stage_name}:actor-{worker_id[:8]}" if worker_id else stage_name,
        node_id=node_id,
        gpu_id=f"{node_id}:{gpu_indices[0]}"
        if node_id and gpu_indices
        else str(gpu_indices[0])
        if gpu_indices
        else "",
        physical_address=f"{address_host}:{indices}" if indices else "",
        pod_ip=_pod_ip(),
        hostname=hostname,
        gpu_indices=gpu_indices,
        gpu_uuids=gpu_uuids,
    )


def build_xenna_perf_identity(
    stage_name: str,
    *,
    worker_id: str,
    node_id: str,
    allocation: object | None,
    requires_gpu: bool,
) -> WorkerPerfIdentity:
    """Resolve identity from a Xenna worker allocation."""
    node_id = (node_id or "").strip()
    if not node_id:
        rank = os.environ.get("OMPI_COMM_WORLD_RANK")
        node_id = f"node-{rank}" if rank not in (None, "") else str(getattr(allocation, "node", "") or "").strip()
    gpu_values = [gpu.index for gpu in (getattr(allocation, "gpus", None) or [])] if requires_gpu else []
    indices, uuids = _gpu_assignment(gpu_values)
    return _build_identity(stage_name, worker_id, node_id, indices, uuids)


def _runtime_value(context: object, method: str) -> str:
    try:
        return _text(getattr(context, method, lambda: "")() or "")
    except Exception:  # noqa: BLE001
        return ""


def build_ray_perf_identity(stage_name: str, *, requires_gpu: bool) -> WorkerPerfIdentity:
    """Resolve identity from the current Ray worker."""
    try:
        import ray

        if hasattr(ray, "is_initialized") and not ray.is_initialized():
            return WorkerPerfIdentity()
        context = ray.get_runtime_context()
    except Exception:  # noqa: BLE001
        return WorkerPerfIdentity()

    worker_id = _runtime_value(context, "get_actor_id") or _runtime_value(context, "get_worker_id")
    node = _runtime_value(context, "get_node_id")
    node_id = f"node-{node[:8]}" if node else ""
    if not (worker_id or node_id):
        return WorkerPerfIdentity()

    gpu_values: object = ()
    if requires_gpu:
        with suppress(Exception):
            gpu_values = ray.get_gpu_ids()
    indices, uuids = _gpu_assignment(gpu_values)
    if requires_gpu and not indices:
        indices, uuids = _gpu_assignment(os.environ.get("CUDA_VISIBLE_DEVICES"))
    return _build_identity(stage_name, worker_id, node_id, indices, uuids)


def apply_worker_perf_identity(stats: StagePerfStats, identity: WorkerPerfIdentity) -> None:
    """Copy worker identity into one invocation record."""
    for name in ("actor_id", "node_id", "gpu_id", "physical_address", "pod_ip", "hostname"):
        setattr(stats, name, getattr(identity, name))
    stats.gpu_indices = list(identity.gpu_indices)
    stats.gpu_uuids = list(identity.gpu_uuids)


class StageTelemetry:
    """Worker-local identity and sampler for one opted-in stage."""

    def __init__(self, stage: ProcessingStage, worker_metadata: WorkerMetadata | None) -> None:
        self.stage = stage
        identity = getattr(worker_metadata, "perf_identity", None)
        self.identity = identity if isinstance(identity, WorkerPerfIdentity) else WorkerPerfIdentity()
        self.sampler: Any | None = None
        resources = getattr(stage, "resources", None)
        if not getattr(resources, "requires_gpu", False) or not self.identity.gpu_uuids:
            return
        try:
            from nemo_curator.utils.gpu_sampler import GpuUtilSampler

            self.sampler = GpuUtilSampler(gpu_uuids=self.identity.gpu_uuids, sample_all_visible=False)
            self.sampler.start()
        except Exception as exc:  # noqa: BLE001
            logger.debug("GPU sampler unavailable for {}: {}", stage.name, exc)
            self.sampler = None

    def enrich(self, stats: StagePerfStats) -> None:
        if self.sampler is not None:
            stats.custom_metrics.update(self.sampler.window_metrics(stats.window_start_s, stats.window_end_s))
        apply_worker_perf_identity(stats, self.identity)

    def close(self) -> None:
        if self.sampler is not None:
            self.sampler.stop()
            self.sampler = None


class PerformanceTelemetryAdapterMixin:
    """Implement the shared adapter hooks with backend identity and NVML data."""

    stage: ProcessingStage
    _performance_telemetry: StageTelemetry | None

    def _setup_performance_telemetry(self, worker_metadata: WorkerMetadata | None) -> None:
        self._performance_telemetry = (
            StageTelemetry(self.stage, worker_metadata)
            if bool(getattr(self.stage, "extended_performance_metrics", False))
            else None
        )

    def _enrich_stage_perf_record(self, stats: StagePerfStats, _results: list[Task]) -> None:
        if self._performance_telemetry is not None:
            self._performance_telemetry.enrich(stats)

    def _teardown_performance_telemetry(self) -> None:
        if self._performance_telemetry is not None:
            self._performance_telemetry.close()
            self._performance_telemetry = None
