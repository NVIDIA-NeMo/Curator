# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN401, BLE001, C901, EM101, EM102, PLR0912, S110, S112, TRY300, TRY301

from __future__ import annotations

import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.pipeline.payload_refs import (
    PayloadRef,
    _get_named_actor,
    heartbeat_payload_refs_batched,
    release_payload_ref,
    resolve_payload_refs_batched,
    strip_payload_refs,
    task_payload_refs,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, Task

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo


_DEFAULT_NODE_MEMORY_FRACTION = 0.70
_DEFAULT_LEASE_TTL_S = 3600.0
_DEFAULT_POLL_INTERVAL_S = 0.25
_DEFAULT_ADMISSION_WAIT_TIMEOUT_S = 4 * 60 * 60
_DEFAULT_MATERIALIZED_LEASE_TTL_S = 4 * 60 * 60
_DEFAULT_SAMPLE_WIDTH_BYTES = 4


def _ray_get(obj: Any) -> Any:
    import ray

    return ray.get(obj)


def _resolve_node_id() -> str:
    try:
        import ray

        ctx = ray.get_runtime_context()
        node_id = getattr(ctx, "get_node_id", lambda: None)()
        if node_id:
            return str(node_id)
    except Exception:
        pass
    return os.uname().nodename


def _safe_actor_suffix(value: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return suffix or "unknown"


def _current_ray_namespace() -> str | None:
    try:
        import ray

        ctx = ray.get_runtime_context()
        namespace = getattr(ctx, "namespace", None)
        if callable(namespace):
            namespace = namespace()
        if not namespace:
            get_namespace = getattr(ctx, "get_namespace", None)
            if callable(get_namespace):
                namespace = get_namespace()
        if namespace:
            return str(namespace)
    except Exception:
        pass
    return None


def _parse_byte_limit(value: str | None, *, field_name: str = "byte limit") -> int | None:
    if not value:
        return None
    text = value.strip().lower()
    try:
        if text.endswith("k"):
            parsed = int(float(text[:-1]) * 1024)
        elif text.endswith("m"):
            parsed = int(float(text[:-1]) * 1024**2)
        elif text.endswith("g"):
            parsed = int(float(text[:-1]) * 1024**3)
        else:
            parsed = int(text)
    except ValueError as exc:
        msg = f"{field_name} must be an integer byte count or a k/m/g byte string, got {value!r}"
        raise ValueError(msg) from exc
    if parsed <= 0:
        msg = f"{field_name} must be positive, got {value!r}"
        raise ValueError(msg)
    return parsed


def _detect_memory_limit_bytes() -> int | None:
    cgroup_paths = (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )
    for path in cgroup_paths:
        try:
            with open(path, encoding="utf-8") as f:
                raw = f.read().strip()
            if raw and raw != "max":
                value = int(raw)
                if value > 0 and value < 1 << 60:
                    return value
        except Exception:
            continue

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages * page_size)
    except Exception:
        return None
    return None


def _detect_memory_usage_bytes() -> int:
    cgroup_paths = (
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    )
    for path in cgroup_paths:
        try:
            with open(path, encoding="utf-8") as f:
                raw = f.read().strip()
            if raw:
                return max(0, int(raw))
        except Exception:
            continue
    try:
        import resource

        # ru_maxrss is KiB on Linux.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    except Exception:
        return 0


def _resolve_node_payload_budget(
    explicit_bytes: int | None,
    memory_fraction: float,
) -> int:
    if explicit_bytes is not None:
        return max(1, int(explicit_bytes))
    memory_limit = _detect_memory_limit_bytes()
    if memory_limit is None:
        # Conservative fallback for development machines where cgroups are not visible.
        return int(32 * 1024**3 * memory_fraction)
    return max(1, int(memory_limit * memory_fraction))


def _payload_object_bytes(payload: Any) -> int:
    if isinstance(payload, torch.Tensor):
        return int(payload.element_size() * payload.nelement())
    nbytes = getattr(payload, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return len(payload)
    return 0


def _duration_to_payload_bytes(
    duration_s: float,
    sample_rate: int,
    channels: int,
    sample_width_bytes: int,
) -> int:
    if duration_s <= 0:
        raise ValueError("Audio payload byte admission requires positive duration before decode")
    return max(1, int(duration_s * sample_rate * channels * sample_width_bytes))


def _lease_expires_at(lease_ttl_s: float) -> float | None:
    ttl = float(lease_ttl_s)
    if ttl <= 0:
        return None
    return time.monotonic() + ttl


def _task_payload_estimate_bytes(
    task: Task,
    *,
    duration_key: str,
    sample_rate: int,
    channels: int,
    sample_width_bytes: int,
) -> int:
    raw_duration = task.data.get(duration_key)
    if raw_duration is None:
        raise ValueError(f"Audio payload byte admission requires '{duration_key}' in each row before audio decode")
    try:
        duration_s = float(raw_duration)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Audio payload duration must be numeric, got {raw_duration!r}") from exc
    return _duration_to_payload_bytes(duration_s, sample_rate, channels, sample_width_bytes)


class _PayloadAdmissionState:
    def __init__(
        self,
        *,
        default_node_budget_bytes: int,
        default_cluster_budget_bytes: int | None = None,
        default_lease_ttl_s: float = _DEFAULT_LEASE_TTL_S,
    ) -> None:
        self.default_node_budget_bytes = max(1, int(default_node_budget_bytes))
        self.default_cluster_budget_bytes = (
            max(1, int(default_cluster_budget_bytes)) if default_cluster_budget_bytes is not None else None
        )
        self.default_lease_ttl_s = float(default_lease_ttl_s)
        self._node_budget: dict[str, int] = {}
        self._node_used: dict[str, int] = {}
        self._cluster_used = 0
        self._leases: dict[tuple[str, str], tuple[int, float | None]] = {}

    def register_node(self, node_id: str, budget_bytes: int | None = None) -> None:
        budget = self.default_node_budget_bytes if budget_bytes is None else int(budget_bytes)
        self._node_budget[node_id] = max(1, budget)
        self._node_used.setdefault(node_id, 0)
        self._reap_expired()

    def try_acquire(self, node_id: str, owner_id: str, amount_bytes: int, lease_ttl_s: float | None = None) -> bool:
        amount = int(amount_bytes)
        if amount <= 0:
            return True
        self._reap_expired()
        self.register_node(node_id)
        used = self._node_used[node_id]
        budget = self._node_budget[node_id]
        cluster_budget = self._cluster_budget_bytes()
        if amount > budget:
            return False
        if used + amount > budget:
            return False
        if amount > cluster_budget:
            return False
        if self._cluster_used + amount > cluster_budget:
            return False
        ttl = self.default_lease_ttl_s if lease_ttl_s is None else float(lease_ttl_s)
        self._node_used[node_id] = used + amount
        self._cluster_used += amount
        self._leases[(node_id, owner_id)] = (amount, _lease_expires_at(ttl))
        return True

    def heartbeat(self, node_id: str, owner_id: str, lease_ttl_s: float | None = None) -> bool:
        self._reap_expired()
        return self._heartbeat(node_id, owner_id, lease_ttl_s)

    def heartbeat_many(self, requests: list[tuple[str, str, float | None]]) -> list[bool]:
        """Refresh several admission leases in one actor RPC and one reap pass."""
        self._reap_expired()
        return [self._heartbeat(node_id, owner_id, lease_ttl_s) for node_id, owner_id, lease_ttl_s in requests]

    def _heartbeat(self, node_id: str, owner_id: str, lease_ttl_s: float | None) -> bool:
        key = (node_id, owner_id)
        if key not in self._leases:
            return False
        amount, expires_at = self._leases[key]
        if expires_at is not None:
            ttl = self.default_lease_ttl_s if lease_ttl_s is None else float(lease_ttl_s)
            expires_at = _lease_expires_at(ttl)
        self._leases[key] = (amount, expires_at)
        return True

    def release(self, node_id: str, owner_id: str, amount_bytes: int | None = None) -> None:
        key = (node_id, owner_id)
        lease = self._leases.pop(key, None)
        if lease is None:
            return
        reserved, _ = lease
        amount = reserved if amount_bytes is None else min(reserved, int(amount_bytes))
        self._node_used[node_id] = max(0, self._node_used.get(node_id, 0) - amount)
        self._cluster_used = max(0, self._cluster_used - amount)

    def resize(self, node_id: str, owner_id: str, new_amount_bytes: int, lease_ttl_s: float | None = None) -> bool:
        self._reap_expired()
        key = (node_id, owner_id)
        lease = self._leases.get(key)
        if lease is None:
            return self.try_acquire(node_id, owner_id, new_amount_bytes, lease_ttl_s)

        old_amount, _ = lease
        new_amount = int(new_amount_bytes)
        if new_amount <= old_amount:
            delta = old_amount - new_amount
            self._node_used[node_id] = max(0, self._node_used.get(node_id, 0) - delta)
            self._cluster_used = max(0, self._cluster_used - delta)
            ttl = self.default_lease_ttl_s if lease_ttl_s is None else float(lease_ttl_s)
            expires_at = lease[1]
            if expires_at is not None:
                expires_at = _lease_expires_at(ttl)
            self._leases[key] = (new_amount, expires_at)
            return True

        delta = new_amount - old_amount
        budget = self._node_budget.get(node_id, self.default_node_budget_bytes)
        used = self._node_used.get(node_id, 0)
        cluster_budget = self._cluster_budget_bytes()
        if used + delta > budget:
            return False
        if self._cluster_used + delta > cluster_budget:
            return False
        self._node_used[node_id] = used + delta
        self._cluster_used += delta
        ttl = self.default_lease_ttl_s if lease_ttl_s is None else float(lease_ttl_s)
        expires_at = lease[1]
        if expires_at is not None:
            expires_at = _lease_expires_at(ttl)
        self._leases[key] = (new_amount, expires_at)
        return True

    def snapshot(self) -> dict[str, Any]:
        self._reap_expired()
        return {
            "node_budget": dict(self._node_budget),
            "node_used": dict(self._node_used),
            "cluster_budget": self._cluster_budget_bytes(),
            "cluster_used": self._cluster_used,
            "lease_count": len(self._leases),
        }

    def _cluster_budget_bytes(self) -> int:
        if self.default_cluster_budget_bytes is not None:
            return self.default_cluster_budget_bytes
        return max(1, sum(self._node_budget.values()) or self.default_node_budget_bytes)

    def _reap_expired(self) -> None:
        now = time.monotonic()
        expired = [key for key, (_, expires_at) in self._leases.items() if expires_at is not None and expires_at < now]
        for node_id, owner_id in expired:
            self.release(node_id, owner_id)


@dataclass
class _StoredPayload:
    payload: Any
    amount_bytes: int
    expires_at: float | None


class _PayloadStoreState:
    def __init__(self, *, default_lease_ttl_s: float = _DEFAULT_LEASE_TTL_S) -> None:
        self.default_lease_ttl_s = float(default_lease_ttl_s)
        self._payloads: dict[str, _StoredPayload] = {}

    def put(self, payload_id: str, payload: Any, amount_bytes: int, lease_ttl_s: float | None = None) -> None:
        self._reap_expired()
        ttl = self.default_lease_ttl_s if lease_ttl_s is None else float(lease_ttl_s)
        self._payloads[payload_id] = _StoredPayload(payload, int(amount_bytes), _lease_expires_at(ttl))

    def get(self, payload_id: str, lease_ttl_s: float | None = None) -> Any:
        self._reap_expired()
        return self._get(payload_id, lease_ttl_s)

    def get_many(self, requests: list[tuple[str, float | None]]) -> list[Any]:
        """Resolve several payloads in request order in one actor RPC and one reap pass."""
        self._reap_expired()
        return [self._get(payload_id, lease_ttl_s) for payload_id, lease_ttl_s in requests]

    def _get(self, payload_id: str, lease_ttl_s: float | None) -> Any:
        stored = self._payloads[payload_id]
        if stored.expires_at is not None:
            ttl = self.default_lease_ttl_s if lease_ttl_s is None else float(lease_ttl_s)
            stored.expires_at = _lease_expires_at(ttl)
        return stored.payload

    def pin(self, payload_id: str, lease_ttl_s: float | None = None) -> bool:
        self._reap_expired()
        return self._pin(payload_id, lease_ttl_s)

    def pin_many(self, requests: list[tuple[str, float | None]]) -> list[bool]:
        """Refresh several store leases in one actor RPC and one reap pass."""
        self._reap_expired()
        return [self._pin(payload_id, lease_ttl_s) for payload_id, lease_ttl_s in requests]

    def _pin(self, payload_id: str, lease_ttl_s: float | None) -> bool:
        stored = self._payloads.get(payload_id)
        if stored is None:
            return False
        if stored.expires_at is not None:
            ttl = self.default_lease_ttl_s if lease_ttl_s is None else float(lease_ttl_s)
            stored.expires_at = _lease_expires_at(ttl)
        return True

    def release(self, payload_id: str) -> int:
        stored = self._payloads.pop(payload_id, None)
        if stored is None:
            return 0
        return stored.amount_bytes

    def snapshot(self) -> dict[str, Any]:
        self._reap_expired()
        return {
            "payload_count": len(self._payloads),
            "payload_bytes": sum(payload.amount_bytes for payload in self._payloads.values()),
        }

    def _reap_expired(self) -> None:
        now = time.monotonic()
        expired = [
            payload_id
            for payload_id, payload in self._payloads.items()
            if payload.expires_at is not None and payload.expires_at < now
        ]
        for payload_id in expired:
            self._payloads.pop(payload_id, None)


def _kill_named_actor(name: str, namespace: str | None = None) -> bool:
    try:
        import ray

        actor = _get_named_actor(name, namespace)
        ray.kill(actor, no_restart=True)
        return True
    except ValueError:
        return False
    except Exception as exc:
        logger.warning(f"Failed to kill payload actor {name!r}: {exc}")
        return False


def _active_ray_node_ids() -> list[str]:
    try:
        import ray

        return [str(node["NodeID"]) for node in ray.nodes() if node.get("NodeID") and node.get("Alive", True)]
    except Exception:
        return []


def _get_named_actor_or_create(
    actor_cls: type,
    name: str,
    *,
    node_id: str | None = None,
    namespace: str | None = None,
    **kwargs: Any,
) -> Any:
    import ray

    try:
        return _get_named_actor(name, namespace)
    except ValueError:
        options: dict[str, Any] = {"name": name, "get_if_exists": True, "lifetime": "detached"}
        if namespace:
            options["namespace"] = namespace
        if node_id:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        return ray.remote(actor_cls).options(**options).remote(**kwargs)


def _get_admission_actor(
    actor_name: str,
    *,
    default_node_budget_bytes: int,
    default_cluster_budget_bytes: int | None,
    default_lease_ttl_s: float,
    namespace: str | None,
) -> Any:
    return _get_named_actor_or_create(
        _PayloadAdmissionState,
        actor_name,
        namespace=namespace,
        default_node_budget_bytes=default_node_budget_bytes,
        default_cluster_budget_bytes=default_cluster_budget_bytes,
        default_lease_ttl_s=default_lease_ttl_s,
    )


def _get_store_actor(actor_name: str, *, node_id: str, default_lease_ttl_s: float, namespace: str | None) -> Any:
    return _get_named_actor_or_create(
        _PayloadStoreState,
        actor_name,
        node_id=node_id,
        namespace=namespace,
        default_lease_ttl_s=default_lease_ttl_s,
    )


class _PayloadLeaseKeeper:
    def __init__(self, payload_refs: list[PayloadRef], *, interval_s: float | None = None) -> None:
        deduped: dict[tuple[str | None, str, str], PayloadRef] = {}
        for payload_ref in payload_refs:
            key = (payload_ref.actor_namespace, payload_ref.store_actor_name, payload_ref.payload_id)
            deduped[key] = payload_ref
        self._payload_refs = list(deduped.values())
        if interval_s is None:
            ttl_s = min((payload_ref.lease_ttl_s for payload_ref in self._payload_refs), default=_DEFAULT_LEASE_TTL_S)
            interval_s = min(30.0, max(1.0, ttl_s / 3.0))
        self._interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned = False

    def start(self) -> None:
        if not self._payload_refs or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="curator-payload-lease-keeper",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, min(5.0, self._interval_s)))
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                heartbeat_payload_refs_batched(self._payload_refs)
            except Exception as exc:
                if not self._warned:
                    logger.warning(
                        "Payload lease heartbeat failed; one or more payloads may expire during long stage work: {}",
                        exc,
                    )
                    self._warned = True


class PayloadAwareStageMixin:
    """Mixin for stages that need waveform payload handles at ``process_batch`` time."""

    waveform_ref_key: str | None
    waveform_key: str
    sample_rate_key: str
    num_samples_key: str

    def payload_bindings(self) -> list[dict[str, str]]:
        """Return payload-ref bindings consumed by this stage.

        Stages with one waveform can rely on the legacy ``waveform_*`` fields.
        Multi-input stages can override this method and return one mapping per
        payload, each with ``ref_key`` and ``waveform_key`` plus optional
        ``sample_rate_key`` and ``num_samples_key``.
        """

        payload_ref_key = getattr(self, "waveform_ref_key", None)
        if not payload_ref_key:
            return []
        return [
            {
                "ref_key": str(payload_ref_key),
                "waveform_key": str(getattr(self, "waveform_key", "waveform")),
                "sample_rate_key": str(getattr(self, "sample_rate_key", "sample_rate")),
                "num_samples_key": str(getattr(self, "num_samples_key", "num_samples")),
            }
        ]

    def resolve_payload_refs_for_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        bindings = self.payload_bindings()
        if not bindings:
            return []
        self._stop_payload_lease_keeper()
        inserted: list[AudioTask] = []
        payload_refs: list[PayloadRef] = []
        pending: list[tuple[AudioTask, dict[str, str], PayloadRef]] = []
        consumer_node_id = _resolve_node_id()
        resolution_start = time.perf_counter()
        same_node_count = 0
        cross_node_count = 0
        resolved_bytes = 0
        try:
            for task in tasks:
                task_inserted = False
                for binding in bindings:
                    payload_ref_key = binding["ref_key"]
                    payload_key = binding["waveform_key"]
                    if payload_key in task.data:
                        continue
                    payload_ref = task.data.get(payload_ref_key)
                    if payload_ref is None:
                        continue
                    if not isinstance(payload_ref, PayloadRef):
                        msg = (
                            f"Task {task.task_id} has non-PayloadRef '{payload_ref_key}' "
                            f"value: {type(payload_ref).__name__}"
                        )
                        raise TypeError(msg)
                    if payload_ref.owner_node_id and payload_ref.owner_node_id == consumer_node_id:
                        same_node_count += 1
                    else:
                        cross_node_count += 1
                    resolved_bytes += int(payload_ref.amount_bytes)
                    pending.append((task, binding, payload_ref))
                    payload_refs.append(payload_ref)
                    task_inserted = True
                if task_inserted:
                    inserted.append(task)

            payloads = resolve_payload_refs_batched(
                payload_refs,
                max_batch_bytes=getattr(self, "payload_resolve_max_batch_bytes", None),
            )
            for (task, binding, payload_ref), payload in zip(pending, payloads, strict=True):
                task.data[binding["waveform_key"]] = payload
                task.data[binding.get("sample_rate_key", "sample_rate")] = payload_ref.sample_rate
                task.data.setdefault(binding.get("num_samples_key", "num_samples"), payload_ref.num_samples)
        except Exception:
            for task in inserted:
                for binding in bindings:
                    task.data.pop(binding["waveform_key"], None)
            self._stop_payload_lease_keeper()
            raise
        self._start_payload_lease_keeper(payload_refs)
        if payload_refs:
            log_metrics = getattr(self, "_log_metrics", None)
            if callable(log_metrics):
                log_metrics(
                    {
                        "payload_resolution_count": float(len(payload_refs)),
                        "payload_resolution_same_node_count": float(same_node_count),
                        "payload_resolution_cross_node_count": float(cross_node_count),
                        "payload_resolution_bytes": float(resolved_bytes),
                        "payload_resolution_time_s": time.perf_counter() - resolution_start,
                    }
                )
        return inserted

    def drop_resolved_payloads(self, tasks: list[AudioTask]) -> None:
        self._stop_payload_lease_keeper()
        payload_keys = [binding["waveform_key"] for binding in self.payload_bindings()]
        for task in tasks:
            for payload_key in payload_keys:
                task.data.pop(payload_key, None)

    def terminal_tombstone_drop_data_keys(self) -> tuple[str, ...]:
        return tuple({binding["waveform_key"] for binding in self.payload_bindings()})

    @staticmethod
    def payload_consumer_node_id() -> str:
        """Return the node currently resolving payloads for locality metrics."""
        return _resolve_node_id()

    def _start_payload_lease_keeper(self, payload_refs: list[PayloadRef]) -> None:
        if not payload_refs:
            return
        keeper = _PayloadLeaseKeeper(payload_refs)
        keeper.start()
        self._payload_lease_keeper = keeper

    def _stop_payload_lease_keeper(self) -> None:
        keeper = getattr(self, "_payload_lease_keeper", None)
        if keeper is None:
            return
        keeper.stop()
        self._payload_lease_keeper = None


@dataclass
class AudioPayloadMaterializeStage(ProcessingStage[AudioTask, AudioTask]):
    """Read audio once, decode to memory, and replace the waveform with a payload handle."""

    _curator_pipeline_helper_stage = True

    name: str = "AudioPayloadMaterializeStage"
    target_sample_rate: int = 16000
    target_nchannels: int = 1
    audio_filepath_key: str = "audio_filepath"
    duration_key: str = "duration"
    segment_start_key: str = "segment_start_s"
    segment_duration_key: str = "segment_duration_s"
    waveform_key: str = "waveform"
    waveform_ref_key: str = "waveform_ref"
    sample_rate_key: str = "sample_rate"
    num_samples_key: str = "num_samples"
    skip_me_key: str = "_skip_me"
    read_error_key: str = "audio_read_error"
    skip_on_read_error: bool = False
    node_memory_fraction: float = _DEFAULT_NODE_MEMORY_FRACTION
    max_node_payload_bytes: int | str | None = None
    max_cluster_payload_bytes: int | str | None = None
    lease_ttl_s: float = _DEFAULT_LEASE_TTL_S
    materialized_lease_ttl_s: float = _DEFAULT_MATERIALIZED_LEASE_TTL_S
    admission_poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S
    admission_wait_timeout_s: float = _DEFAULT_ADMISSION_WAIT_TIMEOUT_S
    admission_actor_name: str = "curator_payload_admission"
    store_actor_prefix: str = "curator_payload_store"
    run_id: str | None = None
    sample_width_bytes: int = _DEFAULT_SAMPLE_WIDTH_BYTES
    verbose: bool = False

    _reader: Any = field(init=False, default=None, repr=False)
    _node_id: str = field(init=False, default="", repr=False)
    _node_budget_bytes: int = field(init=False, default=0, repr=False)
    _cluster_budget_bytes: int | None = field(init=False, default=None, repr=False)
    _actor_run_suffix: str = field(init=False, default="", repr=False)
    _admission_actor_name: str = field(init=False, default="", repr=False)
    _store_actor_name: str = field(init=False, default="", repr=False)
    _actor_namespace: str | None = field(init=False, default=None, repr=False)
    _admission: Any = field(init=False, default=None, repr=False)
    _store: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.lease_ttl_s <= 0:
            raise ValueError("lease_ttl_s must be positive while a payload is being materialized")
        if self.materialized_lease_ttl_s <= 0:
            raise ValueError("materialized_lease_ttl_s must be positive")
        if self.admission_poll_interval_s <= 0:
            raise ValueError("admission_poll_interval_s must be positive")
        if self.admission_wait_timeout_s <= 0:
            raise ValueError("admission_wait_timeout_s must be positive")
        self.batch_size = 1
        if self.resources is None:
            self.resources = {"cpus": 1.0}
        self.run_id = str(self.run_id or uuid.uuid4().hex)
        self._actor_run_suffix = _safe_actor_suffix(self.run_id)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, self.duration_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.waveform_ref_key, self.sample_rate_key, self.num_samples_key]

    def setup_on_node(self, node_info: NodeInfo, worker_metadata: dict[str, Any] | None = None) -> None:
        self._node_id = node_info.node_id or _resolve_node_id()
        self._ensure_ready()
        self._reader.setup_on_node(node_info, worker_metadata)

    def setup(self, worker_metadata: dict[str, Any] | None = None) -> None:
        self._ensure_ready()
        self._reader.setup(worker_metadata)

    def teardown(self) -> None:
        if self._reader is not None:
            self._reader.teardown()

    def process(self, task: AudioTask) -> AudioTask:
        self._ensure_ready()
        payload_id = uuid.uuid4().hex
        estimated_bytes = _task_payload_estimate_bytes(
            task,
            duration_key=self._estimate_duration_key(task),
            sample_rate=self.target_sample_rate,
            channels=self.target_nchannels,
            sample_width_bytes=self.sample_width_bytes,
        )
        admission_wait_s, admission_poll_count = self._acquire(payload_id, estimated_bytes)
        reserved_bytes = estimated_bytes
        stored = False
        self._log_metrics(
            {
                "payload_admission_wait_s": admission_wait_s,
                "payload_admission_poll_count": admission_poll_count,
                "payload_estimated_bytes": float(estimated_bytes),
                "payload_reserved_bytes": float(reserved_bytes),
                "payload_node_budget_bytes": float(self._node_budget_bytes),
                "payload_cluster_budget_bytes": float(self._cluster_budget_bytes or 0),
            }
        )
        try:
            decoded = self._reader.process(task)
            waveform = decoded.data.pop(self.waveform_key, None)
            if waveform is None:
                self._release(payload_id, reserved_bytes)
                return decoded
            if self._is_reader_skip_result(decoded):
                self._release(payload_id, reserved_bytes)
                decoded.data.pop(self.waveform_key, None)
                return decoded

            actual_bytes = _payload_object_bytes(waveform)
            if actual_bytes <= 0:
                self._release(payload_id, reserved_bytes)
                raise RuntimeError("Decoded audio waveform has unknown or zero byte size")

            if actual_bytes != reserved_bytes:
                if not _ray_get(
                    self._admission.resize.remote(
                        self._node_id,
                        payload_id,
                        actual_bytes,
                        self.lease_ttl_s,
                    )
                ):
                    self._release(payload_id, reserved_bytes)
                    raise RuntimeError(
                        "Insufficient payload memory budget after audio decode "
                        f"(estimated={reserved_bytes}, actual={actual_bytes})"
                    )
                reserved_bytes = actual_bytes

            _ray_get(self._store.put.remote(payload_id, waveform, actual_bytes, self.materialized_lease_ttl_s))
            stored = True
            self._log_metrics(
                {
                    "payload_stored_bytes": float(actual_bytes),
                    "payload_materialized_count": 1.0,
                }
            )
            decoded.data[self.waveform_ref_key] = PayloadRef(
                payload_id=payload_id,
                owner_node_id=self._node_id,
                store_actor_name=self._store_actor_name,
                admission_actor_name=self._admission_actor_name,
                amount_bytes=actual_bytes,
                sample_rate=int(decoded.data[self.sample_rate_key]),
                num_samples=int(decoded.data[self.num_samples_key]),
                lease_ttl_s=self.lease_ttl_s,
                actor_namespace=self._actor_namespace,
            )
            decoded.data["_curator_payload_estimated_bytes"] = estimated_bytes
            decoded.data["_curator_payload_bytes"] = actual_bytes
            if not _ray_get(
                self._admission.heartbeat.remote(
                    self._node_id,
                    payload_id,
                    self.materialized_lease_ttl_s,
                )
            ):
                raise RuntimeError(f"Payload reservation expired before materialization completed: {payload_id}")
            return decoded
        except Exception:
            if stored:
                try:
                    _ray_get(self._store.release.remote(payload_id))
                except Exception:
                    logger.debug("Failed to release stored payload {} after materialization error", payload_id)
            self._release(payload_id, reserved_bytes)
            task.data.pop(self.waveform_key, None)
            raise

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task!s} failed validation for stage {self}"
                raise ValueError(msg)
        return [self.process(task) for task in tasks]

    def _ensure_ready(self) -> None:
        if self._reader is None:
            from nemo_curator.stages.audio.io.audio_file_reader import AudioFileReaderStage

            self._reader = AudioFileReaderStage(
                target_sample_rate=self.target_sample_rate,
                target_nchannels=self.target_nchannels,
                audio_filepath_key=self.audio_filepath_key,
                duration_key=self.duration_key,
                segment_start_key=self.segment_start_key,
                segment_duration_key=self.segment_duration_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
                num_samples_key=self.num_samples_key,
                skip_me_key=self.skip_me_key,
                read_error_key=self.read_error_key,
                skip_on_read_error=self.skip_on_read_error,
                verbose=self.verbose,
            )

        if self._admission is None or self._store is None:
            self._node_id = self._node_id or _resolve_node_id()
            self._actor_namespace = _current_ray_namespace()
            explicit_budget = self.max_node_payload_bytes
            if isinstance(explicit_budget, str):
                explicit_budget = _parse_byte_limit(explicit_budget, field_name="max_node_payload_bytes")
            explicit_cluster_budget = self.max_cluster_payload_bytes
            if isinstance(explicit_cluster_budget, str):
                explicit_cluster_budget = _parse_byte_limit(
                    explicit_cluster_budget,
                    field_name="max_cluster_payload_bytes",
                )
            self._cluster_budget_bytes = explicit_cluster_budget
            self._node_budget_bytes = _resolve_node_payload_budget(
                explicit_budget,
                self.node_memory_fraction,
            )
            self._admission_actor_name = f"{self.admission_actor_name}_{self._actor_run_suffix}"
            self._store_actor_name = (
                f"{self.store_actor_prefix}_{self._actor_run_suffix}_{_safe_actor_suffix(self._node_id)}"
            )
            self._admission = _get_admission_actor(
                self._admission_actor_name,
                default_node_budget_bytes=self._node_budget_bytes,
                default_cluster_budget_bytes=explicit_cluster_budget,
                default_lease_ttl_s=self.lease_ttl_s,
                namespace=self._actor_namespace,
            )
            self._store = _get_store_actor(
                self._store_actor_name,
                node_id=self._node_id,
                default_lease_ttl_s=self.lease_ttl_s,
                namespace=self._actor_namespace,
            )
            _ray_get(self._admission.register_node.remote(self._node_id, self._node_budget_bytes))

    def _estimate_duration_key(self, task: AudioTask) -> str:
        if self.segment_duration_key in task.data and task.data.get(self.segment_duration_key) is not None:
            return self.segment_duration_key
        return self.duration_key

    def _is_reader_skip_result(self, task: AudioTask) -> bool:
        if not self.skip_on_read_error:
            return False
        return self.skip_me_key in task.data or self.read_error_key in task.data

    def _acquire(self, payload_id: str, amount_bytes: int) -> tuple[float, int]:
        if amount_bytes > self._node_budget_bytes:
            raise RuntimeError(
                f"Single audio payload estimate {amount_bytes} bytes exceeds node payload budget "
                f"{self._node_budget_bytes} bytes"
            )
        if self._cluster_budget_bytes is not None and amount_bytes > self._cluster_budget_bytes:
            raise RuntimeError(
                f"Single audio payload estimate {amount_bytes} bytes exceeds cluster payload budget "
                f"{self._cluster_budget_bytes} bytes"
            )
        start = time.perf_counter()
        polls = 0
        while True:
            polls += 1
            acquired = _ray_get(
                self._admission.try_acquire.remote(
                    self._node_id,
                    payload_id,
                    amount_bytes,
                    self.lease_ttl_s,
                )
            )
            if acquired:
                return time.perf_counter() - start, polls
            elapsed_s = time.perf_counter() - start
            if elapsed_s >= self.admission_wait_timeout_s:
                snapshot = _ray_get(self._admission.snapshot.remote())
                raise RuntimeError(
                    "Timed out waiting for payload admission "
                    f"after {elapsed_s:.3f}s for {amount_bytes} bytes; admission={snapshot}"
                )
            time.sleep(self.admission_poll_interval_s)

    def _release(self, payload_id: str, amount_bytes: int) -> None:
        try:
            _ray_get(self._admission.release.remote(self._node_id, payload_id, amount_bytes))
        except Exception:
            logger.debug("Failed to release payload admission tokens for {}", payload_id)

    def cleanup_run_resources(self) -> None:
        suffix = _safe_actor_suffix(str(self.run_id))
        namespace = self._actor_namespace or _current_ray_namespace()
        _kill_named_actor(f"{self.admission_actor_name}_{suffix}", namespace)

        store_prefix = f"{self.store_actor_prefix}_{suffix}_"
        for node_id in _active_ray_node_ids():
            _kill_named_actor(f"{store_prefix}{_safe_actor_suffix(node_id)}", namespace)

    def ray_stage_spec(self) -> dict[str, Any]:
        spec = super().ray_stage_spec()
        spec[RayStageSpecKeys.IS_ACTOR_STAGE] = False
        return spec

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {
            "is_actor_stage": False,
            "is_fanout_stage": False,
            "is_repartition_stage": False,
        }


@dataclass
class PayloadReleaseStage(ProcessingStage[AudioTask, AudioTask]):
    _curator_pipeline_helper_stage = True

    name: str = "PayloadReleaseStage"
    payload_ref_key: str = "waveform_ref"
    waveform_key: str = "waveform"
    remove_payload_metadata: bool = True

    def __post_init__(self) -> None:
        self.batch_size = 1
        if self.resources is None:
            self.resources = {"cpus": 0.1}

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: AudioTask) -> AudioTask:
        released_ids: set[str] = set()
        released_bytes = 0
        for payload_ref in task_payload_refs(task):
            if payload_ref.payload_id in released_ids:
                continue
            release_payload_ref(payload_ref)
            released_ids.add(payload_ref.payload_id)
            released_bytes += int(payload_ref.amount_bytes)
        if released_ids:
            self._log_metrics(
                {
                    "payload_release_count": float(len(released_ids)),
                    "payload_release_bytes": float(released_bytes),
                }
            )
        if isinstance(task.data, dict):
            stripped_data = strip_payload_refs(task.data)
            task.data.clear()
            task.data.update(stripped_data)
        task.data.pop(self.waveform_key, None)
        if self.remove_payload_metadata:
            for key in tuple(task.data):
                if str(key).startswith("_curator_payload_"):
                    task.data.pop(key, None)
        return task

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        return [self.process(task) for task in tasks]

    def ray_stage_spec(self) -> dict[str, Any]:
        spec = super().ray_stage_spec()
        spec[RayStageSpecKeys.IS_ACTOR_STAGE] = False
        return spec

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {
            "is_actor_stage": False,
            "is_fanout_stage": False,
            "is_repartition_stage": False,
        }
