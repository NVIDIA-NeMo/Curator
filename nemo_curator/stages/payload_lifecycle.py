# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Payload values and backend actor handles are intentionally runtime-typed.
# ruff: noqa: ANN401, S110, TRY301

"""Backend-visible payload storage and audio materialization stages."""

from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from nemo_curator.pipeline.payload_refs import (
    PAYLOAD_RESERVATION_LEASE_TTL_S,
    PayloadRef,
    _get_named_actor,
    iter_payload_items,
    map_payload_items,
    release_payload_ref,
    resolve_payload_refs_batched,
    strip_payload_refs,
    task_payload_refs,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, Task

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

_DEFAULT_NODE_MEMORY_FRACTION = 0.70
_DEFAULT_SAMPLE_WIDTH_BYTES = 4


def _ray_get(value: Any) -> Any:
    import ray

    return ray.get(value)


def _resolve_node_id() -> str:
    try:
        import ray

        node_id = ray.get_runtime_context().get_node_id()
        if node_id:
            return str(node_id)
    except Exception:  # noqa: BLE001
        pass
    return os.uname().nodename


def _current_ray_namespace() -> str | None:
    try:
        import ray

        context = ray.get_runtime_context()
        namespace = getattr(context, "namespace", None)
        namespace = namespace() if callable(namespace) else namespace
        return str(namespace) if namespace else None
    except Exception:  # noqa: BLE001
        return None


def _safe_actor_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value) or "unknown"


def _parse_byte_limit(value: int | str | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        msg = f"{field_name} must be a positive byte count"
        raise TypeError(msg)
    if isinstance(value, int):
        parsed = value
    else:
        text = str(value).strip().lower()
        multiplier = 1
        if text.endswith(("k", "m", "g")):
            multiplier = {"k": 1024, "m": 1024**2, "g": 1024**3}[text[-1]]
            text = text[:-1]
        try:
            parsed = int(float(text) * multiplier)
        except ValueError as exc:
            msg = f"{field_name} must be a byte count or k/m/g byte string, got {value!r}"
            raise ValueError(msg) from exc
    if parsed <= 0:
        msg = f"{field_name} must be positive, got {value!r}"
        raise ValueError(msg)
    return parsed


def _detect_memory_limit_bytes() -> int | None:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path, encoding="utf-8") as stream:
                raw = stream.read().strip()
            if raw and raw != "max":
                value = int(raw)
                if 0 < value < 1 << 60:
                    return value
        except (OSError, ValueError):
            continue
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        return None


def _payload_size_bytes(payload: Any) -> int:
    if isinstance(payload, torch.Tensor):
        return int(payload.element_size() * payload.nelement())
    nbytes = getattr(payload, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return len(payload)
    msg = f"Cannot determine payload size for {type(payload).__name__}"
    raise TypeError(msg)


def _lease_expiry(ttl_s: float) -> float | None:
    return time.monotonic() + ttl_s if ttl_s > 0 else None


class PayloadAdmission:
    """Byte-budget admission state hosted by one named Ray actor."""

    def __init__(
        self,
        node_budget_bytes: int,
        cluster_budget_bytes: int | None = None,
        lease_ttl_s: float = PAYLOAD_RESERVATION_LEASE_TTL_S,
    ) -> None:
        self.node_budget_bytes = int(node_budget_bytes)
        self.cluster_budget_bytes = int(cluster_budget_bytes) if cluster_budget_bytes is not None else None
        self.lease_ttl_s = float(lease_ttl_s)
        self._node_budgets: dict[str, int] = {}
        self._node_used: dict[str, int] = {}
        self._cluster_used = 0
        self._leases: dict[tuple[str, str], tuple[int, float | None]] = {}

    def register_node(self, node_id: str, budget_bytes: int | None = None) -> None:
        self._node_budgets[node_id] = int(budget_bytes or self.node_budget_bytes)
        self._node_used.setdefault(node_id, 0)
        self._reap_expired()

    def try_acquire(self, node_id: str, owner_id: str, amount_bytes: int, ttl_s: float | None = None) -> bool:
        self._reap_expired()
        self.register_node(node_id)
        amount = int(amount_bytes)
        node_budget = self._node_budgets[node_id]
        cluster_budget = self.cluster_budget_bytes or max(1, sum(self._node_budgets.values()))
        if amount <= 0 or amount > node_budget or amount > cluster_budget:
            return amount <= 0
        if self._node_used[node_id] + amount > node_budget or self._cluster_used + amount > cluster_budget:
            return False
        self._node_used[node_id] += amount
        self._cluster_used += amount
        self._leases[(node_id, owner_id)] = (amount, _lease_expiry(self.lease_ttl_s if ttl_s is None else ttl_s))
        return True

    def resize(self, node_id: str, owner_id: str, amount_bytes: int) -> bool:
        key = (node_id, owner_id)
        previous = self._leases.get(key)
        if previous is None:
            return self.try_acquire(node_id, owner_id, amount_bytes)
        old_amount, expiry = previous
        amount = int(amount_bytes)
        delta = amount - old_amount
        node_budget = self._node_budgets[node_id]
        cluster_budget = self.cluster_budget_bytes or max(1, sum(self._node_budgets.values()))
        if delta > 0 and (
            self._node_used[node_id] + delta > node_budget or self._cluster_used + delta > cluster_budget
        ):
            return False
        self._node_used[node_id] += delta
        self._cluster_used += delta
        self._leases[key] = (amount, expiry)
        return True

    def publish(self, node_id: str, owner_id: str) -> bool:
        self._reap_expired()
        key = (node_id, owner_id)
        lease = self._leases.get(key)
        if lease is None:
            return False
        self._leases[key] = (lease[0], None)
        return True

    def release(self, node_id: str, owner_id: str, _amount_bytes: int | None = None) -> None:
        lease = self._leases.pop((node_id, owner_id), None)
        if lease is None:
            return
        amount = lease[0]
        self._node_used[node_id] = max(0, self._node_used.get(node_id, 0) - amount)
        self._cluster_used = max(0, self._cluster_used - amount)

    def snapshot(self) -> dict[str, Any]:
        self._reap_expired()
        return {
            "node_budget": dict(self._node_budgets),
            "node_used": dict(self._node_used),
            "cluster_used": self._cluster_used,
            "lease_count": len(self._leases),
        }

    def _reap_expired(self) -> None:
        now = time.monotonic()
        expired = [key for key, (_, expiry) in self._leases.items() if expiry is not None and expiry < now]
        for node_id, owner_id in expired:
            self.release(node_id, owner_id)


@dataclass
class _StoredPayload:
    value: Any
    amount_bytes: int


class PayloadStore:
    """Node-affine payload storage hosted by one named Ray actor."""

    def __init__(self) -> None:
        self._payloads: dict[str, _StoredPayload] = {}

    def put(self, payload_id: str, payload: Any, amount_bytes: int) -> None:
        self._payloads[payload_id] = _StoredPayload(payload, int(amount_bytes))

    def get_many(self, payload_ids: list[str]) -> list[Any]:
        return [self._payloads[payload_id].value for payload_id in payload_ids]

    def release(self, payload_id: str) -> int:
        stored = self._payloads.pop(payload_id, None)
        return stored.amount_bytes if stored is not None else 0

    def snapshot(self) -> dict[str, int]:
        return {
            "payload_count": len(self._payloads),
            "payload_bytes": sum(item.amount_bytes for item in self._payloads.values()),
        }


def _get_or_create_actor(
    actor_class: type,
    name: str,
    *,
    namespace: str | None,
    node_id: str | None = None,
    args: tuple[Any, ...] = (),
) -> Any:
    import ray

    try:
        return _get_named_actor(name, namespace)
    except ValueError:
        options: dict[str, Any] = {"name": name, "get_if_exists": True}
        if namespace:
            options["namespace"] = namespace
        if node_id:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        return ray.remote(actor_class).options(**options).remote(*args)


@dataclass
class PayloadManager:
    """Worker-local client for admission-controlled, node-affine payload actors."""

    run_id: str
    max_node_payload_bytes: int | str | None = None
    max_cluster_payload_bytes: int | str | None = None
    node_memory_fraction: float = _DEFAULT_NODE_MEMORY_FRACTION
    admission_poll_interval_s: float = 0.25
    admission_wait_timeout_s: float = 4 * 60 * 60
    admission_actor_prefix: str = "curator_payload_admission"
    store_actor_prefix: str = "curator_payload_store"
    _node_id: str = field(init=False, default="")
    _namespace: str | None = field(init=False, default=None)
    _node_budget_bytes: int = field(init=False, default=0)
    _cluster_budget_bytes: int | None = field(init=False, default=None)
    _admission_name: str = field(init=False, default="")
    _store_name: str = field(init=False, default="")
    _admission: Any = field(init=False, default=None, repr=False)
    _store: Any = field(init=False, default=None, repr=False)

    def setup(self, node_id: str | None = None) -> None:
        if self._admission is not None:
            return
        self._node_id = node_id or _resolve_node_id()
        self._namespace = _current_ray_namespace()
        explicit_node_budget = _parse_byte_limit(self.max_node_payload_bytes, field_name="max_node_payload_bytes")
        memory_limit = _detect_memory_limit_bytes() or 32 * 1024**3
        self._node_budget_bytes = explicit_node_budget or max(1, int(memory_limit * self.node_memory_fraction))
        self._cluster_budget_bytes = _parse_byte_limit(
            self.max_cluster_payload_bytes, field_name="max_cluster_payload_bytes"
        )
        suffix = _safe_actor_suffix(self.run_id)
        self._admission_name = f"{self.admission_actor_prefix}_{suffix}"
        self._store_name = f"{self.store_actor_prefix}_{suffix}_{_safe_actor_suffix(self._node_id)}"
        self._admission = _get_or_create_actor(
            PayloadAdmission,
            self._admission_name,
            namespace=self._namespace,
            args=(self._node_budget_bytes, self._cluster_budget_bytes),
        )
        self._store = _get_or_create_actor(
            PayloadStore,
            self._store_name,
            namespace=self._namespace,
            node_id=self._node_id,
        )
        _ray_get(self._admission.register_node.remote(self._node_id, self._node_budget_bytes))

    def put(self, payload: Any, *, metadata: dict[str, Any], estimated_bytes: int) -> PayloadRef:
        self.setup()
        payload_id = uuid.uuid4().hex
        self._acquire(payload_id, estimated_bytes)
        reserved_bytes = estimated_bytes
        stored = False
        try:
            actual_bytes = _payload_size_bytes(payload)
            if actual_bytes != reserved_bytes:
                if not _ray_get(self._admission.resize.remote(self._node_id, payload_id, actual_bytes)):
                    msg = (
                        "Insufficient payload memory after materialization "
                        f"(estimated={reserved_bytes}, actual={actual_bytes})"
                    )
                    raise RuntimeError(msg)
                reserved_bytes = actual_bytes
            _ray_get(self._store.put.remote(payload_id, payload, actual_bytes))
            stored = True
            if not _ray_get(self._admission.publish.remote(self._node_id, payload_id)):
                msg = f"Payload reservation expired before publication: {payload_id}"
                raise RuntimeError(msg)
            return PayloadRef(
                payload_id=payload_id,
                owner_node_id=self._node_id,
                store_actor_name=self._store_name,
                admission_actor_name=self._admission_name,
                amount_bytes=actual_bytes,
                actor_namespace=self._namespace,
                metadata=dict(metadata),
            )
        except Exception:
            if stored:
                _ray_get(self._store.release.remote(payload_id))
            _ray_get(self._admission.release.remote(self._node_id, payload_id, reserved_bytes))
            raise

    def _acquire(self, payload_id: str, amount_bytes: int) -> None:
        if amount_bytes > self._node_budget_bytes:
            msg = f"Payload estimate {amount_bytes} exceeds node budget {self._node_budget_bytes}"
            raise RuntimeError(msg)
        started = time.monotonic()
        while not _ray_get(
            self._admission.try_acquire.remote(
                self._node_id,
                payload_id,
                amount_bytes,
                PAYLOAD_RESERVATION_LEASE_TTL_S,
            )
        ):
            if time.monotonic() - started >= self.admission_wait_timeout_s:
                snapshot = _ray_get(self._admission.snapshot.remote())
                msg = f"Timed out waiting for payload admission: {snapshot}"
                raise RuntimeError(msg)
            time.sleep(self.admission_poll_interval_s)


class PayloadAwareStageMixin:
    """Resolve payload handles only for the duration of a consumer batch."""

    waveform_ref_key: str | None
    waveform_key: str
    sample_rate_key: str
    num_samples_key: str

    def payload_binding(self) -> dict[str, str] | None:
        if not self.waveform_ref_key:
            return None
        return {
            "ref_key": self.waveform_ref_key,
            "payload_key": self.waveform_key,
            "sample_rate_key": self.sample_rate_key,
            "num_samples_key": self.num_samples_key,
        }

    def resolve_payload_refs_for_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        binding = self.payload_binding()
        if binding is None:
            return []
        pending: list[tuple[AudioTask, PayloadRef]] = []
        for task in tasks:
            if binding["payload_key"] in task.data:
                continue
            payload_ref = task.data.get(binding["ref_key"])
            if payload_ref is None:
                continue
            if not isinstance(payload_ref, PayloadRef):
                msg = f"Task {task.task_id} has invalid payload ref {type(payload_ref).__name__}"
                raise TypeError(msg)
            pending.append((task, payload_ref))
        payloads = resolve_payload_refs_batched([payload_ref for _, payload_ref in pending])
        inserted: list[AudioTask] = []
        try:
            for (task, payload_ref), payload in zip(pending, payloads, strict=True):
                task.data[binding["payload_key"]] = payload
                task.data[binding["sample_rate_key"]] = int(payload_ref.metadata["sample_rate"])
                task.data.setdefault(binding["num_samples_key"], int(payload_ref.metadata["num_samples"]))
                inserted.append(task)
        except Exception:
            self.drop_resolved_payloads(inserted)
            raise
        return inserted

    def drop_resolved_payloads(self, tasks: list[AudioTask]) -> None:
        binding = self.payload_binding()
        if binding is not None:
            for task in tasks:
                task.data.pop(binding["payload_key"], None)


@dataclass
class AudioPayloadMaterializeStage(ProcessingStage[AudioTask, AudioTask]):
    """Decode audio once and replace the waveform with a lightweight handle."""

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
    max_node_payload_bytes: int | str | None = None
    max_cluster_payload_bytes: int | str | None = None
    node_memory_fraction: float = _DEFAULT_NODE_MEMORY_FRACTION
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 1
    _reader: Any = field(init=False, default=None, repr=False)
    _manager: PayloadManager | None = field(init=False, default=None, repr=False)

    @classmethod
    def from_payload_config(
        cls,
        *,
        payload_spec: Any,
        payload_config: dict[str, Any],
        run_id: str,
    ) -> AudioPayloadMaterializeStage:
        """Construct the audio default through the generic dotted-target contract."""
        return cls(
            name=str(payload_config.get("materialize_stage_name", "audio_payload_materialize")),
            target_sample_rate=int(payload_config.get("target_sample_rate", 16000)),
            target_nchannels=int(payload_config.get("target_nchannels", 1)),
            audio_filepath_key=str(payload_config.get("source_key", "audio_filepath")),
            duration_key=str(payload_config.get("duration_key", "duration")),
            segment_start_key=str(payload_config.get("segment_start_key", "segment_start_s")),
            segment_duration_key=str(payload_config.get("segment_duration_key", "segment_duration_s")),
            waveform_key=payload_spec.payload_key,
            waveform_ref_key=payload_spec.ref_key,
            sample_rate_key=payload_spec.sample_rate_key,
            num_samples_key=payload_spec.num_samples_key,
            max_node_payload_bytes=payload_config.get("max_node_payload_bytes"),
            max_cluster_payload_bytes=payload_config.get("max_cluster_payload_bytes"),
            node_memory_fraction=float(payload_config.get("node_memory_fraction", _DEFAULT_NODE_MEMORY_FRACTION)),
            run_id=run_id,
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, self.duration_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.waveform_ref_key, self.sample_rate_key, self.num_samples_key]

    def setup_on_node(self, node_info: NodeInfo | None = None, worker_metadata: WorkerMetadata | None = None) -> None:
        self._ensure_ready()
        self._reader.setup_on_node(node_info, worker_metadata)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        self._ensure_ready()
        self._reader.setup(worker_metadata)

    def process(self, task: AudioTask) -> AudioTask:
        self._ensure_ready()
        duration_s = float(task.data.get(self.segment_duration_key, task.data[self.duration_key]))
        if duration_s <= 0:
            msg = f"{self.duration_key} must be positive before audio materialization"
            raise ValueError(msg)
        estimated_bytes = max(
            1,
            int(duration_s * self.target_sample_rate * self.target_nchannels * _DEFAULT_SAMPLE_WIDTH_BYTES),
        )
        decoded = self._reader.process(task)
        waveform = decoded.data.pop(self.waveform_key)
        decoded.data[self.waveform_ref_key] = self._manager.put(
            waveform,
            metadata={
                "sample_rate": int(decoded.data[self.sample_rate_key]),
                "num_samples": int(decoded.data[self.num_samples_key]),
                "dtype": str(waveform.dtype),
            },
            estimated_bytes=estimated_bytes,
        )
        return decoded

    def process_batch(self, tasks: list[Any]) -> list[Any]:
        def materialize(item: Any) -> Any:
            if not isinstance(item, AudioTask):
                msg = f"{type(self).__name__} envelope items must be AudioTask, got {type(item).__name__}"
                raise TypeError(msg)
            if not self.validate_input(item):
                msg = f"Task {item} failed validation for {type(self).__name__}"
                raise ValueError(msg)
            return self.process(item)

        return [map_payload_items(task, materialize) for task in tasks]

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
                skip_on_read_error=False,
            )
        if self._manager is None:
            self._manager = PayloadManager(
                run_id=self.run_id,
                max_node_payload_bytes=self.max_node_payload_bytes,
                max_cluster_payload_bytes=self.max_cluster_payload_bytes,
                node_memory_fraction=self.node_memory_fraction,
            )
        self._manager.setup()


def build_audio_payload_materialize_stage(
    *,
    payload_spec: Any,
    payload_config: dict[str, Any],
    run_id: str,
) -> AudioPayloadMaterializeStage:
    """Build this PR's default audio materializer from the generic planner contract."""
    return AudioPayloadMaterializeStage.from_payload_config(
        payload_spec=payload_spec,
        payload_config=payload_config,
        run_id=run_id,
    )


@dataclass
class PayloadResolvingStage(PayloadAwareStageMixin, ProcessingStage[Any, Any]):
    """Hydrate structural AudioTask leaves only while a wrapped stage executes."""

    wrapped_stage: ProcessingStage
    waveform_ref_key: str = "waveform_ref"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    num_samples_key: str = "num_samples"
    name: str = field(init=False)
    resources: Resources = field(init=False)
    batch_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.wrapped_stage.name
        self.resources = self.wrapped_stage.resources
        self.batch_size = self.wrapped_stage.batch_size
        self.runtime_env = self.wrapped_stage.runtime_env
        self.is_source_stage = self.wrapped_stage.is_source_stage
        self.is_sink_stage = self.wrapped_stage.is_sink_stage
        self.is_resumable = self.wrapped_stage.is_resumable
        stage_id = getattr(self.wrapped_stage, "_curator_stage_id", None)
        if stage_id is not None:
            self._curator_stage_id = stage_id

    def __getattr__(self, name: str) -> Any:
        wrapped_stage = self.__dict__.get("wrapped_stage")
        if wrapped_stage is None:
            raise AttributeError(name)
        return getattr(wrapped_stage, name)

    def inputs(self) -> tuple[list[str], list[str]]:
        return self.wrapped_stage.inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self.wrapped_stage.outputs()

    def num_workers(self) -> int | None:
        return self.wrapped_stage.num_workers()

    def ray_stage_spec(self) -> dict[str, Any]:
        return self.wrapped_stage.ray_stage_spec()

    def xenna_stage_spec(self) -> dict[str, Any]:
        return self.wrapped_stage.xenna_stage_spec()

    def setup_on_node(self, node_info: NodeInfo | None = None, worker_metadata: WorkerMetadata | None = None) -> None:
        self.wrapped_stage.setup_on_node(node_info, worker_metadata)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        self.wrapped_stage.setup(worker_metadata)

    def teardown(self) -> None:
        self.wrapped_stage.teardown()

    def process(self, task: Any) -> Any:
        results = self.process_batch([task])
        if len(results) != 1:
            msg = f"Wrapped stage {self.name!r} produced {len(results)} outputs for one input"
            raise RuntimeError(msg)
        return results[0]

    def process_batch(self, tasks: list[Any]) -> list[Any]:
        audio_leaves = [leaf for task in tasks for leaf in iter_payload_items(task) if isinstance(leaf, AudioTask)]
        inserted = self.resolve_payload_refs_for_batch(audio_leaves)
        hydrated = [map_payload_items(task, lambda leaf: leaf) for task in tasks]
        try:
            results = self.wrapped_stage.process_batch(hydrated)
        except Exception:
            self.drop_resolved_payloads(inserted)
            raise
        self.drop_resolved_payloads(inserted)
        self._drop_output_payloads(results)
        return results

    def _drop_output_payloads(self, values: list[Any]) -> None:
        for value in values:
            for leaf in iter_payload_items(value):
                if isinstance(leaf, AudioTask) and isinstance(leaf.data.get(self.waveform_ref_key), PayloadRef):
                    leaf.data.pop(self.waveform_key, None)

    def _consume_custom_metrics(self) -> dict[str, float]:
        return self.wrapped_stage._consume_custom_metrics()


@dataclass
class PayloadReleaseStage(ProcessingStage[Task, Task]):
    """Release nested payload refs while preserving task/envelope structure."""

    name: str = "PayloadReleaseStage"
    payload_ref_key: str = "waveform_ref"
    payload_key: str = "waveform"
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.1))
    batch_size: int = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: Task) -> Task:
        refs = task_payload_refs(task)
        released: set[tuple[str | None, str, str]] = set()
        for payload_ref in refs:
            key = (payload_ref.actor_namespace, payload_ref.store_actor_name, payload_ref.payload_id)
            if key not in released:
                release_payload_ref(payload_ref)
                released.add(key)
        if isinstance(task.data, dict):
            cleaned = strip_payload_refs(task.data)
            task.data.clear()
            task.data.update(cleaned)
            task.data.pop(self.payload_ref_key, None)
            task.data.pop(self.payload_key, None)
        return task

    def process_batch(self, tasks: list[Any]) -> list[Any]:
        def release(item: Any) -> Any:
            if not isinstance(item, Task):
                msg = f"{type(self).__name__} envelope items must be Task, got {type(item).__name__}"
                raise TypeError(msg)
            return self.process(item)

        return [map_payload_items(task, release) for task in tasks]
