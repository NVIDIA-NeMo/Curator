# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN401, BLE001, EM102

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from nemo_curator.tasks import Task

_DEFAULT_LEASE_TTL_S = 3600.0


def _ray_get(obj: Any) -> Any:
    import ray

    return ray.get(obj)


@dataclass(frozen=True)
class PayloadRef:
    payload_id: str
    owner_node_id: str
    store_actor_name: str
    admission_actor_name: str
    amount_bytes: int
    sample_rate: int
    num_samples: int
    dtype: str = "float32"
    lease_ttl_s: float = _DEFAULT_LEASE_TTL_S
    actor_namespace: str | None = None


def _get_named_actor(name: str, namespace: str | None = None) -> Any:
    import ray

    if namespace:
        return ray.get_actor(name, namespace=namespace)
    return ray.get_actor(name)


def resolve_payload_ref(payload_ref: PayloadRef) -> Any:
    heartbeat_payload_ref(payload_ref)
    store = _get_named_actor(payload_ref.store_actor_name, payload_ref.actor_namespace)
    return _ray_get(store.get.remote(payload_ref.payload_id, payload_ref.lease_ttl_s))


def heartbeat_payload_ref(payload_ref: PayloadRef) -> None:
    admission = _get_named_actor(payload_ref.admission_actor_name, payload_ref.actor_namespace)
    if not _ray_get(
        admission.heartbeat.remote(
            payload_ref.owner_node_id,
            payload_ref.payload_id,
            payload_ref.lease_ttl_s,
        )
    ):
        raise KeyError(
            f"Payload admission lease {payload_ref.payload_id} is no longer present in "
            f"{payload_ref.admission_actor_name}"
        )
    store = _get_named_actor(payload_ref.store_actor_name, payload_ref.actor_namespace)
    if not _ray_get(store.pin.remote(payload_ref.payload_id, payload_ref.lease_ttl_s)):
        raise KeyError(f"Payload {payload_ref.payload_id} is no longer present in {payload_ref.store_actor_name}")


def heartbeat_payload_refs_batched(payload_refs: Sequence[PayloadRef]) -> None:
    """Refresh payload leases with one RPC per admission/store actor.

    The singular :func:`heartbeat_payload_ref` contract remains unchanged for
    existing callers. This opt-in batched path is used by payload-aware stages
    that know their actors provide ``heartbeat_many`` and ``pin_many``.
    """
    refs = _unique_payload_refs(payload_refs)
    if not refs:
        return

    admission_groups = _group_payload_refs(refs, actor_name=lambda ref: ref.admission_actor_name)
    admission_calls = []
    admission_group_refs = []
    for (actor_name, namespace), grouped_refs in admission_groups.items():
        actor = _get_named_actor(actor_name, namespace)
        admission_calls.append(
            actor.heartbeat_many.remote([(ref.owner_node_id, ref.payload_id, ref.lease_ttl_s) for ref in grouped_refs])
        )
        admission_group_refs.append(grouped_refs)
    for grouped_refs, results in zip(admission_group_refs, _ray_get(admission_calls), strict=True):
        for ref, present in zip(grouped_refs, results, strict=True):
            if not present:
                raise KeyError(
                    f"Payload admission lease {ref.payload_id} is no longer present in {ref.admission_actor_name}"
                )

    store_groups = _group_payload_refs(refs, actor_name=lambda ref: ref.store_actor_name)
    store_calls = []
    store_group_refs = []
    for (actor_name, namespace), grouped_refs in store_groups.items():
        actor = _get_named_actor(actor_name, namespace)
        store_calls.append(actor.pin_many.remote([(ref.payload_id, ref.lease_ttl_s) for ref in grouped_refs]))
        store_group_refs.append(grouped_refs)
    for grouped_refs, results in zip(store_group_refs, _ray_get(store_calls), strict=True):
        for ref, present in zip(grouped_refs, results, strict=True):
            if not present:
                raise KeyError(f"Payload {ref.payload_id} is no longer present in {ref.store_actor_name}")


def resolve_payload_refs_batched(
    payload_refs: Sequence[PayloadRef],
    *,
    max_batch_bytes: int | None = None,
) -> list[Any]:
    """Resolve refs in input order using byte-bounded, actor-grouped RPCs."""
    refs = list(payload_refs)
    if not refs:
        return []
    if max_batch_bytes is not None and (isinstance(max_batch_bytes, bool) or max_batch_bytes <= 0):
        msg = "max_batch_bytes must be positive when set"
        raise ValueError(msg)

    resolved_by_key: dict[tuple[str | None, str, str], Any] = {}
    for batch in _byte_bounded_payload_batches(_unique_payload_refs(refs), max_batch_bytes):
        heartbeat_payload_refs_batched(batch)
        store_groups = _group_payload_refs(batch, actor_name=lambda ref: ref.store_actor_name)
        calls = []
        grouped = []
        for (actor_name, namespace), grouped_refs in store_groups.items():
            actor = _get_named_actor(actor_name, namespace)
            calls.append(actor.get_many.remote([(ref.payload_id, ref.lease_ttl_s) for ref in grouped_refs]))
            grouped.append(grouped_refs)
        for grouped_refs, payloads in zip(grouped, _ray_get(calls), strict=True):
            for ref, payload in zip(grouped_refs, payloads, strict=True):
                resolved_by_key[_payload_ref_key(ref)] = payload
    return [resolved_by_key[_payload_ref_key(ref)] for ref in refs]


def _payload_ref_key(payload_ref: PayloadRef) -> tuple[str | None, str, str]:
    return payload_ref.actor_namespace, payload_ref.store_actor_name, payload_ref.payload_id


def _unique_payload_refs(payload_refs: Sequence[PayloadRef]) -> list[PayloadRef]:
    unique: dict[tuple[str | None, str, str], PayloadRef] = {}
    for payload_ref in payload_refs:
        unique.setdefault(_payload_ref_key(payload_ref), payload_ref)
    return list(unique.values())


def _group_payload_refs(
    payload_refs: Sequence[PayloadRef],
    *,
    actor_name: Callable[[PayloadRef], str],
) -> dict[tuple[str, str | None], list[PayloadRef]]:
    groups: dict[tuple[str, str | None], list[PayloadRef]] = defaultdict(list)
    for payload_ref in payload_refs:
        groups[(actor_name(payload_ref), payload_ref.actor_namespace)].append(payload_ref)
    return groups


def _byte_bounded_payload_batches(
    payload_refs: Sequence[PayloadRef],
    max_batch_bytes: int | None,
) -> list[list[PayloadRef]]:
    if max_batch_bytes is None:
        return [list(payload_refs)]
    batches: list[list[PayloadRef]] = []
    current: list[PayloadRef] = []
    current_bytes = 0
    for payload_ref in payload_refs:
        amount = max(0, int(payload_ref.amount_bytes))
        if current and current_bytes + amount > max_batch_bytes:
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(payload_ref)
        current_bytes += amount
    if current:
        batches.append(current)
    return batches


def release_payload_ref(payload_ref: PayloadRef) -> None:
    try:
        store = _get_named_actor(payload_ref.store_actor_name, payload_ref.actor_namespace)
        released_bytes = int(_ray_get(store.release.remote(payload_ref.payload_id)))
    except Exception:
        released_bytes = int(payload_ref.amount_bytes)
    if released_bytes <= 0:
        released_bytes = int(payload_ref.amount_bytes)
    try:
        admission = _get_named_actor(payload_ref.admission_actor_name, payload_ref.actor_namespace)
        _ray_get(
            admission.release.remote(
                payload_ref.owner_node_id,
                payload_ref.payload_id,
                released_bytes,
            )
        )
    except Exception:
        logger.debug("Failed to release payload admission tokens for {}", payload_ref.payload_id)


_DROP_PAYLOAD_REF = object()


def strip_payload_refs(value: Any) -> Any:
    stripped = _strip_payload_refs(value)
    if stripped is _DROP_PAYLOAD_REF:
        return None
    return stripped


def _strip_payload_refs(value: Any) -> Any:
    if isinstance(value, PayloadRef):
        return _DROP_PAYLOAD_REF
    if isinstance(value, dict):
        return _strip_payload_ref_dict(value)
    if isinstance(value, list):
        return _strip_payload_ref_list(value)
    if isinstance(value, tuple):
        return tuple(_strip_payload_ref_list(value))
    if isinstance(value, set):
        return set(_strip_payload_ref_list(value))
    return value


def _strip_payload_ref_dict(value: dict[Any, Any]) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key, item in value.items():
        stripped = _strip_payload_refs(item)
        if stripped is not _DROP_PAYLOAD_REF:
            result[key] = stripped
    return result


def _strip_payload_ref_list(value: Any) -> list[Any]:
    result: list[Any] = []
    for item in value:
        stripped = _strip_payload_refs(item)
        if stripped is not _DROP_PAYLOAD_REF:
            result.append(stripped)
    return result


def iter_payload_refs(value: Any) -> list[PayloadRef]:
    refs: list[PayloadRef] = []
    if isinstance(value, PayloadRef):
        refs.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            refs.extend(iter_payload_refs(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            refs.extend(iter_payload_refs(item))
    return refs


def task_payload_refs(task: Task) -> list[PayloadRef]:
    return iter_payload_refs(task.data) if isinstance(task.data, dict) else []
