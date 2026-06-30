# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN401, BLE001

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from nemo_curator.tasks import Task

PAYLOAD_RESERVATION_LEASE_TTL_S = 60 * 60.0


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
    actor_namespace: str | None = None


def _get_named_actor(name: str, namespace: str | None = None) -> Any:
    import ray

    if namespace:
        return ray.get_actor(name, namespace=namespace)
    return ray.get_actor(name)


def resolve_payload_refs_batched(
    payload_refs: Sequence[PayloadRef],
) -> list[Any]:
    """Resolve refs in input order using one actor-grouped request wave."""
    refs = list(payload_refs)
    if not refs:
        return []

    unique_refs = _unique_payload_refs(refs)
    resolved_by_key: dict[tuple[str | None, str, str], Any] = {}
    store_groups = _group_payload_refs(unique_refs, actor_name=lambda ref: ref.store_actor_name)
    calls = []
    grouped = []
    for (actor_name, namespace), grouped_refs in store_groups.items():
        actor = _get_named_actor(actor_name, namespace)
        calls.append(actor.get_many.remote([ref.payload_id for ref in grouped_refs]))
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
    """Recursively find refs for malformed-row and terminal cleanup fallbacks."""
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
    """Return recursively nested refs only for defensive cleanup paths."""
    return iter_payload_refs(task.data) if isinstance(task.data, dict) else []
