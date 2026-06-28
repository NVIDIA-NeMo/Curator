# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pytest

from nemo_curator.pipeline import payload_refs
from nemo_curator.pipeline.payload_refs import PayloadRef


class _RemoteMethod:
    def __init__(self, function: Callable[..., object]) -> None:
        self._function = function

    def remote(self, *args: object, **kwargs: object) -> object:
        return self._function(*args, **kwargs)


class _AdmissionActor:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str, float | None]]] = []
        self.heartbeat_many = _RemoteMethod(self._heartbeat_many)

    def _heartbeat_many(self, requests: list[tuple[str, str, float | None]]) -> list[bool]:
        self.calls.append(requests)
        return [True] * len(requests)


class _StoreActor:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values
        self.pin_calls: list[list[tuple[str, float | None]]] = []
        self.get_calls: list[list[tuple[str, float | None]]] = []
        self.pin_many = _RemoteMethod(self._pin_many)
        self.get_many = _RemoteMethod(self._get_many)

    def _pin_many(self, requests: list[tuple[str, float | None]]) -> list[bool]:
        self.pin_calls.append(requests)
        return [payload_id in self.values for payload_id, _ttl in requests]

    def _get_many(self, requests: list[tuple[str, float | None]]) -> list[object]:
        self.get_calls.append(requests)
        return [self.values[payload_id] for payload_id, _ttl in requests]


def _ref(payload_id: str, *, amount_bytes: int = 6) -> PayloadRef:
    return PayloadRef(
        payload_id=payload_id,
        owner_node_id="node-a",
        store_actor_name="store",
        admission_actor_name="admission",
        amount_bytes=amount_bytes,
        sample_rate=16_000,
        num_samples=100,
    )


def test_resolve_payload_refs_batched_is_byte_bounded_and_ordered(monkeypatch: pytest.MonkeyPatch) -> None:
    admission = _AdmissionActor()
    store = _StoreActor({"a": "payload-a", "b": "payload-b"})
    actors = {"admission": admission, "store": store}
    monkeypatch.setattr(payload_refs, "_get_named_actor", lambda name, _namespace=None: actors[name])
    monkeypatch.setattr(payload_refs, "_ray_get", lambda value: value)

    resolved = payload_refs.resolve_payload_refs_batched(
        [_ref("b"), _ref("a"), _ref("b")],
        max_batch_bytes=10,
    )

    assert resolved == ["payload-b", "payload-a", "payload-b"]
    assert [[request[1] for request in call] for call in admission.calls] == [["b"], ["a"]]
    assert [[request[0] for request in call] for call in store.pin_calls] == [["b"], ["a"]]
    assert [[request[0] for request in call] for call in store.get_calls] == [["b"], ["a"]]


def test_resolve_payload_refs_batched_rejects_missing_store_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    admission = _AdmissionActor()
    store = _StoreActor({})
    actors = {"admission": admission, "store": store}
    monkeypatch.setattr(payload_refs, "_get_named_actor", lambda name, _namespace=None: actors[name])
    monkeypatch.setattr(payload_refs, "_ray_get", lambda value: value)

    with pytest.raises(KeyError, match="no longer present"):
        payload_refs.resolve_payload_refs_batched([_ref("missing")])


def test_resolve_payload_refs_batched_rejects_boolean_byte_limit() -> None:
    with pytest.raises(ValueError, match="max_batch_bytes must be positive"):
        payload_refs.resolve_payload_refs_batched([_ref("payload")], max_batch_bytes=True)
