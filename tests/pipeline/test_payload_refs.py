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


class _StoreActor:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values
        self.get_calls: list[list[str]] = []
        self.get_many = _RemoteMethod(self._get_many)

    def _get_many(self, payload_ids: list[str]) -> list[object]:
        self.get_calls.append(payload_ids)
        missing = next((payload_id for payload_id in payload_ids if payload_id not in self.values), None)
        if missing is not None:
            msg = f"Payload {missing} is no longer present"
            raise KeyError(msg)
        return [self.values[payload_id] for payload_id in payload_ids]


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


def test_resolve_payload_refs_batched_deduplicates_and_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _StoreActor({"a": "payload-a", "b": "payload-b"})
    actors = {"store": store}
    monkeypatch.setattr(payload_refs, "_get_named_actor", lambda name, _namespace=None: actors[name])
    monkeypatch.setattr(payload_refs, "_ray_get", lambda value: value)

    resolved = payload_refs.resolve_payload_refs_batched([_ref("b"), _ref("a"), _ref("b")])

    assert resolved == ["payload-b", "payload-a", "payload-b"]
    assert store.get_calls == [["b", "a"]]


def test_resolve_payload_refs_batched_rejects_missing_store_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _StoreActor({})
    actors = {"store": store}
    monkeypatch.setattr(payload_refs, "_get_named_actor", lambda name, _namespace=None: actors[name])
    monkeypatch.setattr(payload_refs, "_ray_get", lambda value: value)

    with pytest.raises(KeyError, match="no longer present"):
        payload_refs.resolve_payload_refs_batched([_ref("missing")])
