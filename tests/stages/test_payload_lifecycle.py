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

import sys
import time
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

from nemo_curator.stages import payload_lifecycle as lifecycle
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.payload_lifecycle import (
    AudioPayloadMaterializeStage,
    PayloadRef,
    PayloadReleaseStage,
    _PayloadAdmissionState,
    _PayloadStoreState,
    task_payload_refs,
)
from nemo_curator.tasks import AudioTask


class _RemoteMethod:
    def __init__(self, fn: Callable[..., object]) -> None:
        self._fn = fn

    def remote(self, *args: object, **kwargs: object) -> object:
        return self._fn(*args, **kwargs)


class _FakeReader:
    def __init__(self) -> None:
        self.calls = 0

    def process(self, task: AudioTask) -> AudioTask:
        self.calls += 1
        samples = int(float(task.data["duration"]) * 16_000)
        task.data["waveform"] = torch.zeros(1, samples, dtype=torch.float32)
        task.data["sample_rate"] = 16_000
        task.data["num_samples"] = samples
        return task

    def setup(self, *_args: object, **_kwargs: object) -> None:
        return None

    def setup_on_node(self, *_args: object, **_kwargs: object) -> None:
        return None

    def teardown(self) -> None:
        return None


class _FakeSkipReader(_FakeReader):
    def process(self, task: AudioTask) -> AudioTask:
        self.calls += 1
        task.data["waveform"] = torch.empty(1, 0, dtype=torch.float32)
        task.data["sample_rate"] = 16_000
        task.data["num_samples"] = 0
        task.data["duration"] = 0.0
        task.data["_skip_me"] = "audio_read_error"
        task.data["audio_read_error"] = "RuntimeError: decode lost"
        return task


class _RequiresTextStage(ProcessingStage[AudioTask, AudioTask]):
    name = "RequiresTextStage"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["text"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: AudioTask) -> AudioTask:
        return task


@pytest.fixture(autouse=True)
def _fake_ray_get(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lifecycle, "_ray_get", lambda obj: obj)


class _FakeAdmission:
    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.used = 0
        self.acquire_calls: list[tuple[str, int]] = []
        self.acquire_ttls: list[float] = []
        self.resize_calls: list[tuple[str, int]] = []
        self.resize_ttls: list[float] = []
        self.heartbeat_ttls: list[float] = []
        self.release_calls: list[tuple[str, int | None]] = []
        self.register_node = _RemoteMethod(lambda *_args: None)
        self.try_acquire = _RemoteMethod(self._try_acquire)
        self.resize = _RemoteMethod(self._resize)
        self.release = _RemoteMethod(self._release)
        self.heartbeat = _RemoteMethod(self._heartbeat)
        self.snapshot = _RemoteMethod(self._snapshot)

    def _try_acquire(self, _node_id: str, owner_id: str, amount: int, _ttl: float) -> bool:
        self.acquire_calls.append((owner_id, amount))
        self.acquire_ttls.append(_ttl)
        if amount > self.budget or self.used + amount > self.budget:
            return False
        self.used += amount
        return True

    def _resize(self, _node_id: str, owner_id: str, amount: int, _ttl: float) -> bool:
        self.resize_calls.append((owner_id, amount))
        self.resize_ttls.append(_ttl)
        if amount > self.budget:
            return False
        self.used = amount
        return True

    def _release(self, _node_id: str, owner_id: str, amount: int | None = None) -> None:
        self.release_calls.append((owner_id, amount))
        if amount is None:
            self.used = 0
        else:
            self.used = max(0, self.used - amount)

    def _heartbeat(self, _node_id: str, _owner_id: str, ttl: float) -> bool:
        self.heartbeat_ttls.append(ttl)
        return True

    def _snapshot(self) -> dict[str, int]:
        return {"cluster_used": self.used, "cluster_budget": self.budget}


class _FakeStore:
    def __init__(self) -> None:
        self.payloads: dict[str, torch.Tensor] = {}
        self.put_ttls: list[float] = []
        self.put = _RemoteMethod(self._put)
        self.get = _RemoteMethod(self._get)
        self.pin = _RemoteMethod(self._pin)
        self.release = _RemoteMethod(self._release)

    def _put(self, payload_id: str, waveform: torch.Tensor, _amount: int, _ttl: float) -> None:
        self.put_ttls.append(_ttl)
        self.payloads[payload_id] = waveform

    def _get(self, payload_id: str, _ttl: float) -> torch.Tensor:
        return self.payloads[payload_id]

    def _pin(self, payload_id: str, _ttl: float) -> bool:
        return payload_id in self.payloads

    def _release(self, payload_id: str) -> int:
        waveform = self.payloads.pop(payload_id, None)
        if waveform is None:
            return 0
        return int(waveform.element_size() * waveform.nelement())


def _stage_with_fakes(
    *, budget: int = 1_000_000
) -> tuple[AudioPayloadMaterializeStage, _FakeReader, _FakeAdmission, _FakeStore]:
    stage = AudioPayloadMaterializeStage(
        max_node_payload_bytes=budget,
        admission_poll_interval_s=0.001,
    )
    reader = _FakeReader()
    admission = _FakeAdmission(budget)
    store = _FakeStore()
    stage._reader = reader
    stage._node_id = "node-1"
    stage._node_budget_bytes = budget
    stage._store_actor_name = "payload-store-node-1"
    stage._admission = admission
    stage._store = store
    return stage, reader, admission, store


def test_audio_payload_materialize_constructs_reader_with_configured_keys() -> None:
    stage = AudioPayloadMaterializeStage(
        waveform_key="custom_waveform",
        sample_rate_key="custom_sample_rate",
        num_samples_key="custom_num_samples",
    )
    stage._node_id = "node-1"
    stage._admission = object()
    stage._store = object()

    stage._ensure_ready()

    assert stage._reader.waveform_key == "custom_waveform"
    assert stage._reader.sample_rate_key == "custom_sample_rate"
    assert stage._reader.num_samples_key == "custom_num_samples"


def test_audio_payload_materialize_requires_duration_before_decode() -> None:
    stage, reader, admission, _store = _stage_with_fakes()

    with pytest.raises(ValueError, match="requires 'duration'"):
        stage.process(AudioTask(data={"audio_filepath": "s3://bucket/audio.wav"}))

    assert reader.calls == 0
    assert admission.acquire_calls == []


def test_audio_payload_materialize_process_batch_validates_required_inputs() -> None:
    stage, reader, admission, _store = _stage_with_fakes()

    with pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([AudioTask(data={"duration": 0.5})])

    assert reader.calls == 0
    assert admission.acquire_calls == []


def test_audio_payload_materialize_stores_waveform_by_ref_and_removes_payload() -> None:
    stage, reader, admission, store = _stage_with_fakes()
    task = AudioTask(data={"audio_filepath": "s3://bucket/audio.wav", "duration": 0.5})

    output = stage.process(task)

    assert reader.calls == 1
    assert "waveform" not in output.data
    payload_ref = output.data["waveform_ref"]
    assert isinstance(payload_ref, PayloadRef)
    assert payload_ref.amount_bytes == 32_000
    assert payload_ref.payload_id in store.payloads
    assert admission.acquire_calls == [(payload_ref.payload_id, 32_000)]
    assert admission.acquire_ttls == [stage.lease_ttl_s]
    assert admission.resize_calls == []
    assert admission.heartbeat_ttls == [stage.materialized_lease_ttl_s]
    assert store.put_ttls == [stage.materialized_lease_ttl_s]
    assert output.data["_curator_payload_estimated_bytes"] == 32_000
    assert output.data["_curator_payload_bytes"] == 32_000
    assert "_curator_payload_producer_node_id" not in output.data


def test_audio_payload_materialize_passes_reader_skip_without_payload_ref() -> None:
    stage, _reader, admission, store = _stage_with_fakes()
    stage.skip_on_read_error = True
    stage._reader = _FakeSkipReader()
    task = AudioTask(data={"audio_filepath": "/local/audio.wav", "duration": 0.5})

    output = stage.process(task)

    assert output.data["_skip_me"] == "audio_read_error"
    assert output.data["audio_read_error"] == "RuntimeError: decode lost"
    assert output.data["num_samples"] == 0
    assert "waveform" not in output.data
    assert "waveform_ref" not in output.data
    assert store.payloads == {}
    assert admission.acquire_calls
    assert admission.release_calls


def test_audio_payload_ref_carries_ray_namespace() -> None:
    stage, _reader, _admission, _store = _stage_with_fakes()
    stage._actor_namespace = "payload-ns"
    task = AudioTask(data={"audio_filepath": "s3://bucket/audio.wav", "duration": 0.5})

    payload_ref = stage.process(task).data["waveform_ref"]

    assert payload_ref.actor_namespace == "payload-ns"


def test_payload_actor_creation_is_detached_and_namespaced(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeRemoteActor:
        def options(self, **options: object) -> "_FakeRemoteActor":
            captured["options"] = options
            return self

        def remote(self, **kwargs: object) -> str:
            captured["kwargs"] = kwargs
            return "actor-handle"

    fake_ray = SimpleNamespace(
        get_actor=lambda _name, **_kwargs: (_ for _ in ()).throw(ValueError("missing actor")),
        remote=lambda _actor_cls: _FakeRemoteActor(),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    actor = lifecycle._get_named_actor_or_create(object, "payload-admission", namespace="payload-ns", value=123)

    assert actor == "actor-handle"
    assert captured["options"] == {
        "name": "payload-admission",
        "get_if_exists": True,
        "lifetime": "detached",
        "namespace": "payload-ns",
    }
    assert captured["kwargs"] == {"value": 123}


def test_audio_payload_materialize_cleanup_kills_run_scoped_actors(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[tuple[str, str | None]] = []
    monkeypatch.setattr(lifecycle, "_active_ray_node_ids", lambda: ["node-a", "node/b"])
    monkeypatch.setattr(lifecycle, "_current_ray_namespace", lambda: "payload-ns")
    monkeypatch.setattr(
        lifecycle, "_kill_named_actor", lambda name, namespace=None: killed.append((name, namespace)) or True
    )

    stage = AudioPayloadMaterializeStage(
        admission_actor_name="admission",
        store_actor_prefix="store",
        run_id="run/id",
    )

    stage.cleanup_run_resources()

    assert killed == [
        ("admission_run_id", "payload-ns"),
        ("store_run_id_node-a", "payload-ns"),
        ("store_run_id_node_b", "payload-ns"),
    ]


def test_audio_payload_materialize_releases_tokens_when_actual_size_exceeds_budget() -> None:
    stage, _reader, admission, store = _stage_with_fakes(budget=16_000)
    task = AudioTask(data={"audio_filepath": "s3://bucket/audio.wav", "duration": 0.5})
    task.data["duration"] = 0.5

    # The initial estimate is 32k, but the fake reader returns exactly 32k, so
    # force the estimate small enough that resize must reject the actual payload.
    stage.sample_width_bytes = 1

    with pytest.raises(RuntimeError, match="Insufficient payload memory budget"):
        stage.process(task)

    assert store.payloads == {}
    assert admission.release_calls


def test_payload_release_stage_drops_ref_and_payload_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    released: list[str] = []
    ref = PayloadRef(
        payload_id="payload-1",
        owner_node_id="node-1",
        store_actor_name="store",
        admission_actor_name="admission",
        amount_bytes=123,
        sample_rate=16_000,
        num_samples=42,
    )
    monkeypatch.setattr(lifecycle, "release_payload_ref", lambda payload_ref: released.append(payload_ref.payload_id))
    task = AudioTask(
        data={
            "waveform_ref": ref,
            "waveform": torch.zeros(1, 42),
            "_curator_payload_estimated_bytes": 123,
            "_curator_payload_bytes": 123,
            "_curator_payload_producer_node_id": "node-a",
        }
    )

    output = PayloadReleaseStage().process(task)

    assert released == ["payload-1"]
    assert "waveform_ref" not in output.data
    assert "waveform" not in output.data
    assert not any(key.startswith("_curator_payload_") for key in output.data)


def test_payload_release_stage_preserves_data_attr_access_for_downstream_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[str] = []
    ref = PayloadRef(
        payload_id="payload-1",
        owner_node_id="node-1",
        store_actor_name="store",
        admission_actor_name="admission",
        amount_bytes=123,
        sample_rate=16_000,
        num_samples=42,
    )
    monkeypatch.setattr(lifecycle, "release_payload_ref", lambda payload_ref: released.append(payload_ref.payload_id))
    task = AudioTask(data={"text": "keep me", "waveform_ref": ref, "nested": {"refs": [ref]}})
    data_before_release = task.data

    output = PayloadReleaseStage().process(task)

    assert released == ["payload-1"]
    assert output.data is data_before_release
    assert output.data.text == "keep me"
    assert "waveform_ref" not in output.data
    assert output.data["nested"] == {"refs": []}
    assert _RequiresTextStage().validate_input(output)


def test_payload_release_stage_noops_for_rows_without_payload_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lifecycle, "release_payload_ref", lambda _payload_ref: pytest.fail("unexpected release"))
    task = AudioTask(
        data={
            "audio_filepath": "/local/audio.wav",
            "_skip_me": "read_error",
            "_curator_payload_estimated_bytes": 123,
            "_curator_payload_bytes": 123,
        }
    )

    output = PayloadReleaseStage().process(task)

    assert output.data == {"audio_filepath": "/local/audio.wav", "_skip_me": "read_error"}


def test_payload_release_stage_releases_all_nested_payload_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    released: list[str] = []
    ref_a = PayloadRef(
        payload_id="payload-a",
        owner_node_id="node-1",
        store_actor_name="store",
        admission_actor_name="admission",
        amount_bytes=123,
        sample_rate=16_000,
        num_samples=42,
    )
    ref_b = PayloadRef(
        payload_id="payload-b",
        owner_node_id="node-1",
        store_actor_name="store",
        admission_actor_name="admission",
        amount_bytes=456,
        sample_rate=16_000,
        num_samples=84,
    )
    monkeypatch.setattr(lifecycle, "release_payload_ref", lambda payload_ref: released.append(payload_ref.payload_id))
    task = AudioTask(data={"waveform_ref": ref_a, "extra_refs": [ref_b, ref_a]})

    output = PayloadReleaseStage().process(task)

    assert sorted(released) == ["payload-a", "payload-b"]
    assert "waveform_ref" not in output.data
    assert output.data["extra_refs"] == []
    assert task_payload_refs(output) == []


def test_payload_lease_keeper_heartbeats_until_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    heartbeats: list[str] = []
    ref = PayloadRef(
        payload_id="payload-a",
        owner_node_id="node-1",
        store_actor_name="store",
        admission_actor_name="admission",
        amount_bytes=123,
        sample_rate=16_000,
        num_samples=42,
        lease_ttl_s=0.2,
    )
    monkeypatch.setattr(
        lifecycle,
        "heartbeat_payload_refs_batched",
        lambda refs: heartbeats.extend(ref.payload_id for ref in refs),
    )

    keeper = lifecycle._PayloadLeaseKeeper([ref], interval_s=0.01)
    keeper.start()
    time.sleep(0.035)
    keeper.stop()
    count_after_stop = len(heartbeats)
    time.sleep(0.03)

    assert count_after_stop >= 2
    assert len(heartbeats) == count_after_stop


def test_payload_admission_resize_and_release() -> None:
    admission = _PayloadAdmissionState(default_node_budget_bytes=100)
    admission.register_node("node-a", 100)

    assert admission.try_acquire("node-a", "payload-1", 40)
    assert not admission.resize("node-a", "payload-1", 120)
    assert admission.resize("node-a", "payload-1", 80)
    snapshot = admission.snapshot()
    assert snapshot["node_used"]["node-a"] == 80

    admission.release("node-a", "payload-1")
    assert admission.snapshot()["node_used"]["node-a"] == 0


def test_payload_admission_heartbeat_many_preserves_request_order() -> None:
    admission = _PayloadAdmissionState(default_node_budget_bytes=100)
    admission.register_node("node-a", 100)
    assert admission.try_acquire("node-a", "payload-1", 40)
    reap_calls = 0
    original_reap = admission._reap_expired

    def count_reap() -> None:
        nonlocal reap_calls
        reap_calls += 1
        original_reap()

    admission._reap_expired = count_reap  # type: ignore[method-assign]

    assert admission.heartbeat_many(
        [
            ("node-a", "payload-1", 5.0),
            ("node-a", "missing", 5.0),
        ]
    ) == [True, False]
    assert reap_calls == 1


def test_payload_store_pin_and_get_many_preserve_request_order() -> None:
    store = _PayloadStoreState()
    payload_a = torch.tensor([1.0])
    payload_b = torch.tensor([2.0])
    store.put("payload-a", payload_a, 4)
    store.put("payload-b", payload_b, 4)
    reap_calls = 0
    original_reap = store._reap_expired

    def count_reap() -> None:
        nonlocal reap_calls
        reap_calls += 1
        original_reap()

    store._reap_expired = count_reap  # type: ignore[method-assign]

    assert store.pin_many([("payload-b", 5.0), ("missing", 5.0)]) == [True, False]
    assert reap_calls == 1
    assert store.get_many([("payload-b", 5.0), ("payload-a", 5.0)]) == [payload_b, payload_a]
    assert reap_calls == 2


def test_payload_admission_enforces_cluster_budget() -> None:
    admission = _PayloadAdmissionState(default_node_budget_bytes=100, default_cluster_budget_bytes=150)
    admission.register_node("node-a", 100)
    admission.register_node("node-b", 100)

    assert admission.try_acquire("node-a", "payload-1", 100)
    assert not admission.try_acquire("node-b", "payload-2", 100)
    assert admission.try_acquire("node-b", "payload-3", 50)

    snapshot = admission.snapshot()
    assert snapshot["cluster_budget"] == 150
    assert snapshot["cluster_used"] == 150


def test_payload_materialize_rejects_invalid_byte_limit_string() -> None:
    with pytest.raises(ValueError, match="max_node_payload_bytes"):
        AudioPayloadMaterializeStage(max_node_payload_bytes="definitely-not-bytes")._ensure_ready()


def test_payload_materialize_fails_fast_when_single_row_exceeds_cluster_budget() -> None:
    stage = AudioPayloadMaterializeStage(max_node_payload_bytes=1_000, max_cluster_payload_bytes=10)
    stage._node_budget_bytes = 1_000
    stage._cluster_budget_bytes = 10

    with pytest.raises(RuntimeError, match="cluster payload budget"):
        stage._acquire("payload-1", 20)


def test_payload_materialize_times_out_when_admission_budget_never_frees() -> None:
    stage, _reader, admission, _store = _stage_with_fakes(budget=1_000)
    admission.budget = 0
    stage.admission_poll_interval_s = 0.0001
    stage.admission_wait_timeout_s = 0.001

    with pytest.raises(RuntimeError, match="Timed out waiting for payload admission") as exc_info:
        stage._acquire("payload-1", 100)

    assert "cluster_used" in str(exc_info.value)
    assert admission.acquire_ttls
    assert set(admission.acquire_ttls) == {stage.lease_ttl_s}


def test_task_payload_refs_finds_nested_refs() -> None:
    ref = PayloadRef(
        payload_id="payload-1",
        owner_node_id="node-1",
        store_actor_name="store",
        admission_actor_name="admission",
        amount_bytes=123,
        sample_rate=16_000,
        num_samples=42,
    )
    task = AudioTask(data={"nested": {"payloads": [ref]}})

    assert task_payload_refs(task) == [ref]


def test_payload_admission_reaps_expired_leases(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    monkeypatch.setattr(lifecycle.time, "monotonic", lambda: now[0])
    admission = _PayloadAdmissionState(default_node_budget_bytes=100)
    admission.register_node("node-a", 100)

    assert admission.try_acquire("node-a", "payload-1", 100, lease_ttl_s=1.0)
    now[0] = 1.1
    assert admission.try_acquire("node-a", "payload-2", 100, lease_ttl_s=1.0)

    snapshot = admission.snapshot()
    assert snapshot["node_used"]["node-a"] == 100
    assert snapshot["lease_count"] == 1


def test_audio_payload_materialize_rejects_non_positive_materialized_lease() -> None:
    with pytest.raises(ValueError, match="materialized_lease_ttl_s must be positive"):
        AudioPayloadMaterializeStage(materialized_lease_ttl_s=0.0)


def test_payload_admission_explicit_release_lease_survives_without_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    monkeypatch.setattr(lifecycle.time, "monotonic", lambda: now[0])
    admission = _PayloadAdmissionState(default_node_budget_bytes=100)
    admission.register_node("node-a", 100)

    assert admission.try_acquire("node-a", "payload-1", 100, lease_ttl_s=0.0)
    now[0] = 10_000.0

    snapshot = admission.snapshot()
    assert snapshot["node_used"]["node-a"] == 100
    assert snapshot["lease_count"] == 1
    assert admission.heartbeat("node-a", "payload-1", lease_ttl_s=1.0)
    now[0] = 20_000.0
    assert admission.snapshot()["lease_count"] == 1


def test_payload_store_explicit_release_payload_survives_without_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    monkeypatch.setattr(lifecycle.time, "monotonic", lambda: now[0])
    store = _PayloadStoreState(default_lease_ttl_s=1.0)
    waveform = torch.zeros(1, 10)

    store.put("payload-1", waveform, 40, lease_ttl_s=0.0)
    now[0] = 10_000.0

    assert store.snapshot()["payload_count"] == 1
    assert store.pin("payload-1", lease_ttl_s=1.0)
    now[0] = 20_000.0
    assert store.get("payload-1", lease_ttl_s=1.0) is waveform
    assert store.release("payload-1") == 40
