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

from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch

from nemo_curator.pipeline.payload_refs import PayloadRef
from nemo_curator.stages import payload_lifecycle
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.payload_lifecycle import (
    AudioPayloadMaterializeStage,
    PayloadAdmission,
    PayloadReleaseStage,
    PayloadResolvingStage,
    PayloadStore,
)
from nemo_curator.tasks import AudioTask


@dataclass(frozen=True)
class _FakeEnvelope:
    items: tuple[AudioTask, ...]
    marker: str

    def with_items(self, items: list[Any]) -> "_FakeEnvelope":
        return replace(self, items=tuple(items))


@dataclass
class _FakeFastConformerConsumer(ProcessingStage[object, object]):
    name: str = "fastconformer_owner"
    batch_size: int = 4
    saw_waveform: bool = False

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: object) -> object:
        return task

    def process_batch(self, tasks: list[object]) -> list[object]:
        results: list[object] = []
        for envelope in tasks:
            assert isinstance(envelope, _FakeEnvelope)
            children = []
            for child in envelope.items:
                waveform = child.data["waveform"]
                self.saw_waveform = torch.is_tensor(waveform)
                child.data["pred_text"] = f"samples={waveform.shape[-1]}"
                children.append(child)
            results.append(envelope.with_items(children))
        return results


def _payload_ref(payload_id: str = "payload") -> PayloadRef:
    return PayloadRef(
        payload_id=payload_id,
        owner_node_id="node",
        store_actor_name="store",
        admission_actor_name="admission",
        amount_bytes=16,
        metadata={"sample_rate": 16_000, "num_samples": 4},
    )


def _raise_actor_missing(*_args: object, **_kwargs: object) -> None:
    raise ValueError


def test_payload_actors_are_created_detached(monkeypatch: pytest.MonkeyPatch) -> None:
    import ray

    captured: dict[str, object] = {}

    class _Bound:
        @staticmethod
        def remote(*_args: object) -> str:
            return "handle"

    class _ActorClass:
        @staticmethod
        def options(**options: object) -> type[_Bound]:
            captured.update(options)
            return _Bound

    monkeypatch.setattr(payload_lifecycle, "_get_named_actor", _raise_actor_missing)
    monkeypatch.setattr(ray, "remote", lambda _cls: _ActorClass)

    handle = payload_lifecycle._get_or_create_actor(PayloadStore, "store", namespace="ns")

    assert handle == "handle"
    assert captured["lifetime"] == "detached"
    assert captured["name"] == "store"
    assert captured["namespace"] == "ns"


def test_materialize_cleanup_kills_run_actors_without_worker_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[str] = []
    monkeypatch.setattr(payload_lifecycle, "_kill_named_actor", lambda name, _namespace=None: killed.append(name))
    monkeypatch.setattr(payload_lifecycle, "_active_ray_node_ids", lambda: ["node-a", "node-b"])
    monkeypatch.setattr(payload_lifecycle, "_current_ray_namespace", lambda: None)

    stage = AudioPayloadMaterializeStage(run_id="run/1")
    assert stage._manager is None
    stage.cleanup_run_resources()

    assert killed == [
        "curator_payload_admission_run_1",
        "curator_payload_store_run_1_node-a",
        "curator_payload_store_run_1_node-b",
    ]


def test_payload_admission_and_store_release_are_idempotent() -> None:
    admission = PayloadAdmission(node_budget_bytes=32)
    admission.register_node("node")
    assert admission.try_acquire("node", "payload", 16)
    assert not admission.try_acquire("node", "other", 24)
    assert admission.publish("node", "payload")

    store = PayloadStore()
    store.put("payload", b"1234", 4)
    assert store.get_many(["payload"]) == [b"1234"]
    assert store.release("payload") == 4
    assert store.release("payload") == 0

    admission.release("node", "payload")
    admission.release("node", "payload")
    assert admission.snapshot()["node_used"]["node"] == 0


def test_audio_materializer_preserves_payload_neutral_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    original = AudioTask(data={"audio_filepath": "a.wav", "duration": 1.0})
    envelope = _FakeEnvelope(items=(original,), marker="keep-me")
    stage = AudioPayloadMaterializeStage()

    def materialize(task: AudioTask) -> AudioTask:
        task.data["waveform_ref"] = _payload_ref()
        return task

    monkeypatch.setattr(stage, "process", materialize)
    result = stage.process_batch([envelope])

    assert result == [_FakeEnvelope(items=(original,), marker="keep-me")]
    assert result[0] is not envelope
    assert result[0].items[0].data["waveform_ref"].payload_id == "payload"


def test_audio_materializer_uses_segment_duration_for_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    estimates: list[int] = []

    class _Reader:
        @staticmethod
        def process(task: AudioTask) -> AudioTask:
            task.data["waveform"] = torch.zeros((1, 4), dtype=torch.float32)
            task.data["sample_rate"] = 16_000
            task.data["num_samples"] = 4
            return task

    class _Manager:
        @staticmethod
        def put(payload: object, *, metadata: dict[str, Any], estimated_bytes: int) -> PayloadRef:
            assert torch.is_tensor(payload)
            assert metadata["num_samples"] == 4
            estimates.append(estimated_bytes)
            return _payload_ref()

    stage = AudioPayloadMaterializeStage()
    stage._reader = _Reader()
    stage._manager = _Manager()
    monkeypatch.setattr(stage, "_ensure_ready", lambda: None)
    task = AudioTask(
        data={
            "audio_filepath": "parent.wav",
            "duration": 10.0,
            "segment_start_s": 2.0,
            "segment_duration_s": 0.25,
        }
    )

    stage.process(task)

    assert estimates == [16_000]


def test_release_preserves_envelope_and_strips_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    released: list[str] = []
    monkeypatch.setattr(
        payload_lifecycle,
        "release_payload_ref",
        lambda payload_ref: released.append(payload_ref.payload_id),
    )
    item = AudioTask(data={"waveform_ref": _payload_ref(), "waveform": object(), "text": "kept"})
    envelope = _FakeEnvelope(items=(item,), marker="keep-me")

    result = PayloadReleaseStage().process_batch([envelope])

    assert result[0].marker == "keep-me"
    assert result[0].items[0].data == {"text": "kept"}
    assert released == ["payload"]


def test_structural_envelope_materialize_consume_release_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[str] = []
    monkeypatch.setattr(
        payload_lifecycle,
        "resolve_payload_refs_batched",
        lambda _refs: [torch.tensor([[0.0, 0.1, 0.2, 0.3]])],
    )
    monkeypatch.setattr(
        payload_lifecycle,
        "release_payload_ref",
        lambda payload_ref: released.append(payload_ref.payload_id),
    )
    child = AudioTask(
        data={
            "audio_filepath": "segment.wav",
            "duration": 10.0,
            "segment_start_s": 2.0,
            "segment_duration_s": 0.25,
        }
    )
    envelope = _FakeEnvelope(items=(child,), marker="structural")
    materializer = AudioPayloadMaterializeStage()
    monkeypatch.setattr(
        materializer,
        "process",
        lambda task: task.data.__setitem__("waveform_ref", _payload_ref()) or task,
    )
    consumer = _FakeFastConformerConsumer()
    wrapper = PayloadResolvingStage(wrapped_stage=consumer)
    release = PayloadReleaseStage()

    materialized = materializer.process_batch([envelope])
    consumed = wrapper.process_batch(materialized)

    assert consumer.saw_waveform is True
    assert consumed[0].marker == "structural"
    assert consumed[0].items[0].data["pred_text"] == "samples=4"
    assert "waveform" not in consumed[0].items[0].data
    assert consumed[0].items[0].data["waveform_ref"].payload_id == "payload"

    final = release.process_batch(consumed)
    assert final[0].marker == "structural"
    assert final[0].items[0].data["pred_text"] == "samples=4"
    assert "waveform_ref" not in final[0].items[0].data
    assert released == ["payload"]
