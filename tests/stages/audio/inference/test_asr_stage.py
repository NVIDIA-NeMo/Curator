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
# ruff: noqa: S108

"""Tests for the generic ``ASRStage`` exercised against a mock ``ASRAdapter`` (no real model load)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.models.asr.base import ASRResult
from nemo_curator.pipeline.payload_refs import PayloadRef
from nemo_curator.stages.audio.inference.asr import ASRStage
from nemo_curator.stages.audio.inference.asr import stage as asr_stage_module
from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy
from nemo_curator.tasks import AudioTask, DispatchBatchTask

_QWEN_ADAPTER_TARGET = "nemo_curator.models.asr.qwen_omni.QwenOmniASRAdapter"
_SR = 16000


def _make_stage(  # noqa: PLR0913
    *,
    disfluency_text_key: str | None = None,
    keep_waveform: bool = True,
    default_language: str | None = None,
    max_inference_duration_s: float = 2400.0,
    batch_policy: BatchPolicy | None = None,
    batch_size: int = 32,
    reference_text_key: str | None = None,
    supported_language_codes: list[str] | None = None,
    payload_prefetch_enabled: bool = False,
    payload_prefetch_max_bytes: int | None = None,
) -> ASRStage:
    """Build an ASRStage wired to a mock adapter (no real model load)."""
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/qwen-omni",
        pred_text_key="qwen3_prediction_s1",
        disfluency_text_key=disfluency_text_key,
        keep_waveform=keep_waveform,
        default_language=default_language,
        max_inference_duration_s=max_inference_duration_s,
        batch_policy=batch_policy,
        batch_size=batch_size,
        reference_text_key=reference_text_key,
        supported_language_codes=supported_language_codes,
        payload_prefetch_enabled=payload_prefetch_enabled,
        payload_prefetch_max_bytes=payload_prefetch_max_bytes,
    )
    mock_adapter = MagicMock()
    mock_adapter.last_metrics = {}
    stage._adapter = mock_adapter
    return stage


def _make_task(waveform_len: int = _SR, source_lang: str | None = "en") -> AudioTask:
    data: dict[str, object] = {
        "waveform": np.zeros(waveform_len, dtype=np.float32),
        "sample_rate": _SR,
    }
    if source_lang is not None:
        data["source_lang"] = source_lang
    return AudioTask(data=data)


def _chunking_policy() -> BatchPolicy:
    return BatchPolicy(
        buckets_sec=[0],
        max_items_per_batch_by_bucket=[32],
        max_audio_sec_per_batch=None,
    )


def _dispatch_batch(
    stage: ASRStage,
    policy: BatchPolicy,
    *,
    batch_index: int,
    durations_s: list[float],
) -> DispatchBatchTask:
    tasks = [_make_task(waveform_len=int(_SR * duration_s)) for duration_s in durations_s]
    bucket_index = policy.bucket_for(durations_s[0])
    assert all(policy.bucket_for(duration_s) == bucket_index for duration_s in durations_s)
    return DispatchBatchTask(
        dataset_name="dataset",
        data=tasks,
        batch_id=f"run:dispatch:{batch_index}",
        owner_stage=stage.name,
        sequence_index=batch_index,
        bucket_index=bucket_index,
        total_cost=sum(durations_s),
        item_costs=tuple(durations_s),
        cost_unit="audio_seconds",
        policy_signature=policy.dispatch_signature(cost_unit="audio_seconds"),
    )


# ----------------------------------------------------------------------
# Stage-level: process / process_batch contract
# ----------------------------------------------------------------------


def test_process_raises_not_implemented() -> None:
    stage = _make_stage()
    with pytest.raises(NotImplementedError):
        stage.process(_make_task())


def test_empty_batch() -> None:
    stage = _make_stage()
    assert stage.process_batch([]) == []


def test_basic_inference_single_turn() -> None:
    stage = _make_stage(keep_waveform=False)
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hello world")]

    results = stage.process_batch([_make_task()])

    assert results[0].data["qwen3_prediction_s1"] == "hello world"
    assert "waveform" not in results[0].data  # keep_waveform=False -> dropped


def test_pre_skipped_plain_row_bypasses_adapter_without_blocking_valid_neighbor() -> None:
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="valid transcript")]
    skipped = AudioTask(data={"_skip_me": "audio_read_error", "audio_read_error": "decode failed"})
    valid = _make_task()

    results = stage.process_batch([skipped, valid])

    [adapter_items] = [call.args[0] for call in stage._adapter.transcribe_batch.call_args_list]
    assert len(adapter_items) == 1
    assert results[0].data["_skip_me"] == "audio_read_error"
    assert results[0].data["qwen3_prediction_s1"] == ""
    assert results[1].data["qwen3_prediction_s1"] == "valid transcript"


def test_pre_skipped_dispatch_row_bypasses_adapter_without_blocking_valid_neighbor() -> None:
    policy = _chunking_policy()
    stage = _make_stage(batch_policy=policy)
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="valid transcript")]
    skipped = AudioTask(data={"_skip_me": "audio_read_error", "audio_read_error": "decode failed"})
    valid = _make_task()
    batch = DispatchBatchTask(
        dataset_name="dataset",
        data=[skipped, valid],
        batch_id="run:dispatch:mixed-read-error",
        owner_stage=stage.name,
        sequence_index=0,
        bucket_index=policy.bucket_for(1.0),
        total_cost=2.0,
        item_costs=(1.0, 1.0),
        cost_unit="audio_seconds",
        policy_signature=policy.dispatch_signature(cost_unit="audio_seconds"),
    )

    [result_batch] = stage.process_batch([batch])

    assert isinstance(result_batch, DispatchBatchTask)
    [adapter_items] = [call.args[0] for call in stage._adapter.transcribe_batch.call_args_list]
    assert len(adapter_items) == 1
    assert result_batch.items[0].data["_skip_me"] == "audio_read_error"
    assert result_batch.items[0].data["qwen3_prediction_s1"] == ""
    assert result_batch.items[1].data["qwen3_prediction_s1"] == "valid transcript"


def test_keep_waveform_default_is_true() -> None:
    """Default ``keep_waveform`` is True so downstream stages can reuse the waveform."""
    stage = _make_stage()  # no keep_waveform override
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hello world")]

    results = stage.process_batch([_make_task()])
    assert "waveform" in results[0].data


def test_disfluency_text_key_stores_secondary() -> None:
    stage = _make_stage(disfluency_text_key="qwen3_prediction_s2")
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="hello world", secondary_text="hello world cleaned"),
    ]

    results = stage.process_batch([_make_task()])

    assert results[0].data["qwen3_prediction_s1"] == "hello world"
    assert results[0].data["qwen3_prediction_s2"] == "hello world cleaned"
    assert "qwen3_prediction_s2" in stage.outputs()[1]


def test_disfluency_text_key_none_is_normalised_to_empty_string() -> None:
    stage = _make_stage(disfluency_text_key="qwen3_prediction_s2")
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="hello world", secondary_text=None),
    ]

    results = stage.process_batch([_make_task()])
    assert results[0].data["qwen3_prediction_s2"] == ""


def test_keep_waveform_false_drops_waveform() -> None:
    stage = _make_stage(keep_waveform=False)
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="text")]

    results = stage.process_batch([_make_task()])
    assert "waveform" not in results[0].data


def test_adapter_not_initialized_raises() -> None:
    stage = ASRStage(adapter_target=_QWEN_ADAPTER_TARGET, model_id="mock/model")
    with pytest.raises(RuntimeError, match="setup"):
        stage.process_batch([_make_task()])


def test_multi_task_batch_preserves_order() -> None:
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="text1"),
        ASRResult(text="text2"),
    ]
    results = stage.process_batch([_make_task(), _make_task()])

    assert results[0].data["qwen3_prediction_s1"] == "text1"
    assert results[1].data["qwen3_prediction_s1"] == "text2"


def test_adapter_result_length_mismatch_raises() -> None:
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="x")]  # 1 result
    with pytest.raises(RuntimeError, match=r"returned 1 results for 2 items"):
        stage.process_batch([_make_task(), _make_task()])


def test_dispatch_batches_are_exact_adapter_call_boundaries() -> None:
    policy = BatchPolicy(
        buckets_sec=[0.0, 30.0, 60.0],
        max_items_per_batch_by_bucket=[3, 2, 1],
        max_audio_sec_per_batch=60.0,
    )
    stage = _make_stage(max_inference_duration_s=60.0, batch_policy=policy)
    first = _dispatch_batch(stage, policy, batch_index=0, durations_s=[10.0, 20.0])
    second = _dispatch_batch(stage, policy, batch_index=1, durations_s=[40.0])
    stage._adapter.transcribe_batch.side_effect = lambda items: [
        ASRResult(text=f"samples={len(item['waveform'])}") for item in items
    ]
    policy.bucketize_with_costs = MagicMock(side_effect=AssertionError("owner must not rebucket dispatch batches"))

    result = stage.process_batch([first, second])

    assert stage._adapter.transcribe_batch.call_count == 2
    calls = stage._adapter.transcribe_batch.call_args_list
    assert [[len(item["waveform"]) / _SR for item in call.args[0]] for call in calls] == [
        [10.0, 20.0],
        [40.0],
    ]
    assert [batch.batch_id for batch in result] == [first.batch_id, second.batch_id]
    assert [[item.data["qwen3_prediction_s1"] for item in batch.items] for batch in result] == [
        [f"samples={10 * _SR}", f"samples={20 * _SR}"],
        [f"samples={40 * _SR}"],
    ]


def test_dispatch_batch_rejects_owner_policy_mismatch() -> None:
    policy = BatchPolicy(
        buckets_sec=[0.0, 30.0],
        max_items_per_batch_by_bucket=[2, 1],
        max_audio_sec_per_batch=60.0,
    )
    stage = _make_stage(max_inference_duration_s=60.0, batch_policy=policy)
    batch = _dispatch_batch(stage, policy, batch_index=0, durations_s=[10.0, 20.0])
    batch.policy_signature = "different-policy"

    with pytest.raises(ValueError, match="policy constraints do not match"):
        stage.process_batch([batch])

    stage._adapter.transcribe_batch.assert_not_called()


def test_dispatch_batch_accepts_safe_decoded_duration_drift() -> None:
    policy = BatchPolicy(
        buckets_sec=[0.0, 30.0],
        max_items_per_batch_by_bucket=[2, 1],
        max_audio_sec_per_batch=60.0,
    )
    stage = _make_stage(max_inference_duration_s=60.0, batch_policy=policy)
    batch = _dispatch_batch(stage, policy, batch_index=0, durations_s=[30.0])
    batch.items[0].data["waveform"] = np.zeros(int(_SR * 29.9), dtype=np.float32)
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="ok")]

    [result] = stage.process_batch([batch])

    assert result.items[0].data["qwen3_prediction_s1"] == "ok"
    stage._adapter.transcribe_batch.assert_called_once()


def test_dispatch_batch_rejects_item_that_owner_would_segment_again() -> None:
    policy = BatchPolicy(
        buckets_sec=[0.0, 30.0, 60.0],
        max_items_per_batch_by_bucket=[2, 1, 1],
        max_audio_sec_per_batch=120.0,
    )
    stage = _make_stage(max_inference_duration_s=60.0, batch_policy=policy)
    batch = _dispatch_batch(stage, policy, batch_index=0, durations_s=[61.0])

    with pytest.raises(RuntimeError, match="segmentation would change its item boundaries"):
        stage.process_batch([batch])

    stage._adapter.transcribe_batch.assert_not_called()


# ----------------------------------------------------------------------
# Model-input segmentation + stitch-back
# ----------------------------------------------------------------------


def test_pre_slice_short_clip_passes_through_unchanged() -> None:
    """A clip under max_inference_duration_s yields one sub-chunk; no stitching."""
    stage = _make_stage(
        max_inference_duration_s=2400.0,
        batch_policy=_chunking_policy(),
    )
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="single")]

    BaseStageAdapter(stage).process_batch([_make_task(waveform_len=_SR * 30)])  # 30 s clip

    items = stage._adapter.transcribe_batch.call_args[0][0]
    assert len(items) == 1
    assert items[0]["chunk_count"] == 1
    assert items[0]["chunk_idx"] == 0


def test_pre_slice_over_long_clip_into_contiguous_sub_chunks() -> None:
    """A 95-s clip with max_inference_duration_s=30 s slices into [30, 30, 30, 5] sub-chunks."""
    stage = _make_stage(
        max_inference_duration_s=30.0,
        batch_policy=_chunking_policy(),
    )
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="chunk0"),
        ASRResult(text="chunk1"),
        ASRResult(text="chunk2"),
        ASRResult(text="chunk3"),
    ]
    waveform = np.arange(_SR * 95, dtype=np.float32)
    task = AudioTask(data={"waveform": waveform, "sample_rate": _SR, "source_lang": "en"})
    BaseStageAdapter(stage).process_batch([task])

    items = stage._adapter.transcribe_batch.call_args[0][0]
    assert len(items) == 4
    assert [it["chunk_idx"] for it in items] == [0, 1, 2, 3]
    assert all(it["chunk_count"] == 4 for it in items)
    chunk_lengths = [int(it["waveform"].shape[0]) for it in items]
    assert chunk_lengths == [_SR * 30, _SR * 30, _SR * 30, _SR * 5]
    assert sum(chunk_lengths) == int(waveform.shape[0])  # no audio lost / repeated
    # Sub-chunks are the contiguous prefix of waveform.
    np.testing.assert_array_equal(items[0]["waveform"], waveform[: _SR * 30])
    np.testing.assert_array_equal(items[1]["waveform"], waveform[_SR * 30 : _SR * 60])
    np.testing.assert_array_equal(items[2]["waveform"], waveform[_SR * 60 : _SR * 90])
    np.testing.assert_array_equal(items[3]["waveform"], waveform[_SR * 90 :])


def test_pre_slice_canonical_torch_waveform_uses_sample_axis() -> None:
    """Canonical ``(channels, samples)`` tensors are sliced along the sample axis."""
    stage = _make_stage(
        max_inference_duration_s=30.0,
        batch_policy=_chunking_policy(),
    )
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="chunk0"),
        ASRResult(text="chunk1"),
        ASRResult(text="chunk2"),
        ASRResult(text="chunk3"),
    ]
    waveform = torch.arange(_SR * 95, dtype=torch.float32).reshape(1, -1)
    task = AudioTask(data={"waveform": waveform, "sample_rate": _SR, "source_lang": "en"})
    BaseStageAdapter(stage).process_batch([task])

    items = stage._adapter.transcribe_batch.call_args[0][0]
    assert len(items) == 4
    assert [tuple(it["waveform"].shape) for it in items] == [
        (1, _SR * 30),
        (1, _SR * 30),
        (1, _SR * 30),
        (1, _SR * 5),
    ]
    torch.testing.assert_close(items[0]["waveform"], waveform[:, : _SR * 30])


def test_pre_slice_stitch_back_joins_per_parent_with_single_space() -> None:
    """Stitch-back joins sub-chunk texts (and secondary texts) with a single space; one row per parent."""
    stage = _make_stage(
        disfluency_text_key="qwen3_prediction_s2",
        max_inference_duration_s=30.0,
        batch_policy=_chunking_policy(),
    )
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="hello", secondary_text="hello clean"),
        ASRResult(text="world", secondary_text="world clean"),
    ]
    waveform = np.zeros(_SR * 50, dtype=np.float32)  # 50s -> 2 sub-chunks
    task = AudioTask(data={"waveform": waveform, "sample_rate": _SR, "source_lang": "en"})
    results = BaseStageAdapter(stage).process_batch([task])

    assert len(results) == 1  # one parent in, one parent out
    assert results[0].data["qwen3_prediction_s1"] == "hello world"
    assert results[0].data["qwen3_prediction_s2"] == "hello clean world clean"


def test_pre_slice_marks_parent_skipped_only_if_all_chunks_skipped() -> None:
    """A parent is marked skipped only if every sub-chunk was skipped."""
    stage = _make_stage(
        max_inference_duration_s=30.0,
        batch_policy=_chunking_policy(),
    )
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="good", skipped=False),
        ASRResult(text="", skipped=True),
        # task2: every chunk skipped
        ASRResult(text="", skipped=True),
        ASRResult(text="", skipped=True),
    ]
    task_partial = AudioTask(
        data={
            "waveform": np.zeros(_SR * 50, dtype=np.float32),
            "sample_rate": _SR,
        }
    )
    task_all_skip = AudioTask(
        data={
            "waveform": np.zeros(_SR * 50, dtype=np.float32),
            "sample_rate": _SR,
        }
    )
    results = BaseStageAdapter(stage).process_batch([task_partial, task_all_skip])

    assert results[0].data["qwen3_prediction_s1"] == "good"
    assert results[0].data.get("_skip_me") != "empty_audio"
    assert results[1].data["qwen3_prediction_s1"] == ""
    assert results[1].data["_skip_me"] == "empty_audio"


def test_segment_metrics_count_model_items_and_parent_rows() -> None:
    """Metrics count model segments while stitch-back restores parent rows."""
    stage = _make_stage(
        max_inference_duration_s=30.0,
        batch_policy=_chunking_policy(),
    )
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="a"),
        ASRResult(text="b"),
        ASRResult(text="c"),
        ASRResult(text="d"),
    ]
    stage._log_metrics = MagicMock()  # type: ignore[method-assign]

    task_short = AudioTask(data={"waveform": np.zeros(_SR * 10, dtype=np.float32), "sample_rate": _SR})  # 1 chunk
    task_long = AudioTask(data={"waveform": np.zeros(_SR * 75, dtype=np.float32), "sample_rate": _SR})  # 3 chunks
    BaseStageAdapter(stage).process_batch([task_short, task_long])

    metrics = stage._log_metrics.call_args[0][0]
    assert metrics["utterances_input"] == 2.0
    assert metrics["utterances_processed"] == 2.0
    assert metrics["sub_chunks_generated"] == 4.0


def test_model_input_segmentation_without_batch_policy_slices_normal_flow() -> None:
    """Chunking is independent from scheduler bucketing and works in normal process_batch."""
    stage = _make_stage(
        max_inference_duration_s=30.0,
        batch_policy=BatchPolicy(
            enabled=False,
            strategy="placeholder",
            buckets_sec=[],
            max_items_per_batch_by_bucket=[],
        ),
    )
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="chunk0"),
        ASRResult(text="chunk1"),
        ASRResult(text="chunk2"),
        ASRResult(text="chunk3"),
    ]
    waveform = np.arange(_SR * 95, dtype=np.float32)
    task = AudioTask(data={"waveform": waveform, "sample_rate": _SR, "source_lang": "en"})

    result = stage.process_batch([task])

    items = stage._adapter.transcribe_batch.call_args[0][0]
    assert [it["chunk_idx"] for it in items] == [0, 1, 2, 3]
    assert all(it["chunk_count"] == 4 for it in items)
    assert result[0].data["qwen3_prediction_s1"] == "chunk0 chunk1 chunk2 chunk3"


def test_model_input_segmentation_normal_flow_caps_adapter_calls_by_batch_size() -> None:
    stage = _make_stage(
        max_inference_duration_s=30.0,
        batch_policy=None,
        batch_size=2,
    )
    stage._adapter.transcribe_batch.side_effect = [
        [ASRResult(text="chunk0"), ASRResult(text="chunk1")],
        [ASRResult(text="chunk2"), ASRResult(text="chunk3")],
    ]
    waveform = np.arange(_SR * 95, dtype=np.float32)
    task = AudioTask(data={"waveform": waveform, "sample_rate": _SR, "source_lang": "en"})

    result = stage.process_batch([task])

    assert [len(call.args[0]) for call in stage._adapter.transcribe_batch.call_args_list] == [2, 2]
    assert result[0].data["qwen3_prediction_s1"] == "chunk0 chunk1 chunk2 chunk3"


def test_payload_prefetch_plans_from_metadata_resolves_parent_once_and_slices_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waveform = np.arange(_SR * 75, dtype=np.float32)
    payload_ref = PayloadRef(
        payload_id="payload-1",
        owner_node_id="node-a",
        store_actor_name="store",
        admission_actor_name="admission",
        amount_bytes=int(waveform.nbytes),
        sample_rate=_SR,
        num_samples=len(waveform),
    )
    resolve_calls: list[list[str]] = []

    def resolve(refs: list[PayloadRef]) -> list[np.ndarray]:
        resolve_calls.append([ref.payload_id for ref in refs])
        return [waveform for _ref in refs]

    monkeypatch.setattr(asr_stage_module, "resolve_payload_refs_batched", resolve)
    stage = _make_stage(
        max_inference_duration_s=30.0,
        batch_size=1,
        payload_prefetch_enabled=True,
        payload_prefetch_max_bytes=10_000_000,
    )
    stage.payload_consumer_node_id = MagicMock(return_value="node-a")  # type: ignore[method-assign]
    stage._adapter.transcribe_batch.side_effect = [
        [ASRResult(text="chunk0")],
        [ASRResult(text="chunk1")],
        [ASRResult(text="chunk2")],
    ]
    task = AudioTask(data={"waveform_ref": payload_ref, "sample_rate": _SR, "source_lang": "en"})

    result = stage.process_batch([task])

    assert resolve_calls == [["payload-1"]]
    assert [len(call.args[0][0]["waveform"]) for call in stage._adapter.transcribe_batch.call_args_list] == [
        _SR * 30,
        _SR * 30,
        _SR * 15,
    ]
    assert result[0].data["qwen3_prediction_s1"] == "chunk0 chunk1 chunk2"
    assert "waveform" not in result[0].data


def test_plain_payload_ref_survives_failed_attempt_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    waveform = np.zeros(_SR, dtype=np.float32)
    payload_ref = PayloadRef("plain-retry", "node-a", "store", "admission", waveform.nbytes, _SR, len(waveform))
    resolve_calls: list[str] = []

    def resolve(refs: list[PayloadRef]) -> list[np.ndarray]:
        resolve_calls.extend(ref.payload_id for ref in refs)
        return [waveform for _ref in refs]

    monkeypatch.setattr("nemo_curator.stages.payload_lifecycle.resolve_payload_refs_batched", resolve)
    stage = _make_stage()
    stage._adapter.transcribe_batch.side_effect = [RuntimeError("transient"), [ASRResult(text="recovered")]]
    task = AudioTask(data={"waveform_ref": payload_ref, "sample_rate": _SR, "source_lang": "en"})
    backend = BaseStageAdapter(stage)

    with pytest.raises(RuntimeError, match="transient"):
        backend.process_batch([task])

    assert task.data["waveform_ref"] is payload_ref
    assert "waveform" not in task.data
    [output] = backend.process_batch([task])
    assert output.data["qwen3_prediction_s1"] == "recovered"
    assert resolve_calls == ["plain-retry", "plain-retry"]


def test_dispatch_payload_ref_survives_failed_attempt_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    waveform = np.zeros(_SR, dtype=np.float32)
    payload_ref = PayloadRef(
        "dispatch-retry",
        "node-a",
        "store",
        "admission",
        waveform.nbytes,
        _SR,
        len(waveform),
    )

    def resolve(refs: list[PayloadRef]) -> list[np.ndarray]:
        return [waveform for _ref in refs]

    monkeypatch.setattr("nemo_curator.stages.payload_lifecycle.resolve_payload_refs_batched", resolve)
    policy = _chunking_policy()
    stage = _make_stage(batch_policy=policy)
    stage._adapter.transcribe_batch.side_effect = [RuntimeError("transient"), [ASRResult(text="recovered")]]
    child = AudioTask(data={"waveform_ref": payload_ref, "sample_rate": _SR, "source_lang": "en"})
    batch = DispatchBatchTask(
        dataset_name="dataset",
        data=[child],
        batch_id="run:dispatch:retry",
        owner_stage=stage.name,
        sequence_index=0,
        bucket_index=policy.bucket_for(1.0),
        total_cost=1.0,
        item_costs=(1.0,),
        cost_unit="audio_seconds",
        policy_signature=policy.dispatch_signature(cost_unit="audio_seconds"),
    )

    with pytest.raises(RuntimeError, match="transient"):
        stage.process_batch([batch])

    assert child.data["waveform_ref"] is payload_ref
    assert "waveform" not in child.data
    [retried_batch] = stage.process_batch([batch])
    assert isinstance(retried_batch, DispatchBatchTask)
    assert retried_batch.items[0].data["qwen3_prediction_s1"] == "recovered"


def test_payload_prefetch_requires_explicit_byte_budget() -> None:
    with pytest.raises(ValueError, match="payload_prefetch_max_bytes is required"):
        _make_stage(payload_prefetch_enabled=True)


def test_payload_prefetch_enabled_requires_bool() -> None:
    with pytest.raises(TypeError, match="payload_prefetch_enabled must be a bool"):
        _make_stage(payload_prefetch_enabled="true")  # type: ignore[arg-type]


def test_payload_prefetch_byte_limit_rejects_bool() -> None:
    with pytest.raises(ValueError, match="payload_prefetch_max_bytes"):
        _make_stage(payload_prefetch_max_bytes=True)


# ----------------------------------------------------------------------
# Stage-level: language mapping (ISO code -> name)
# ----------------------------------------------------------------------


def test_language_resolution_from_task() -> None:
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hola")]

    task = AudioTask(
        data={
            "waveform": np.zeros(_SR, dtype=np.float32),
            "sample_rate": _SR,
            "source_lang": "es",
        }
    )
    stage.process_batch([task])

    items = stage._adapter.transcribe_batch.call_args[0][0]
    assert items[0]["language"] == "Spanish"


def test_default_language_used_when_task_language_missing() -> None:
    stage = _make_stage(default_language="en")
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hello")]

    task = AudioTask(
        data={
            "waveform": np.zeros(_SR, dtype=np.float32),
            "sample_rate": _SR,
        }
    )
    stage.process_batch([task])

    items = stage._adapter.transcribe_batch.call_args[0][0]
    assert items[0]["language"] == "English"


def test_supported_language_filter_skips_before_adapter_call() -> None:
    stage = _make_stage(supported_language_codes=["en"])

    results = stage.process_batch([_make_task(source_lang="pl")])

    stage._adapter.transcribe_batch.assert_not_called()
    assert results[0].data["qwen3_prediction_s1"] == ""
    assert results[0].data["_skip_me"] == "lang_not_supported:pl"


def test_reference_text_key_is_passed_to_adapter_items() -> None:
    stage = _make_stage(reference_text_key="text")
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hello")]

    task = AudioTask(
        data={
            "waveform": np.zeros(_SR, dtype=np.float32),
            "sample_rate": _SR,
            "source_lang": "en",
            "text": "reference transcript",
        }
    )
    stage.process_batch([task])

    items = stage._adapter.transcribe_batch.call_args[0][0]
    assert items[0]["reference_text"] == "reference transcript"


def test_reference_text_key_is_preserved_for_chunked_items() -> None:
    stage = _make_stage(
        max_inference_duration_s=30.0,
        reference_text_key="text",
    )
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="chunk0"),
        ASRResult(text="chunk1"),
    ]

    task = AudioTask(
        data={
            "waveform": np.zeros(_SR * 50, dtype=np.float32),
            "sample_rate": _SR,
            "source_lang": "en",
            "text": "reference transcript",
        }
    )
    stage.process_batch([task])

    items = stage._adapter.transcribe_batch.call_args[0][0]
    assert [item["reference_text"] for item in items] == ["reference transcript", "reference transcript"]


# ----------------------------------------------------------------------
# Stage-level: I/O contract
# ----------------------------------------------------------------------


def test_inputs_outputs_single_turn() -> None:
    stage = ASRStage(adapter_target=_QWEN_ADAPTER_TARGET, model_id="mock/model")
    _required, optional_inputs = stage.inputs()
    assert "waveform_ref" in optional_inputs
    assert "sample_rate" in optional_inputs

    _required, optional_outputs = stage.outputs()
    assert "_skip_me" in optional_outputs
    assert "pred_text" in optional_outputs


def test_inputs_use_waveform_when_payload_ref_is_disabled() -> None:
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/model",
        waveform_ref_key=None,
    )
    _required, optional_inputs = stage.inputs()
    assert "waveform" in optional_inputs


def test_inputs_include_reference_text_key_when_configured() -> None:
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/model",
        reference_text_key="text",
    )
    _required, optional_inputs = stage.inputs()
    assert "text" in optional_inputs


def test_outputs_two_turn_includes_disfluency_key() -> None:
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/model",
        disfluency_text_key="pred_text_secondary",
    )
    _required, optional_outputs = stage.outputs()
    assert "pred_text_secondary" in optional_outputs


# ----------------------------------------------------------------------
# Stage-level: skip / metrics
# ----------------------------------------------------------------------


def test_skipped_result_sets_skip_key() -> None:
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="", skipped=True),
    ]
    results = stage.process_batch([_make_task()])
    assert results[0].data["_skip_me"] == "empty_audio"


def test_metrics_account_skipped_utterances() -> None:
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="text"),
        ASRResult(text="", skipped=True),
    ]
    stage._log_metrics = MagicMock()  # type: ignore[method-assign]

    stage.process_batch([_make_task(), _make_task()])

    metrics = stage._log_metrics.call_args[0][0]
    assert metrics["utterances_input"] == 2.0
    assert metrics["utterances_processed"] == 1.0
    assert metrics["utterances_skipped"] == 1.0
    assert metrics["sub_chunks_generated"] == 2.0
    assert metrics["adapter_inference_calls"] == 1.0
    assert metrics["adapter_inference_items"] == 2.0


def test_metrics_count_actual_adapter_inference_calls_after_chunk_splitting() -> None:
    stage = _make_stage(
        max_inference_duration_s=30.0,
        batch_size=1,
    )
    stage._adapter.transcribe_batch.side_effect = [
        [ASRResult(text="a")],
        [ASRResult(text="b")],
        [ASRResult(text="c")],
    ]
    stage._log_metrics = MagicMock()  # type: ignore[method-assign]

    task = AudioTask(data={"waveform": np.zeros(_SR * 75, dtype=np.float32), "sample_rate": _SR})
    stage.process_batch([task])

    metrics = stage._log_metrics.call_args[0][0]
    assert metrics["utterances_input"] == 1.0
    assert metrics["sub_chunks_generated"] == 3.0
    assert metrics["adapter_inference_calls"] == 3.0
    assert metrics["adapter_inference_items"] == 3.0
    assert stage._adapter.transcribe_batch.call_count == 3


def test_metrics_model_alias_skips_already_emitted_keys() -> None:
    """Adapter metrics the stage already emits must NOT be re-aliased as ``model_<name>``."""
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="text")]
    stage._adapter.last_metrics = {
        "audio_duration_s": 999.0,
        "extra_diagnostic_metric": 42.0,
    }
    stage._log_metrics = MagicMock()  # type: ignore[method-assign]

    stage.process_batch([_make_task()])

    metrics = stage._log_metrics.call_args[0][0]
    assert "model_audio_duration_s" not in metrics
    assert metrics["model_extra_diagnostic_metric"] == 42.0


# ----------------------------------------------------------------------
# Stage-level: setup_on_node weight prefetch + setup() adapter construction
# ----------------------------------------------------------------------


@patch("nemo_curator.models.asr.qwen_omni.snapshot_download")
def test_setup_on_node_downloads_weights(mock_download: MagicMock) -> None:
    stage = ASRStage(adapter_target=_QWEN_ADAPTER_TARGET, model_id="mock/model")
    stage.setup_on_node()
    mock_download.assert_called_once_with("mock/model")


@patch(
    "nemo_curator.models.asr.qwen_omni.snapshot_download",
    side_effect=RuntimeError("missing auth"),
)
def test_setup_on_node_raises_by_default(mock_download: MagicMock) -> None:
    stage = ASRStage(adapter_target=_QWEN_ADAPTER_TARGET, model_id="mock/model")
    with pytest.raises(RuntimeError, match="prefetch_weights failed"):
        stage.setup_on_node()
    mock_download.assert_called_once_with("mock/model")


@patch(
    "nemo_curator.models.asr.qwen_omni.snapshot_download",
    side_effect=RuntimeError("offline"),
)
def test_setup_on_node_can_warn_and_retry_later(mock_download: MagicMock) -> None:
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/model",
        prefetch_fail_on_error=False,
    )
    stage.setup_on_node()
    mock_download.assert_called_once_with("mock/model")


def test_adapter_target_required() -> None:
    with pytest.raises(TypeError):
        ASRStage(model_id="mock/model")


def test_model_id_required() -> None:
    with pytest.raises(TypeError):
        ASRStage(adapter_target=_QWEN_ADAPTER_TARGET)


def test_max_inference_duration_defaults_to_qwen_window() -> None:
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/model",
    )
    assert stage.max_inference_duration_s == 2400.0


def test_setup_uses_adapter_target_and_kwargs() -> None:
    """``setup()`` resolves adapter_target via hydra.utils.get_class and
    constructs the adapter with model_id+revision+**adapter_kwargs."""
    adapter_kwargs = {
        "prompt_file": "/tmp/ml.md",
        "en_prompt_file": "/tmp/en.md",
        "followup_prompt_file": "/tmp/followup.md",
        "system_prompt_file": "/tmp/system.md",
        "max_model_len": 8192,
        "max_num_batched_tokens": 49152,
        "max_num_seqs": 8,
        "gpu_memory_utilization": 0.8,
        "tensor_parallel_size": 4,
        "max_output_tokens": 384,
        "temperature": 0.1,
        "top_k": 5,
        "prep_workers": 16,
        "enable_prefix_caching": False,
        "prefix_caching_hash_algo": "sha256",
        "limit_mm_per_prompt_audio": 1,
        "seed": 999,
    }
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/model",
        revision="abc123",
        adapter_kwargs=adapter_kwargs,
    )

    fake_adapter = MagicMock()
    fake_cls = MagicMock(return_value=fake_adapter)
    with patch("hydra.utils.get_class", return_value=fake_cls) as get_class:
        stage.setup()

    get_class.assert_called_with(_QWEN_ADAPTER_TARGET)
    fake_cls.assert_called_once_with(
        model_id="mock/model",
        revision="abc123",
        **adapter_kwargs,
    )
    fake_adapter.setup.assert_called_once_with()
    assert stage._adapter is fake_adapter


def test_setup_on_node_prefetches_without_constructing_adapter() -> None:
    """``setup_on_node()`` uses the adapter classmethod only.

    Adapter construction is reserved for ``setup()``, where stage-level
    ``adapter_kwargs`` are forwarded to the worker-local adapter instance.
    """
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/model",
        revision="abc123",
        adapter_kwargs={"prompt_file": "/tmp/ml.md"},
    )

    fake_cls = MagicMock()
    fake_cls.prefetch_weights = MagicMock()
    with patch("hydra.utils.get_class", return_value=fake_cls):
        stage.setup_on_node()

    fake_cls.assert_not_called()
    fake_cls.prefetch_weights.assert_called_once_with("mock/model", "abc123")
