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

"""Tests for the generic ``ASRStage`` driven by ``QwenOmniASRAdapter``.

Covers:
    * Tier-1 / Tier-2 stage<->adapter contract;
    * stage-side pre-slice + stitch-back;
    * ``keep_waveform: True`` default;
    * adapter vLLM knobs exposed via ``adapter_kwargs``;
    * within-call duration-bucketed batching via ``BatchPolicy``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nemo_curator.adapters.asr.base import ASRAdapter, ASRResult
from nemo_curator.adapters.asr.qwen_omni import QwenOmniASRAdapter
from nemo_curator.stages.audio.batch_policy import BatchPolicy
from nemo_curator.stages.audio.inference.asr import ASRStage
from nemo_curator.tasks import AudioTask

_QWEN_ADAPTER_TARGET = "nemo_curator.adapters.asr.qwen_omni.QwenOmniASRAdapter"
_SR = 16000


def _make_stage(
    *,
    disfluency_text_key: str | None = None,
    keep_waveform: bool = True,
    default_language: str | None = None,
    ideal_inference_segment_s: float = 2400.0,
    max_inference_duration_s: float | None = None,
    batch_policy: BatchPolicy | None = None,
) -> ASRStage:
    """Build an ASRStage wired to a mock adapter (no real model load)."""
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/qwen-omni",
        pred_text_key="qwen3_prediction_s1",
        disfluency_text_key=disfluency_text_key,
        keep_waveform=keep_waveform,
        default_language=default_language,
        ideal_inference_segment_s=ideal_inference_segment_s,
        max_inference_duration_s=max_inference_duration_s,
        batch_policy=batch_policy,
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


# ----------------------------------------------------------------------
# Stage-side pre-slice + stitch-back
# ----------------------------------------------------------------------


def test_pre_slice_short_clip_passes_through_unchanged() -> None:
    """A clip well under max_inference_duration_s yields exactly one
    sub-chunk; no stitching needed."""
    stage = _make_stage(max_inference_duration_s=2400.0)
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="single")]

    stage.process_batch([_make_task(waveform_len=_SR * 30)])  # 30 s clip

    items = stage._adapter.transcribe_batch.call_args[0][0]
    assert len(items) == 1
    assert items[0]["chunk_count"] == 1
    assert items[0]["chunk_idx"] == 0


def test_pre_slice_over_long_clip_into_contiguous_sub_chunks() -> None:
    """A 95-s clip with max_inference_duration_s=30 s slices into
    [30, 30, 30, 5] sub-chunks (no padding, no overlap, last is short)."""
    stage = _make_stage(
        ideal_inference_segment_s=30.0,
        max_inference_duration_s=30.0,
    )
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="chunk0"),
        ASRResult(text="chunk1"),
        ASRResult(text="chunk2"),
        ASRResult(text="chunk3"),
    ]
    waveform = np.arange(_SR * 95, dtype=np.float32)
    task = AudioTask(data={"waveform": waveform, "sample_rate": _SR, "source_lang": "en"})
    stage.process_batch([task])

    items = stage._adapter.transcribe_batch.call_args[0][0]
    assert len(items) == 4
    assert [it["chunk_idx"] for it in items] == [0, 1, 2, 3]
    assert all(it["chunk_count"] == 4 for it in items)
    chunk_lengths = [int(it["waveform"].shape[0]) for it in items]
    assert chunk_lengths == [_SR * 30, _SR * 30, _SR * 30, _SR * 5]
    assert sum(chunk_lengths) == int(waveform.shape[0])  # no audio lost / repeated
    # First three sub-chunks should be the contiguous prefix of waveform.
    np.testing.assert_array_equal(items[0]["waveform"], waveform[: _SR * 30])
    np.testing.assert_array_equal(items[1]["waveform"], waveform[_SR * 30 : _SR * 60])
    np.testing.assert_array_equal(items[2]["waveform"], waveform[_SR * 60 : _SR * 90])
    np.testing.assert_array_equal(items[3]["waveform"], waveform[_SR * 90 :])


def test_pre_slice_stitch_back_joins_per_parent_with_single_space() -> None:
    """When N sub-chunks per parent come back, stitch-back joins their
    texts (and secondary texts) with a single space; the parent task
    gets one row, not N."""
    stage = _make_stage(
        disfluency_text_key="qwen3_prediction_s2",
        ideal_inference_segment_s=30.0,
        max_inference_duration_s=30.0,
    )
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="hello", secondary_text="hello clean"),
        ASRResult(text="world", secondary_text="world clean"),
    ]
    waveform = np.zeros(_SR * 50, dtype=np.float32)  # 50s -> 2 sub-chunks
    task = AudioTask(data={"waveform": waveform, "sample_rate": _SR, "source_lang": "en"})
    results = stage.process_batch([task])

    assert len(results) == 1  # one parent in, one parent out
    assert results[0].data["qwen3_prediction_s1"] == "hello world"
    assert results[0].data["qwen3_prediction_s2"] == "hello clean world clean"


def test_pre_slice_marks_parent_skipped_only_if_all_chunks_skipped() -> None:
    """Partial success: if any sub-chunk yielded text, the parent is NOT
    marked skipped. If every sub-chunk was skipped, the parent IS marked."""
    stage = _make_stage(
        ideal_inference_segment_s=30.0,
        max_inference_duration_s=30.0,
    )
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="good", skipped=False),
        ASRResult(text="", skipped=True),
        # task2: every chunk skipped
        ASRResult(text="", skipped=True),
        ASRResult(text="", skipped=True),
    ]
    task_partial = AudioTask(data={
        "waveform": np.zeros(_SR * 50, dtype=np.float32),
        "sample_rate": _SR,
    })
    task_all_skip = AudioTask(data={
        "waveform": np.zeros(_SR * 50, dtype=np.float32),
        "sample_rate": _SR,
    })
    results = stage.process_batch([task_partial, task_all_skip])

    assert results[0].data["qwen3_prediction_s1"] == "good"
    assert results[0].data.get("_skip_me") != "empty_audio"
    assert results[1].data["qwen3_prediction_s1"] == ""
    assert results[1].data["_skip_me"] == "empty_audio"


def test_pre_slice_metrics_count_parents_not_chunks() -> None:
    """``utterances_input`` and ``utterances_processed`` count PARENT
    tasks (one row per input); ``sub_chunks_generated`` surfaces the
    fan-out factor for transparency."""
    stage = _make_stage(
        ideal_inference_segment_s=30.0,
        max_inference_duration_s=30.0,
    )
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="a"), ASRResult(text="b"), ASRResult(text="c"), ASRResult(text="d"),
    ]
    stage._log_metrics = MagicMock()  # type: ignore[method-assign]

    task_short = AudioTask(data={"waveform": np.zeros(_SR * 10, dtype=np.float32), "sample_rate": _SR})  # 1 chunk
    task_long = AudioTask(data={"waveform": np.zeros(_SR * 75, dtype=np.float32), "sample_rate": _SR})  # 3 chunks
    stage.process_batch([task_short, task_long])

    metrics = stage._log_metrics.call_args[0][0]
    assert metrics["utterances_input"] == 2.0
    assert metrics["utterances_processed"] == 2.0
    assert metrics["sub_chunks_generated"] == 4.0


# ----------------------------------------------------------------------
# Within-call duration-bucketed batching
# ----------------------------------------------------------------------


def test_batch_policy_partitions_items_by_bucket() -> None:
    """Items in different buckets land in different adapter calls so a
    single vLLM call never mixes a 40-min sub-chunk with 5-s sub-chunks."""
    policy = BatchPolicy(
        strategy="duration_bucketed",
        buckets_sec=[0, 30, 1200, 2400],
        max_items_per_batch_by_bucket=[32, 16, 8, 4],
        max_audio_sec_per_batch=10_000.0,
        flush_interval_ms=250,
    )
    stage = _make_stage(batch_policy=policy)
    # Three tasks: 5 s, 10 s, 600 s. Bucket 1 (ÿÿÿ 30 s) gets [5s, 10s];
    # bucket 2 (ÿÿÿ 1200 s) gets [600 s]. Two adapter calls expected.
    short_a = AudioTask(data={"waveform": np.zeros(_SR * 5, dtype=np.float32), "sample_rate": _SR})
    short_b = AudioTask(data={"waveform": np.zeros(_SR * 10, dtype=np.float32), "sample_rate": _SR})
    long_a = AudioTask(data={"waveform": np.zeros(_SR * 600, dtype=np.float32), "sample_rate": _SR})

    # Mock returns: first call (short bucket: 2 items) -> 2 results;
    # second call (long bucket: 1 item) -> 1 result.
    stage._adapter.transcribe_batch.side_effect = [
        [ASRResult(text="short-a"), ASRResult(text="short-b")],
        [ASRResult(text="long")],
    ]
    results = stage.process_batch([short_a, short_b, long_a])

    assert stage._adapter.transcribe_batch.call_count == 2
    # Parent-order is preserved in the final result list even though
    # internal sub-batching is bucket-ordered.
    assert results[0].data["qwen3_prediction_s1"] == "short-a"
    assert results[1].data["qwen3_prediction_s1"] == "short-b"
    assert results[2].data["qwen3_prediction_s1"] == "long"


def test_batch_policy_respects_per_bucket_item_cap() -> None:
    """A bucket with cap=2 and 3 items must split into 2 sub-batches."""
    policy = BatchPolicy(
        strategy="duration_bucketed",
        buckets_sec=[0, 60],
        max_items_per_batch_by_bucket=[2, 1],
        max_audio_sec_per_batch=None,
    )
    stage = _make_stage(batch_policy=policy)
    stage._adapter.transcribe_batch.side_effect = [
        [ASRResult(text="a"), ASRResult(text="b")],
        [ASRResult(text="c")],
    ]
    tasks = [_make_task() for _ in range(3)]  # all 1s clips -> bucket 0
    stage.process_batch(tasks)
    assert stage._adapter.transcribe_batch.call_count == 2


def test_batch_policy_respects_audio_sec_cap() -> None:
    """A global audio-sec cap forces sub-batch flushes within a bucket."""
    policy = BatchPolicy(
        strategy="duration_bucketed",
        buckets_sec=[0, 60],
        max_items_per_batch_by_bucket=[100, 100],
        max_audio_sec_per_batch=15.0,  # tight: two 10s clips can't share a sub-batch
    )
    stage = _make_stage(batch_policy=policy)
    stage._adapter.transcribe_batch.side_effect = [
        [ASRResult(text="a")],
        [ASRResult(text="b")],
    ]
    tasks = [
        AudioTask(data={"waveform": np.zeros(_SR * 10, dtype=np.float32), "sample_rate": _SR}),
        AudioTask(data={"waveform": np.zeros(_SR * 10, dtype=np.float32), "sample_rate": _SR}),
    ]
    stage.process_batch(tasks)
    assert stage._adapter.transcribe_batch.call_count == 2


def test_batch_policy_none_runs_single_adapter_call() -> None:
    """Default (no policy) keeps the pre-?0.3 single-adapter-call shape."""
    stage = _make_stage(batch_policy=None)
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="a"), ASRResult(text="b"),
    ]
    stage.process_batch([_make_task(), _make_task()])
    assert stage._adapter.transcribe_batch.call_count == 1


def test_batch_policy_invalid_strategy_rejected() -> None:
    with pytest.raises(ValueError, match="duration_bucketed"):
        BatchPolicy(strategy="token_bucketed")


def test_batch_policy_inconsistent_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="lengths must match"):
        BatchPolicy(buckets_sec=[0, 60, 600], max_items_per_batch_by_bucket=[10, 5])


def test_batch_policy_bucket_for_clamps_above_top_edge() -> None:
    """Left-edge semantics: bucket i covers [buckets_sec[i], buckets_sec[i+1])."""
    p = BatchPolicy(buckets_sec=[0, 60, 600], max_items_per_batch_by_bucket=[10, 5, 1])
    assert p.bucket_for(0.0) == 0     # [0, 60)
    assert p.bucket_for(30.0) == 0    # [0, 60)
    assert p.bucket_for(60.0) == 1    # boundary lands in the bucket that starts at 60
    assert p.bucket_for(599.0) == 1   # [60, 600)
    assert p.bucket_for(600.0) == 2   # [600, +inf)
    assert p.bucket_for(9999.0) == 2  # clamped into top bucket


# ----------------------------------------------------------------------
# Stage-level: language mapping (ISO code -> name)
# ----------------------------------------------------------------------


def test_language_resolution_from_task() -> None:
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hola")]

    task = AudioTask(data={
        "waveform": np.zeros(_SR, dtype=np.float32),
        "sample_rate": _SR,
        "source_lang": "es",
    })
    stage.process_batch([task])

    items = stage._adapter.transcribe_batch.call_args[0][0]
    assert items[0]["language"] == "Spanish"


def test_default_language_used_when_task_language_missing() -> None:
    stage = _make_stage(default_language="en")
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hello")]

    task = AudioTask(data={
        "waveform": np.zeros(_SR, dtype=np.float32),
        "sample_rate": _SR,
    })
    stage.process_batch([task])

    items = stage._adapter.transcribe_batch.call_args[0][0]
    assert items[0]["language"] == "English"


# ----------------------------------------------------------------------
# Stage-level: I/O contract
# ----------------------------------------------------------------------


def test_inputs_outputs_single_turn() -> None:
    stage = ASRStage(adapter_target=_QWEN_ADAPTER_TARGET, model_id="mock/model")
    _required, optional_inputs = stage.inputs()
    assert "waveform" in optional_inputs
    assert "sample_rate" in optional_inputs

    _required, optional_outputs = stage.outputs()
    assert "_skip_me" in optional_outputs
    assert "pred_text" in optional_outputs


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


def test_metrics_model_alias_skips_already_emitted_keys() -> None:
    """A5-fix: adapter metrics that the stage already emits must NOT be
    re-aliased as ``model_<name>``."""
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
# Stage-level: setup_on_node weight prefetch
# ----------------------------------------------------------------------


@patch("nemo_curator.adapters.asr.qwen_omni.snapshot_download")
def test_setup_on_node_downloads_weights(mock_download: MagicMock) -> None:
    stage = ASRStage(adapter_target=_QWEN_ADAPTER_TARGET, model_id="mock/model")
    stage.setup_on_node()
    mock_download.assert_called_once_with("mock/model")


@patch(
    "nemo_curator.adapters.asr.qwen_omni.snapshot_download",
    side_effect=RuntimeError("missing auth"),
)
def test_setup_on_node_raises_by_default(mock_download: MagicMock) -> None:
    stage = ASRStage(adapter_target=_QWEN_ADAPTER_TARGET, model_id="mock/model")
    with pytest.raises(RuntimeError, match="prefetch_weights failed"):
        stage.setup_on_node()
    mock_download.assert_called_once_with("mock/model")


@patch(
    "nemo_curator.adapters.asr.qwen_omni.snapshot_download",
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


def test_max_inference_duration_must_not_exceed_ideal() -> None:
    with pytest.raises(ValueError, match=r"must be .* ideal_inference_segment_s"):
        ASRStage(
            adapter_target=_QWEN_ADAPTER_TARGET,
            model_id="mock/model",
            ideal_inference_segment_s=30.0,
            max_inference_duration_s=60.0,
        )


def test_max_inference_duration_defaults_to_ideal() -> None:
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/model",
        ideal_inference_segment_s=1800.0,
    )
    assert stage.max_inference_duration_s == 1800.0


def test_setup_uses_adapter_target_and_kwargs() -> None:
    """``setup()`` resolves adapter_target via hydra.utils.get_class and
    constructs the adapter with model_id+revision+**adapter_kwargs."""
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/model",
        revision="abc123",
        adapter_kwargs={"max_model_len": 8192, "enable_prefix_caching": False},
    )

    fake_adapter = MagicMock()
    fake_cls = MagicMock(return_value=fake_adapter)
    with patch("hydra.utils.get_class", return_value=fake_cls) as get_class:
        stage.setup()

    get_class.assert_called_with(_QWEN_ADAPTER_TARGET)
    fake_cls.assert_called_once_with(
        model_id="mock/model",
        revision="abc123",
        max_model_len=8192,
        enable_prefix_caching=False,
    )
    fake_adapter.setup.assert_called_once_with()
    assert stage._adapter is fake_adapter


# ----------------------------------------------------------------------
# Adapter-level: protocol conformance (requires @runtime_checkable)
# ----------------------------------------------------------------------


def test_qwen_adapter_conforms_to_asr_protocol() -> None:
    """Smoke-check that QwenOmniASRAdapter satisfies the structural ASRAdapter contract.

    ``isinstance(..., ASRAdapter)`` only works because ``ASRAdapter`` is
    decorated with ``@runtime_checkable``; without it Python raises
    ``TypeError`` and we cannot write this (or future adapter-family)
    conformance tests. Mirrors ``test_conforms_to_protocol`` on the
    diarization / VAD / alignment adapter tests on main.
    """
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni")
    assert isinstance(adapter, ASRAdapter)


# ----------------------------------------------------------------------
# Adapter-level: QwenOmniASRAdapter helpers (no GPU, no vLLM required)
# ----------------------------------------------------------------------


def test_qwen_adapter_first_output_text_handles_empty_vllm_output() -> None:
    assert QwenOmniASRAdapter._first_output_text(SimpleNamespace(outputs=[])) == ""


def test_qwen_adapter_count_output_tokens_handles_empty_vllm_output() -> None:
    assert QwenOmniASRAdapter._count_output_tokens([SimpleNamespace(outputs=[])]) == 0.0


def test_qwen_adapter_transcribe_batch_packages_results() -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni", followup_prompt="refine")
    adapter._generate = MagicMock(  # type: ignore[method-assign]
        return_value=(
            ["text-a", "text-b", ""],
            ["refined-a", "", ""],
            {2},
        ),
    )
    items = [
        {"waveform": np.zeros(_SR, dtype=np.float32), "sample_rate": _SR, "language": "English"},
        {"waveform": np.zeros(_SR, dtype=np.float32), "sample_rate": _SR, "language": "English"},
        {"waveform": np.zeros(0, dtype=np.float32), "sample_rate": _SR, "language": None},
    ]
    results = adapter.transcribe_batch(items)

    assert [r.text for r in results] == ["text-a", "text-b", ""]
    assert [r.secondary_text for r in results] == ["refined-a", "", ""]
    assert [r.skipped for r in results] == [False, False, True]
    assert all(r.model_id == "mock/qwen-omni" for r in results)

    adapter._generate.assert_called_once()
    _waveforms, _srs, langs = adapter._generate.call_args[0]
    assert langs == ["English", "English", None]


def test_qwen_adapter_single_turn_drops_secondary_text() -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni", followup_prompt=None)
    adapter._generate = MagicMock(  # type: ignore[method-assign]
        return_value=(["text-a"], [""], set()),
    )
    results = adapter.transcribe_batch([
        {"waveform": np.zeros(_SR, dtype=np.float32), "sample_rate": _SR},
    ])
    assert results[0].secondary_text is None


# ----------------------------------------------------------------------
# Adapter-level vLLM knobs
# ----------------------------------------------------------------------


def test_qwen_adapter_has_elevated_vllm_knobs_as_dataclass_fields() -> None:
    """enable_prefix_caching, prefix_caching_hash_algo, limit_mm_per_prompt_audio,
    and seed are dataclass fields settable from YAML ``adapter_kwargs``.
    """
    adapter = QwenOmniASRAdapter(
        model_id="mock/qwen-omni",
        enable_prefix_caching=False,
        prefix_caching_hash_algo="sha256",
        limit_mm_per_prompt_audio=1,
        seed=99,
    )
    assert adapter.enable_prefix_caching is False
    assert adapter.prefix_caching_hash_algo == "sha256"
    assert adapter.limit_mm_per_prompt_audio == 1
    assert adapter.seed == 99


def test_qwen_adapter_vllm_knob_defaults_match_doc() -> None:
    """Default vLLM knob values match the tutorial when YAML omits overrides."""
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni")
    assert adapter.enable_prefix_caching is True
    assert adapter.prefix_caching_hash_algo == "xxhash"
    assert adapter.limit_mm_per_prompt_audio == 2
    assert adapter.seed == 1234


def test_qwen_adapter_setup_threads_vllm_knobs_into_llm_ctor() -> None:
    """The setup() path must forward the elevated knobs to the vLLM LLM
    ctor (rather than re-using hardcoded constants)."""
    adapter = QwenOmniASRAdapter(
        model_id="mock/qwen-omni",
        enable_prefix_caching=False,
        prefix_caching_hash_algo="sha256",
        limit_mm_per_prompt_audio=3,
        seed=42,
        tensor_parallel_size=1,
    )
    fake_llm = MagicMock()
    fake_processor = MagicMock()
    with (
        patch("nemo_curator.adapters.asr.qwen_omni.VLLM_AVAILABLE", new=True),
        patch("nemo_curator.adapters.asr.qwen_omni.LLM", return_value=fake_llm) as LLM_ctor,
        patch(
            "nemo_curator.adapters.asr.qwen_omni.Qwen3OmniMoeProcessor.from_pretrained",
            return_value=fake_processor,
        ),
        patch("nemo_curator.adapters.asr.qwen_omni.SamplingParams"),
    ):
        adapter.setup()

    LLM_ctor.assert_called_once()
    kwargs = LLM_ctor.call_args.kwargs
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["prefix_caching_hash_algo"] == "sha256"
    assert kwargs["limit_mm_per_prompt"] == {"image": 1, "video": 1, "audio": 3}
    assert kwargs["seed"] == 42
    assert "revision" not in kwargs


def test_qwen_adapter_setup_forwards_revision_to_llm_and_processor() -> None:
    """Tier-1 revision must reach inference loaders, not only prefetch_weights."""
    adapter = QwenOmniASRAdapter(
        model_id="mock/qwen-omni",
        revision="abc123",
        tensor_parallel_size=1,
    )
    fake_llm = MagicMock()
    fake_processor = MagicMock()
    with (
        patch("nemo_curator.adapters.asr.qwen_omni.VLLM_AVAILABLE", new=True),
        patch("nemo_curator.adapters.asr.qwen_omni.LLM", return_value=fake_llm) as LLM_ctor,
        patch(
            "nemo_curator.adapters.asr.qwen_omni.Qwen3OmniMoeProcessor.from_pretrained",
            return_value=fake_processor,
        ) as proc_ctor,
        patch("nemo_curator.adapters.asr.qwen_omni.SamplingParams"),
    ):
        adapter.setup()

    assert LLM_ctor.call_args.kwargs["revision"] == "abc123"
    proc_ctor.assert_called_once_with("mock/qwen-omni", revision="abc123")
