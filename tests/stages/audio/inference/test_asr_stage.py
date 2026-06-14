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

"""Tests for the generic ``ASRStage`` exercised against a mock ``ASRAdapter`` (no real model load)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.models.asr.base import ASRResult
from nemo_curator.stages.audio.inference.asr import ASRStage
from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy
from nemo_curator.tasks import AudioTask

_QWEN_ADAPTER_TARGET = "nemo_curator.models.asr.qwen_omni.QwenOmniASRAdapter"
_SR = 16000


def _make_stage(  # noqa: PLR0913
    *,
    disfluency_text_key: str | None = None,
    keep_waveform: bool = True,
    default_language: str | None = None,
    ideal_inference_segment_s: float = 2400.0,
    max_inference_duration_s: float | None = None,
    chunking_enabled: bool = False,
    batch_policy: BatchPolicy | None = None,
    batch_size: int = 32,
) -> ASRStage:
    """Build an ASRStage wired to a mock adapter (no real model load)."""
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/qwen-omni",
        pred_text_key="qwen3_prediction_s1",
        disfluency_text_key=disfluency_text_key,
        keep_waveform=keep_waveform,
        default_language=default_language,
        chunking_enabled=chunking_enabled,
        ideal_inference_segment_s=ideal_inference_segment_s,
        max_inference_duration_s=max_inference_duration_s,
        batch_policy=batch_policy,
        batch_size=batch_size,
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
# Enabled scheduler pre-slice + stitch-back
# ----------------------------------------------------------------------


def test_pre_slice_short_clip_passes_through_unchanged() -> None:
    """A clip under max_inference_duration_s yields one sub-chunk; no stitching."""
    stage = _make_stage(
        max_inference_duration_s=2400.0,
        chunking_enabled=True,
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
        ideal_inference_segment_s=30.0,
        max_inference_duration_s=30.0,
        chunking_enabled=True,
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
        ideal_inference_segment_s=30.0,
        max_inference_duration_s=30.0,
        chunking_enabled=True,
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
    assert stage.batch_task_cost(task) == 95.0


def test_pre_slice_stitch_back_joins_per_parent_with_single_space() -> None:
    """Stitch-back joins sub-chunk texts (and secondary texts) with a single space; one row per parent."""
    stage = _make_stage(
        disfluency_text_key="qwen3_prediction_s2",
        ideal_inference_segment_s=30.0,
        max_inference_duration_s=30.0,
        chunking_enabled=True,
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
        ideal_inference_segment_s=30.0,
        max_inference_duration_s=30.0,
        chunking_enabled=True,
        batch_policy=_chunking_policy(),
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
    results = BaseStageAdapter(stage).process_batch([task_partial, task_all_skip])

    assert results[0].data["qwen3_prediction_s1"] == "good"
    assert results[0].data.get("_skip_me") != "empty_audio"
    assert results[1].data["qwen3_prediction_s1"] == ""
    assert results[1].data["_skip_me"] == "empty_audio"


def test_enabled_scheduler_worker_metrics_count_dispatched_chunks() -> None:
    """Scheduler-ready worker metrics describe dispatched chunk tasks; stitch-back restores parent rows."""
    stage = _make_stage(
        ideal_inference_segment_s=30.0,
        max_inference_duration_s=30.0,
        chunking_enabled=True,
        batch_policy=_chunking_policy(),
    )
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="a"), ASRResult(text="b"), ASRResult(text="c"), ASRResult(text="d"),
    ]
    stage._log_metrics = MagicMock()  # type: ignore[method-assign]

    task_short = AudioTask(data={"waveform": np.zeros(_SR * 10, dtype=np.float32), "sample_rate": _SR})  # 1 chunk
    task_long = AudioTask(data={"waveform": np.zeros(_SR * 75, dtype=np.float32), "sample_rate": _SR})  # 3 chunks
    BaseStageAdapter(stage).process_batch([task_short, task_long])

    metrics = stage._log_metrics.call_args[0][0]
    assert metrics["utterances_input"] == 4.0
    assert metrics["utterances_processed"] == 4.0
    assert metrics["sub_chunks_generated"] == 4.0


def test_chunking_disabled_does_not_pre_slice_long_clip() -> None:
    """Disabling chunking sends one full item per parent, like current main backend batching."""
    stage = _make_stage(
        ideal_inference_segment_s=30.0,
        max_inference_duration_s=30.0,
        chunking_enabled=False,
        batch_policy=BatchPolicy(
            enabled=False,
            strategy="placeholder",
            buckets_sec=[],
            max_items_per_batch_by_bucket=[],
        ),
    )
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="full")]
    waveform = np.arange(_SR * 95, dtype=np.float32)
    task = AudioTask(data={"waveform": waveform, "sample_rate": _SR, "source_lang": "en"})

    stage.process_batch([task])

    items = stage._adapter.transcribe_batch.call_args[0][0]
    assert len(items) == 1
    assert items[0]["chunk_count"] == 1
    assert items[0]["chunk_idx"] == 0
    np.testing.assert_array_equal(items[0]["waveform"], waveform)


def test_chunking_enabled_without_batch_policy_slices_normal_flow() -> None:
    """Chunking is independent from scheduler bucketing and works in normal process_batch."""
    stage = _make_stage(
        ideal_inference_segment_s=30.0,
        max_inference_duration_s=30.0,
        chunking_enabled=True,
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


def test_chunking_enabled_normal_flow_caps_adapter_calls_by_batch_size() -> None:
    stage = _make_stage(
        ideal_inference_segment_s=30.0,
        max_inference_duration_s=30.0,
        chunking_enabled=True,
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


def test_prebucketed_chunk_batch_caps_adapter_calls_by_batch_size() -> None:
    """Scheduler-ready chunks keep bucket order but still respect the model-call cap."""
    stage = _make_stage(
        chunking_enabled=True,
        batch_policy=_chunking_policy(),
        batch_size=2,
    )
    stage._adapter.transcribe_batch.side_effect = [
        [ASRResult(text="chunk0"), ASRResult(text="chunk1")],
        [ASRResult(text="chunk2"), ASRResult(text="chunk3")],
    ]
    tasks = []
    for chunk_idx in range(4):
        task = _make_task(waveform_len=_SR * 30)
        task.data["_curator_asr_chunk_idx"] = chunk_idx
        task.data["_curator_asr_chunk_count"] = 4
        task.data["_curator_asr_parent_idx"] = chunk_idx
        task.data["_curator_asr_chunk_cost"] = 30.0
        tasks.append(task)

    results = stage.process_batch(tasks)

    assert [len(call.args[0]) for call in stage._adapter.transcribe_batch.call_args_list] == [2, 2]
    assert [result.data["qwen3_prediction_s1"] for result in results] == [
        "chunk0",
        "chunk1",
        "chunk2",
        "chunk3",
    ]


def test_prebucketed_chunk_batch_result_length_mismatch_raises() -> None:
    stage = _make_stage(
        chunking_enabled=True,
        batch_policy=_chunking_policy(),
    )
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="chunk0")]
    tasks = []
    for chunk_idx in range(2):
        task = _make_task(waveform_len=_SR * 30)
        task.data["_curator_asr_chunk_idx"] = chunk_idx
        task.data["_curator_asr_chunk_count"] = 2
        task.data["_curator_asr_parent_idx"] = chunk_idx
        task.data["_curator_asr_chunk_cost"] = 30.0
        tasks.append(task)

    with pytest.raises(RuntimeError, match=r"returned 1 results for 2 items"):
        stage.process_batch(tasks)


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
