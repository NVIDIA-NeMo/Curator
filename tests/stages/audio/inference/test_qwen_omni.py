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

Mirrors the behaviour the pre-split ``InferenceQwenOmniStage`` tests
covered. The stage delegates inference to ``self._adapter`` so tests
inject a mock adapter (matching the ``ASRAdapter`` protocol surface)
rather than the previous ``stage._model`` mock.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nemo_curator.adapters.asr.base import ASRResult
from nemo_curator.adapters.asr.qwen_omni import QwenOmniASRAdapter
from nemo_curator.stages.audio.inference.asr import ASRStage
from nemo_curator.tasks import AudioTask

_QWEN_ADAPTER_TARGET = "nemo_curator.adapters.asr.qwen_omni.QwenOmniASRAdapter"


def _make_stage(
    *,
    secondary_text_key: str | None = None,
    keep_waveform: bool = False,
    default_language: str | None = None,
) -> ASRStage:
    """Build an ASRStage wired to a mock adapter (no real model load)."""
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/qwen-omni",
        pred_text_key="qwen3_prediction_s1",
        secondary_text_key=secondary_text_key,
        keep_waveform=keep_waveform,
        default_language=default_language,
    )
    mock_adapter = MagicMock()
    mock_adapter.last_metrics = {}
    stage._adapter = mock_adapter
    return stage


def _make_task(waveform_len: int = 16000, source_lang: str | None = "en") -> AudioTask:
    data: dict[str, object] = {
        "waveform": np.zeros(waveform_len, dtype=np.float32),
        "sample_rate": 16000,
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
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hello world")]

    results = stage.process_batch([_make_task()])

    assert results[0].data["qwen3_prediction_s1"] == "hello world"
    assert "waveform" not in results[0].data  # default keep_waveform=False


def test_secondary_text_key_stores_disfluency() -> None:
    stage = _make_stage(secondary_text_key="qwen3_prediction_s2")
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="hello world", secondary_text="hello world cleaned"),
    ]

    results = stage.process_batch([_make_task()])

    assert results[0].data["qwen3_prediction_s1"] == "hello world"
    assert results[0].data["qwen3_prediction_s2"] == "hello world cleaned"
    # outputs() advertises the secondary key once it's enabled
    assert "qwen3_prediction_s2" in stage.outputs()[1]


def test_secondary_text_key_none_is_normalised_to_empty_string() -> None:
    """When secondary_text_key is set but the adapter returns None, write ""."""
    stage = _make_stage(secondary_text_key="qwen3_prediction_s2")
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="hello world", secondary_text=None),
    ]

    results = stage.process_batch([_make_task()])
    assert results[0].data["qwen3_prediction_s2"] == ""


def test_keep_waveform_flag() -> None:
    stage = _make_stage(keep_waveform=True)
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="text")]

    results = stage.process_batch([_make_task()])
    assert "waveform" in results[0].data


def test_adapter_not_initialized_raises() -> None:
    stage = ASRStage(adapter_target=_QWEN_ADAPTER_TARGET, model_id="mock/model")
    # No setup() called -> _adapter is None
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
    with pytest.raises(RuntimeError, match="returned 1 results for 2 tasks"):
        stage.process_batch([_make_task(), _make_task()])


# ----------------------------------------------------------------------
# Stage-level: language mapping (ISO code -> name)
# ----------------------------------------------------------------------


def test_language_resolution_from_task() -> None:
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hola")]

    task = AudioTask(data={
        "waveform": np.zeros(16000, dtype=np.float32),
        "sample_rate": 16000,
        "source_lang": "es",
    })
    stage.process_batch([task])

    items = stage._adapter.transcribe_batch.call_args[0][0]
    assert items[0]["language"] == "Spanish"


def test_default_language_used_when_task_language_missing() -> None:
    stage = _make_stage(default_language="en")
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hello")]

    task = AudioTask(data={
        "waveform": np.zeros(16000, dtype=np.float32),
        "sample_rate": 16000,
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


def test_outputs_two_turn_includes_secondary_key() -> None:
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/model",
        secondary_text_key="pred_text_secondary",
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


def test_metrics_model_alias_skips_already_emitted_keys() -> None:
    """A5-fix: adapter metrics that the stage already emits must NOT be
    re-aliased as ``model_<name>``."""
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="text")]
    stage._adapter.last_metrics = {
        "audio_duration_s": 999.0,         # already emitted by the stage
        "extra_diagnostic_metric": 42.0,   # not emitted by the stage
    }
    stage._log_metrics = MagicMock()  # type: ignore[method-assign]

    stage.process_batch([_make_task()])

    metrics = stage._log_metrics.call_args[0][0]
    assert "model_audio_duration_s" not in metrics  # de-duplicated
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
    stage.setup_on_node()  # should warn, not raise
    mock_download.assert_called_once_with("mock/model")


def test_adapter_target_required() -> None:
    with pytest.raises(ValueError, match="adapter_target is required"):
        ASRStage(model_id="mock/model")


def test_setup_uses_adapter_target_and_kwargs() -> None:
    """``setup()`` resolves adapter_target via hydra.utils.get_class and
    constructs the adapter with model_id+revision+**adapter_kwargs."""
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/model",
        revision="abc123",
        adapter_kwargs={"max_model_len": 8192},
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
    )
    fake_adapter.setup.assert_called_once_with()
    assert stage._adapter is fake_adapter


# ----------------------------------------------------------------------
# Adapter-level: QwenOmniASRAdapter helpers (no GPU, no vLLM required)
# ----------------------------------------------------------------------


def test_qwen_adapter_first_output_text_handles_empty_vllm_output() -> None:
    assert QwenOmniASRAdapter._first_output_text(SimpleNamespace(outputs=[])) == ""


def test_qwen_adapter_count_output_tokens_handles_empty_vllm_output() -> None:
    assert QwenOmniASRAdapter._count_output_tokens([SimpleNamespace(outputs=[])]) == 0.0


def test_qwen_adapter_transcribe_batch_packages_results() -> None:
    """The adapter's transcribe_batch contract: unpacks per-item dicts,
    calls into _generate, and packs (text, secondary_text, skipped) per item.
    """
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni", followup_prompt="refine")
    adapter._generate = MagicMock(  # type: ignore[method-assign]
        return_value=(
            ["text-a", "text-b", ""],
            ["refined-a", "", ""],
            {2},
        ),
    )
    items = [
        {"waveform": np.zeros(16000, dtype=np.float32), "sample_rate": 16000, "language": "English"},
        {"waveform": np.zeros(16000, dtype=np.float32), "sample_rate": 16000, "language": "English"},
        {"waveform": np.zeros(0, dtype=np.float32), "sample_rate": 16000, "language": None},
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
    """When followup_prompt is None, the adapter must set secondary_text=None
    on every result regardless of what _generate returns."""
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni", followup_prompt=None)
    adapter._generate = MagicMock(  # type: ignore[method-assign]
        return_value=(["text-a"], [""], set()),
    )
    results = adapter.transcribe_batch([
        {"waveform": np.zeros(16000, dtype=np.float32), "sample_rate": 16000},
    ])
    assert results[0].secondary_text is None
