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

from nemo_curator.models.asr.base import ASRResult
from nemo_curator.stages.audio.inference.asr import ASRStage
from nemo_curator.tasks import AudioTask

_QWEN_ADAPTER_TARGET = "nemo_curator.models.asr.qwen_omni.QwenOmniASRAdapter"
_SR = 16000


def _make_stage(  # noqa: PLR0913
    *,
    disfluency_text_key: str | None = None,
    default_language: str | None = None,
    batch_size: int = 32,
    reference_text_key: str | None = None,
    supported_language_codes: list[str] | None = None,
    skip_if_output_exists: bool = False,
) -> ASRStage:
    """Build an ASRStage wired to a mock adapter (no real model load)."""
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/qwen-omni",
        pred_text_key="qwen3_prediction_s1",
        disfluency_text_key=disfluency_text_key,
        default_language=default_language,
        batch_size=batch_size,
        reference_text_key=reference_text_key,
        supported_language_codes=supported_language_codes,
        skip_if_output_exists=skip_if_output_exists,
    )
    mock_adapter = MagicMock()
    stage._adapter = mock_adapter
    return stage


def _make_task(waveform_len: int = _SR, source_lang: str | None = "en") -> AudioTask:
    data: dict[str, object] = {
        "waveform": np.zeros(waveform_len, dtype=np.float32),
        "sampling_rate": _SR,
    }
    if source_lang is not None:
        data["source_lang"] = source_lang
    return AudioTask(data=data)


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
    assert "waveform" not in results[0].data


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


def test_skip_if_output_exists_reuses_prediction_and_only_infers_missing_rows() -> None:
    stage = _make_stage(skip_if_output_exists=True)
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="new prediction")]
    existing = _make_task()
    existing.data["qwen3_prediction_s1"] = "existing prediction"
    missing = _make_task()

    results = stage.process_batch([existing, missing])

    assert results == [existing, missing]
    assert existing.data["qwen3_prediction_s1"] == "existing prediction"
    assert missing.data["qwen3_prediction_s1"] == "new prediction"
    assert "waveform" not in existing.data
    assert "waveform" not in missing.data
    inferred_items = stage._adapter.transcribe_batch.call_args.args[0]
    assert len(inferred_items) == 1


def test_skip_if_output_exists_skips_entire_prefilled_batch() -> None:
    stage = _make_stage(skip_if_output_exists=True)
    tasks = [_make_task(), _make_task()]
    tasks[0].data["qwen3_prediction_s1"] = "first"
    tasks[1].data["qwen3_prediction_s1"] = "second"

    results = stage.process_batch(tasks)

    assert [task.data["qwen3_prediction_s1"] for task in results] == ["first", "second"]
    stage._adapter.transcribe_batch.assert_not_called()


def test_skip_if_output_exists_does_not_skip_empty_prediction() -> None:
    stage = _make_stage(skip_if_output_exists=True)
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="filled")]
    task = _make_task()
    task.data["qwen3_prediction_s1"] = ""

    result = stage.process_batch([task])

    assert result[0].data["qwen3_prediction_s1"] == "filled"
    stage._adapter.transcribe_batch.assert_called_once()


def test_adapter_result_length_mismatch_raises() -> None:
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="x")]  # 1 result
    with pytest.raises(RuntimeError, match=r"returned 1 results for 2 supported items"):
        stage.process_batch([_make_task(), _make_task()])


def test_language_resolution_from_task() -> None:
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hola")]

    task = AudioTask(
        data={
            "waveform": np.zeros(_SR, dtype=np.float32),
            "sampling_rate": _SR,
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
            "sampling_rate": _SR,
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
    assert "_skipme" not in results[0].data
    assert results[0].data["additional_notes"]["ASR_inference"] == "skipped (unsupported language: pl)"
    assert results[0].data["additional_notes"]["qwen3_prediction_s1"] == "lang_not_supported:pl"


def test_reference_text_key_is_passed_to_adapter_items() -> None:
    stage = _make_stage(reference_text_key="text")
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hello")]
    task = AudioTask(
        data={
            "waveform": np.zeros(_SR, dtype=np.float32),
            "sampling_rate": _SR,
            "source_lang": "en",
            "text": "reference transcript",
        }
    )

    stage.process_batch([task])

    items = stage._adapter.transcribe_batch.call_args[0][0]
    assert items[0]["reference_text"] == "reference transcript"


def test_inputs_outputs_single_turn() -> None:
    stage = ASRStage(adapter_target=_QWEN_ADAPTER_TARGET, model_id="mock/model")
    _required, optional_inputs = stage.inputs()
    assert "waveform" in optional_inputs
    assert "sampling_rate" in optional_inputs

    _required, optional_outputs = stage.outputs()
    assert "pred_text" in optional_outputs


def test_outputs_two_turn_includes_disfluency_key() -> None:
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/model",
        disfluency_text_key="pred_text_secondary",
    )
    _required, optional_outputs = stage.outputs()
    assert "pred_text_secondary" in optional_outputs


@pytest.mark.parametrize(
    ("result", "expected_reason"),
    [
        (ASRResult(text="", skipped=True), "empty_audio"),
        (ASRResult(text="", skipped=True, skip_reason="decode_failed"), "decode_failed"),
        (ASRResult(text="", skipped=True, extras={"skip_reason": "ignored"}), "empty_audio"),
    ],
)
def test_skipped_result_sets_typed_skip_reason(result: ASRResult, expected_reason: str) -> None:
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [result]
    results = stage.process_batch([_make_task()])
    assert results[0].data["_skipme"] == expected_reason


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


def test_setup_failure_cleans_partial_adapter_and_allows_retry() -> None:
    stage = ASRStage(adapter_target=_QWEN_ADAPTER_TARGET, model_id="mock/model")
    failed_adapter = MagicMock()
    failed_adapter.setup.side_effect = RuntimeError("engine init failed")
    working_adapter = MagicMock()
    fake_cls = MagicMock(side_effect=[failed_adapter, working_adapter])

    with patch("hydra.utils.get_class", return_value=fake_cls):
        with pytest.raises(RuntimeError, match="engine init failed"):
            stage.setup()

        assert stage._adapter is None
        failed_adapter.teardown.assert_called_once_with()

        stage.setup()

    assert stage._adapter is working_adapter
    working_adapter.setup.assert_called_once_with()
