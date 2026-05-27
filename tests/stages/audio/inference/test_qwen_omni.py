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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nemo_curator.models.qwen_omni import QwenOmni
from nemo_curator.stages.audio.inference.qwen_omni import InferenceQwenOmniStage
from nemo_curator.tasks import AudioTask


def _make_stage(followup: str | None = None) -> InferenceQwenOmniStage:
    stage = InferenceQwenOmniStage(
        model_id="mock/qwen-omni",
        followup_prompt=followup,
    )
    mock_model = MagicMock()
    mock_model.last_metrics = {}
    stage._model = mock_model
    return stage


def _make_task(waveform_len: int = 16000) -> AudioTask:
    return AudioTask(data={
        "waveform": np.zeros(waveform_len, dtype=np.float32),
        "sample_rate": 16000,
        "source_lang": "en",
    })


def test_process_raises_not_implemented() -> None:
    stage = _make_stage()
    with pytest.raises(NotImplementedError):
        stage.process(_make_task())


def test_empty_batch() -> None:
    stage = _make_stage()
    assert stage.process_batch([]) == []


def test_qwen_model_handles_empty_vllm_outputs() -> None:
    assert QwenOmni._first_output_text(SimpleNamespace(outputs=[])) == ""


def test_basic_inference() -> None:
    stage = _make_stage()
    stage._model.generate.return_value = (["hello world"], [""], set())

    tasks = [_make_task()]
    results = stage.process_batch(tasks)

    assert results[0].data["qwen3_prediction_s1"] == "hello world"
    assert "waveform" not in results[0].data


def test_followup_prompt_stores_disfluency() -> None:
    stage = _make_stage(followup="Remove disfluencies")
    stage._model.generate.return_value = (["hello world"], ["hello world cleaned"], set())

    tasks = [_make_task()]
    results = stage.process_batch(tasks)

    assert results[0].data["qwen3_prediction_s1"] == "hello world"
    assert results[0].data["qwen3_prediction_s2"] == "hello world cleaned"


def test_followup_prompt_file_stores_disfluency() -> None:
    stage = _make_stage()
    stage.followup_prompt_file = "prompt.md"
    stage._model.generate.return_value = (["hello world"], ["hello world cleaned"], set())

    tasks = [_make_task()]
    results = stage.process_batch(tasks)

    assert "qwen3_prediction_s2" in stage.outputs()[1]
    assert results[0].data["qwen3_prediction_s2"] == "hello world cleaned"


def test_keep_waveform_flag() -> None:
    stage = _make_stage()
    stage.keep_waveform = True
    stage._model.generate.return_value = (["text"], [""], set())

    tasks = [_make_task()]
    results = stage.process_batch(tasks)

    assert "waveform" in results[0].data


def test_model_not_initialized_raises() -> None:
    stage = InferenceQwenOmniStage(model_id="mock/model")
    tasks = [_make_task()]
    with pytest.raises(RuntimeError, match="setup"):
        stage.process_batch(tasks)


def test_multi_task_batch() -> None:
    stage = _make_stage()
    stage._model.generate.return_value = (["text1", "text2"], ["", ""], set())

    tasks = [_make_task(), _make_task()]
    results = stage.process_batch(tasks)

    assert results[0].data["qwen3_prediction_s1"] == "text1"
    assert results[1].data["qwen3_prediction_s1"] == "text2"


def test_language_resolution() -> None:
    stage = _make_stage()
    stage._model.generate.return_value = (["hola"], [""], set())

    task = AudioTask(data={
        "waveform": np.zeros(16000, dtype=np.float32),
        "sample_rate": 16000,
        "source_lang": "es",
    })
    stage.process_batch([task])

    call_args = stage._model.generate.call_args
    languages = call_args[0][2]
    assert languages == ["Spanish"]


def test_default_language_used_when_task_language_missing() -> None:
    stage = _make_stage()
    stage.default_language = "en"
    stage._model.generate.return_value = (["hello"], [""], set())

    task = AudioTask(data={
        "waveform": np.zeros(16000, dtype=np.float32),
        "sample_rate": 16000,
    })
    stage.process_batch([task])

    call_args = stage._model.generate.call_args
    languages = call_args[0][2]
    assert languages == ["English"]


@patch("nemo_curator.stages.audio.inference.qwen_omni.snapshot_download")
def test_setup_on_node_downloads_weights(mock_download: MagicMock) -> None:
    stage = InferenceQwenOmniStage(model_id="mock/model")
    stage.setup_on_node()
    mock_download.assert_called_once_with("mock/model")


@patch("nemo_curator.stages.audio.inference.qwen_omni.snapshot_download", side_effect=RuntimeError("missing auth"))
def test_setup_on_node_raises_by_default(mock_download: MagicMock) -> None:
    stage = InferenceQwenOmniStage(model_id="mock/model")

    with pytest.raises(RuntimeError, match="snapshot_download failed"):
        stage.setup_on_node()

    mock_download.assert_called_once_with("mock/model")


@patch("nemo_curator.stages.audio.inference.qwen_omni.snapshot_download", side_effect=RuntimeError("offline"))
def test_setup_on_node_can_warn_and_retry_later(mock_download: MagicMock) -> None:
    stage = InferenceQwenOmniStage(model_id="mock/model", prefetch_fail_on_error=False)

    stage.setup_on_node()

    mock_download.assert_called_once_with("mock/model")


def test_inputs_outputs() -> None:
    stage = InferenceQwenOmniStage()
    _required, optional = stage.inputs()
    assert "waveform" in optional
    assert "sample_rate" in optional
    _required, output_optional = stage.outputs()
    assert "_skip_me" in output_optional


def test_skipped_indices_set_skip_key() -> None:
    stage = _make_stage()
    stage._model.generate.return_value = ([""], [""], {0})

    results = stage.process_batch([_make_task()])

    assert results[0].data["_skip_me"] == "empty_audio"


def test_metrics_exclude_skipped_utterances() -> None:
    stage = _make_stage()
    stage._model.generate.return_value = (["text", ""], ["", ""], {1})
    stage._log_metrics = MagicMock()  # type: ignore[method-assign]

    stage.process_batch([_make_task(), _make_task()])

    metrics = stage._log_metrics.call_args[0][0]
    assert metrics["utterances_input"] == 2.0
    assert metrics["utterances_processed"] == 1.0
    assert metrics["utterances_skipped"] == 1.0
