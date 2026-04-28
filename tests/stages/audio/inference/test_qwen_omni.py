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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nemo_curator.stages.audio.inference.qwen_omni import InferenceQwenOmniStage
from nemo_curator.tasks import AudioTask


def _make_stage(followup: str | None = None) -> InferenceQwenOmniStage:
    stage = InferenceQwenOmniStage(
        model_id="mock/qwen-omni",
        followup_prompt=followup,
    )
    mock_model = MagicMock()
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


def test_basic_inference() -> None:
    stage = _make_stage()
    stage._model.generate.return_value = (["hello world"], [""])

    tasks = [_make_task()]
    results = stage.process_batch(tasks)

    assert results[0].data["qwen3_prediction_s1"] == "hello world"
    assert "waveform" not in results[0].data


def test_followup_prompt_stores_disfluency() -> None:
    stage = _make_stage(followup="Remove disfluencies")
    stage._model.generate.return_value = (["hello world"], ["hello world cleaned"])

    tasks = [_make_task()]
    results = stage.process_batch(tasks)

    assert results[0].data["qwen3_prediction_s1"] == "hello world"
    assert results[0].data["qwen3_prediction_s2"] == "hello world cleaned"


def test_keep_waveform_flag() -> None:
    stage = _make_stage()
    stage.keep_waveform = True
    stage._model.generate.return_value = (["text"], [""])

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
    stage._model.generate.return_value = (["text1", "text2"], ["", ""])

    tasks = [_make_task(), _make_task()]
    results = stage.process_batch(tasks)

    assert results[0].data["qwen3_prediction_s1"] == "text1"
    assert results[1].data["qwen3_prediction_s1"] == "text2"


def test_language_resolution() -> None:
    stage = _make_stage()
    stage._model.generate.return_value = (["hola"], [""])

    task = AudioTask(data={
        "waveform": np.zeros(16000, dtype=np.float32),
        "sample_rate": 16000,
        "source_lang": "es",
    })
    stage.process_batch([task])

    call_args = stage._model.generate.call_args
    languages = call_args[0][2]
    assert languages == ["Spanish"]


@patch("nemo_curator.stages.audio.inference.qwen_omni.snapshot_download")
def test_setup_on_node_downloads_weights(mock_download: MagicMock) -> None:
    stage = InferenceQwenOmniStage(model_id="mock/model")
    stage.setup_on_node()
    mock_download.assert_called_once_with("mock/model")


def test_inputs_outputs() -> None:
    stage = InferenceQwenOmniStage()
    _required, optional = stage.inputs()
    assert "waveform" in optional
    assert "sample_rate" in optional
