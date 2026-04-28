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

from nemo_curator.stages.audio.inference.qwen_asr import InferenceQwenASRStage
from nemo_curator.tasks import AudioTask


def _make_stage(run_only_if_key: str | None = None) -> InferenceQwenASRStage:
    stage = InferenceQwenASRStage(
        model_id="mock/qwen-asr",
        run_only_if_key=run_only_if_key,
    )
    mock_model = MagicMock()
    stage._model = mock_model
    return stage


def _make_task(skip_me: str = "") -> AudioTask:
    return AudioTask(data={
        "waveform": np.zeros(16000, dtype=np.float32),
        "sample_rate": 16000,
        "source_lang": "en",
        "_skip_me": skip_me,
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
    stage._model.generate.return_value = (["transcribed text"], ["English"])

    tasks = [_make_task()]
    results = stage.process_batch(tasks)

    assert results[0].data["qwen3_asr_prediction"] == "transcribed text"
    assert results[0].data["qwen3_asr_language"] == "English"
    assert "waveform" not in results[0].data


def test_run_only_if_key_filters() -> None:
    stage = _make_stage(run_only_if_key="_skip_me")
    stage._model.generate.return_value = (["recovered"], ["English"])

    tasks = [
        _make_task(skip_me="Hallucination:ngram"),
        _make_task(skip_me=""),
    ]
    results = stage.process_batch(tasks)

    assert results[0].data["qwen3_asr_prediction"] == "recovered"
    assert results[1].data["qwen3_asr_prediction"] == ""


def test_all_skipped_by_run_only_if() -> None:
    stage = _make_stage(run_only_if_key="_skip_me")

    tasks = [_make_task(skip_me=""), _make_task(skip_me="")]
    results = stage.process_batch(tasks)

    stage._model.generate.assert_not_called()
    assert "waveform" not in results[0].data


def test_model_not_initialized_raises() -> None:
    stage = InferenceQwenASRStage(model_id="mock/model")
    tasks = [_make_task()]
    with pytest.raises(RuntimeError, match="setup"):
        stage.process_batch(tasks)


def test_multi_task_batch() -> None:
    stage = _make_stage()
    stage._model.generate.return_value = (["text1", "text2"], ["en", "es"])

    tasks = [_make_task(), _make_task()]
    results = stage.process_batch(tasks)

    assert results[0].data["qwen3_asr_prediction"] == "text1"
    assert results[1].data["qwen3_asr_prediction"] == "text2"


@patch("huggingface_hub.snapshot_download")
def test_setup_on_node_downloads_weights(mock_download: MagicMock) -> None:
    stage = InferenceQwenASRStage(model_id="mock/model")
    stage.setup_on_node()
    mock_download.assert_called_once_with("mock/model")


def test_inputs_outputs() -> None:
    stage = InferenceQwenASRStage()
    _, optional = stage.inputs()
    assert "waveform" in optional
    assert "sample_rate" in optional
    _, out_optional = stage.outputs()
    assert "qwen3_asr_prediction" in out_optional
