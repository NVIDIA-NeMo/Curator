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

from unittest.mock import MagicMock

import pytest

from nemo_curator.stages.audio.text_filtering.pnc_restoration import PnCRestorationStage
from nemo_curator.tasks import AudioTask


def _make_stage_with_mock_model(batch_size: int = 64) -> tuple[PnCRestorationStage, MagicMock]:
    stage = PnCRestorationStage(model_id="mock/model", batch_size=batch_size)
    mock_model = MagicMock()
    stage._model = mock_model
    return stage, mock_model


def test_process_raises_not_implemented() -> None:
    stage = PnCRestorationStage(model_id="mock/model")
    stage._model = MagicMock()
    task = AudioTask(data={"cleaned_text": "hello", "_skip_me": ""})
    with pytest.raises(NotImplementedError):
        stage.process(task)


def test_empty_batch() -> None:
    stage, _ = _make_stage_with_mock_model()
    assert stage.process_batch([]) == []


def test_skipped_tasks_get_empty_output() -> None:
    stage, mock_model = _make_stage_with_mock_model()
    tasks = [
        AudioTask(data={"cleaned_text": "hello", "_skip_me": "Hallucination"}),
    ]
    results = stage.process_batch(tasks)
    assert results[0].data["pnc_text"] == ""
    mock_model.generate.assert_not_called()


def test_empty_text_preserved() -> None:
    stage, mock_model = _make_stage_with_mock_model()
    tasks = [
        AudioTask(data={"cleaned_text": "   ", "_skip_me": ""}),
    ]
    results = stage.process_batch(tasks)
    assert results[0].data["pnc_text"] == "   "
    mock_model.generate.assert_not_called()


def test_eligible_texts_sent_to_model() -> None:
    stage, mock_model = _make_stage_with_mock_model()
    mock_model.generate.return_value = ([True], ["Hello World."])
    tasks = [
        AudioTask(data={"cleaned_text": "hello world", "_skip_me": ""}),
    ]
    results = stage.process_batch(tasks)
    assert results[0].data["pnc_text"] == "Hello World."
    mock_model.generate.assert_called_once()


def test_batching_respects_batch_size() -> None:
    stage, mock_model = _make_stage_with_mock_model(batch_size=2)
    mock_model.generate.side_effect = [
        ([True, True], ["Hello.", "World."]),
        ([True], ["Foo."]),
    ]
    tasks = [
        AudioTask(data={"cleaned_text": "hello", "_skip_me": ""}),
        AudioTask(data={"cleaned_text": "world", "_skip_me": ""}),
        AudioTask(data={"cleaned_text": "foo", "_skip_me": ""}),
    ]
    results = stage.process_batch(tasks)
    assert mock_model.generate.call_count == 2
    assert results[0].data["pnc_text"] == "Hello."
    assert results[1].data["pnc_text"] == "World."
    assert results[2].data["pnc_text"] == "Foo."


def test_mixed_skip_and_eligible() -> None:
    stage, mock_model = _make_stage_with_mock_model()
    mock_model.generate.return_value = ([True], ["Good."])
    tasks = [
        AudioTask(data={"cleaned_text": "skip me", "_skip_me": "Bad"}),
        AudioTask(data={"cleaned_text": "good text", "_skip_me": ""}),
        AudioTask(data={"cleaned_text": "", "_skip_me": ""}),
    ]
    results = stage.process_batch(tasks)
    assert results[0].data["pnc_text"] == ""
    assert results[1].data["pnc_text"] == "Good."
    assert results[2].data["pnc_text"] == ""


def test_model_not_initialized_raises() -> None:
    stage = PnCRestorationStage(model_id="mock/model")
    tasks = [AudioTask(data={"cleaned_text": "hello", "_skip_me": ""})]
    with pytest.raises(RuntimeError, match="setup"):
        stage.process_batch(tasks)
