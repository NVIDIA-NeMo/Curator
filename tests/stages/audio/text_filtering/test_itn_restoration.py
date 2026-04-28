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

from nemo_curator.stages.audio.text_filtering.itn_restoration import (
    ITNRestorationStage,
    _validate_itn_output,
)
from nemo_curator.tasks import AudioTask


class TestValidateItnOutput:
    def test_valid_output(self) -> None:
        ok, _reason = _validate_itn_output("fourteen dollars", "$14")
        assert ok is True

    def test_word_count_increase_rejected(self) -> None:
        ok, reason = _validate_itn_output("hello", "hello world extra words added")
        assert ok is False
        assert "word_count_increase" in reason

    def test_excessive_deletion_rejected(self) -> None:
        input_text = "one two three four five six seven eight nine ten"
        ok, reason = _validate_itn_output(input_text, "one")
        assert ok is False
        assert "excessive_deletion" in reason

    def test_empty_input_passes(self) -> None:
        ok, _ = _validate_itn_output("", "")
        assert ok is True


class TestITNRestorationStage:
    def _make_stage(self) -> ITNRestorationStage:
        stage = ITNRestorationStage(model_id="mock/model", prompt_text="Convert ITN: {text}")
        mock_llm = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = lambda msgs, **_kw: msgs[-1]["content"]
        stage._llm = mock_llm
        stage._tokenizer = mock_tokenizer
        stage._sampling_params = MagicMock()
        stage._system_prompt = "Convert ITN"
        return stage

    def test_process_raises_not_implemented(self) -> None:
        stage = self._make_stage()
        task = AudioTask(data={"pnc_text": "hello", "_skip_me": ""})
        with pytest.raises(NotImplementedError):
            stage.process(task)

    def test_empty_batch(self) -> None:
        stage = self._make_stage()
        assert stage.process_batch([]) == []

    def test_skipped_tasks_get_empty_output(self) -> None:
        stage = self._make_stage()
        tasks = [AudioTask(data={"pnc_text": "hello", "_skip_me": "Hallucination"})]
        results = stage.process_batch(tasks)
        assert results[0].data["itn_text"] == ""
        stage._llm.generate.assert_not_called()

    def test_empty_text_preserved(self) -> None:
        stage = self._make_stage()
        tasks = [AudioTask(data={"pnc_text": "", "_skip_me": ""})]
        results = stage.process_batch(tasks)
        assert results[0].data["itn_text"] == ""

    def test_valid_inference(self) -> None:
        stage = self._make_stage()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="$14")]
        stage._llm.generate.return_value = [mock_output]

        tasks = [AudioTask(data={"pnc_text": "fourteen dollars", "_skip_me": ""})]
        results = stage.process_batch(tasks)
        assert results[0].data["itn_text"] == "$14"

    def test_validation_rejects_hallucination(self) -> None:
        stage = self._make_stage()
        stage.enable_validation = True
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="completely hallucinated extra words added here now")]
        stage._llm.generate.return_value = [mock_output]

        tasks = [AudioTask(data={"pnc_text": "hello world", "_skip_me": ""})]
        results = stage.process_batch(tasks)
        assert results[0].data["itn_text"] == "hello world"
        assert results[0].data["itn_filtered"] != ""

    def test_validation_disabled_accepts_all(self) -> None:
        stage = self._make_stage()
        stage.enable_validation = False
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="any text here regardless of length or content")]
        stage._llm.generate.return_value = [mock_output]

        tasks = [AudioTask(data={"pnc_text": "hello", "_skip_me": ""})]
        results = stage.process_batch(tasks)
        assert results[0].data["itn_text"] == "any text here regardless of length or content"

    def test_model_not_initialized_raises(self) -> None:
        stage = ITNRestorationStage(model_id="mock/model", prompt_text="test")
        tasks = [AudioTask(data={"pnc_text": "hello", "_skip_me": ""})]
        with pytest.raises(RuntimeError, match="setup"):
            stage.process_batch(tasks)

    def test_mixed_batch(self) -> None:
        stage = self._make_stage()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="$14")]
        stage._llm.generate.return_value = [mock_output]

        tasks = [
            AudioTask(data={"pnc_text": "skip", "_skip_me": "Bad"}),
            AudioTask(data={"pnc_text": "fourteen dollars", "_skip_me": ""}),
            AudioTask(data={"pnc_text": "  ", "_skip_me": ""}),
        ]
        results = stage.process_batch(tasks)
        assert results[0].data["itn_text"] == ""
        assert results[1].data["itn_text"] == "$14"
        assert results[2].data["itn_text"] == "  "
