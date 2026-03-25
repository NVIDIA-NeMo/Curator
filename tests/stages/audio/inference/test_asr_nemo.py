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

import pytest

from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.tasks import AudioTask


class TestAsrNeMoStage:
    """Test suite for InferenceAsrNemoStage."""

    def test_stage_properties(self) -> None:
        stage = InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2")
        assert stage.name == "ASR_inference"
        assert stage.inputs() == ([], ["audio_filepath"])
        assert stage.outputs() == ([], ["audio_filepath", "pred_text"])

    def test_validate_input_valid(self) -> None:
        stage = InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2")
        assert stage.validate_input(AudioTask(data={"audio_filepath": "/a.wav"})) is True

    def test_validate_input_missing_filepath(self) -> None:
        stage = InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2")
        assert stage.validate_input(AudioTask(data={"text": "hello"})) is False

    def test_process_raises_not_implemented(self) -> None:
        stage = InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2")
        with pytest.raises(NotImplementedError, match="only supports process_batch"):
            stage.process(AudioTask(data={"audio_filepath": "/a.wav"}))

    def test_process_batch_raises_on_missing_filepath(self) -> None:
        with patch.object(InferenceAsrNemoStage, "transcribe", return_value=["x"]):
            stage = InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2")
            stage.setup_on_node()
            stage.setup()
            with pytest.raises(ValueError, match="missing required columns"):
                stage.process_batch([AudioTask(data={"text": "hello"})])

    def test_process_batch_single_entry(self) -> None:
        with patch.object(InferenceAsrNemoStage, "transcribe", return_value=["the cat"]):
            stage = InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2")
            stage.setup_on_node()
            stage.setup()

            entry = AudioTask(data={"audio_filepath": "/test/audio1.wav"})
            results = stage.process_batch([entry])

            assert len(results) == 1
            assert isinstance(results[0], AudioTask)
            assert results[0].data["audio_filepath"] == "/test/audio1.wav"
            assert results[0].data["pred_text"] == "the cat"

    def test_process_batch_success(self) -> None:
        with patch.object(InferenceAsrNemoStage, "transcribe", return_value=["the cat", "sat on a mat"]):
            stage = InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2")
            stage.setup_on_node()
            stage.setup()

            tasks = [
                AudioTask(data={"audio_filepath": "/test/audio1.wav"}, task_id="t1"),
                AudioTask(data={"audio_filepath": "/test/audio2.mp3"}, task_id="t2"),
            ]
            results = stage.process_batch(tasks)

            assert len(results) == 2
            assert all(isinstance(r, AudioTask) for r in results)
            assert results[0].data["pred_text"] == "the cat"
            assert results[1].data["pred_text"] == "sat on a mat"

    @patch("nemo_curator.stages.audio.inference.asr_nemo.nemo_asr")
    def test_setup_on_node_downloads_only(self, mock_nemo_asr: MagicMock) -> None:
        mock_nemo_asr.models.ASRModel.from_pretrained.return_value = "/cache/model.nemo"
        stage = InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2")
        stage.setup_on_node()
        mock_nemo_asr.models.ASRModel.from_pretrained.assert_called_once_with(
            model_name="nvidia/parakeet-tdt-0.6b-v2", return_model_file=True
        )
        assert stage.asr_model is None

    @patch("nemo_curator.stages.audio.inference.asr_nemo.nemo_asr")
    def test_setup_on_node_failure(self, mock_nemo_asr: MagicMock) -> None:
        mock_nemo_asr.models.ASRModel.from_pretrained.side_effect = Exception("network error")
        stage = InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2")
        with pytest.raises(RuntimeError, match="Failed to download"):
            stage.setup_on_node()

    def test_setup_on_node_skipped_when_model_provided(self) -> None:
        stage = InferenceAsrNemoStage(model_name="dummy", asr_model=MagicMock())
        stage.setup_on_node()

    def test_transcribe_tuple_outputs_hypothesis(self) -> None:
        class Hypo:
            def __init__(self, text: str) -> None:
                self.text = text

        class DummyModel:
            def transcribe(self, _files: list[str]) -> tuple[list[list[Hypo]], None]:
                hyps = [[Hypo("alpha")], [Hypo("beta")]]
                return (hyps, None)

        stage = InferenceAsrNemoStage(model_name="dummy-model", asr_model=DummyModel())
        outputs = stage.transcribe(["/a.wav", "/b.wav"])
        assert outputs == ["alpha", "beta"]

    def test_transcribe_nested_list_of_strings(self) -> None:
        class DummyModel:
            def transcribe(self, _files: list[str]) -> list[list[str]]:
                return [["foo"], ["bar"]]

        stage = InferenceAsrNemoStage(model_name="dummy-model", asr_model=DummyModel())
        outputs = stage.transcribe(["/a.wav", "/b.wav"])
        assert outputs == ["foo", "bar"]

    def test_transcribe_list_of_objects_with_text(self) -> None:
        class Hypo:
            def __init__(self, text: str) -> None:
                self.text = text

        class DummyModel:
            def transcribe(self, _files: list[str]) -> list[Hypo]:
                return [Hypo("x"), Hypo("y")]

        stage = InferenceAsrNemoStage(model_name="dummy-model", asr_model=DummyModel())
        outputs = stage.transcribe(["/a.wav", "/b.wav"])
        assert outputs == ["x", "y"]
