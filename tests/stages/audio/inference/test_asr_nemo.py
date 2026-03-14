# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import patch

from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.tasks import AudioEntry


class TestAsrNeMoStage:
    """Test suite for InferenceAsrNemoStage."""

    def test_stage_properties(self) -> None:
        stage = InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2")
        assert stage.name == "ASR_inference"
        assert stage.inputs() == ([], ["audio_filepath"])
        assert stage.outputs() == ([], ["audio_filepath", "pred_text"])

    def test_stage_initialization(self) -> None:
        stage = InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2")
        assert stage.filepath_key == "audio_filepath"
        assert stage.pred_text_key == "pred_text"
        assert stage.batch_size == 16

    def test_process_entry_success(self) -> None:
        with patch.object(InferenceAsrNemoStage, "transcribe", return_value=["the cat"]):
            stage = InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2")
            stage.setup_on_node()
            stage.setup()

            entry = AudioEntry(data={"audio_filepath": "/test/audio1.wav"})
            result = stage.process(entry)

            assert isinstance(result, AudioEntry)
            assert result.data["audio_filepath"] == "/test/audio1.wav"
            assert result.data["pred_text"] == "the cat"

    def test_process_batch_success(self) -> None:
        with patch.object(InferenceAsrNemoStage, "transcribe", return_value=["the cat", "sat on a mat"]):
            stage = InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2")
            stage.setup_on_node()
            stage.setup()

            tasks = [
                AudioEntry(data={"audio_filepath": "/test/audio1.wav"}, task_id="t1"),
                AudioEntry(data={"audio_filepath": "/test/audio2.mp3"}, task_id="t2"),
            ]
            results = stage.process_batch(tasks)

            assert len(results) == 2
            assert all(isinstance(r, AudioEntry) for r in results)
            assert results[0].data["pred_text"] == "the cat"
            assert results[1].data["pred_text"] == "sat on a mat"

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
