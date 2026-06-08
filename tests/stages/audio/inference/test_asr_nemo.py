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
from nemo_curator.tasks import AudioBatch


class TestAsrNeMoStage:
    """Test suite for TestAsrInference."""

    def test_stage_properties(self) -> None:
        """Test stage properties."""
        stage = InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2")
        assert stage.name == "ASR_inference"
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], ["audio_filepath", "pred_text"])

    def test_stage_initialization(self) -> None:
        """Test stage initialization with different parameters."""
        # Test with input_audio_path
        stage = InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2")
        assert stage.filepath_key == "audio_filepath"
        assert stage.pred_text_key == "pred_text"

        # Test with audio_limit
        stage = InferenceAsrNemoStage(
            model_name="nvidia/parakeet-tdt-0.6b-v2",
        )
        assert stage.batch_size == 16

    def test_process_success(self) -> None:
        """Test process method with successful file discovery."""

        with patch.object(InferenceAsrNemoStage, "transcribe", return_value=["the cat", "set on a mat"]):
            stage = InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2")

            file_paths = AudioBatch(
                data=[{"audio_filepath": "/test/audio1.wav"}, {"audio_filepath": "/test/audio2.mp3"}]
            )

            stage.setup_on_node()
            stage.setup()
            result = stage.process(file_paths)
            assert isinstance(result, AudioBatch)
            assert len(result.data) == 2
            assert all(isinstance(task, dict) for task in result.data)

            assert result.task_id == "task_id_nvidia/parakeet-tdt-0.6b-v2"
            assert result.dataset_name == "nvidia/parakeet-tdt-0.6b-v2_inference"

            # Check that the audio objects are created correctly
            assert isinstance(result.data[0], dict)
            assert isinstance(result.data[1], dict)
            assert result.data[0][result.filepath_key] == "/test/audio1.wav"
            assert result.data[0]["pred_text"] == "the cat"
            assert result.data[1][result.filepath_key] == "/test/audio2.mp3"
            assert result.data[1]["pred_text"] == "set on a mat"

    def test_transcribe_tuple_outputs_hypothesis(self) -> None:
        """Transcribe handles tuple (hyps, all_hyps) where hyps is list[list[obj.text]]."""

        class Hypo:
            def __init__(self, text: str) -> None:
                self.text = text

        class DummyModel:
            def transcribe(self, _files: list[str]) -> tuple[list[list[Hypo]], None]:
                hyps = [[Hypo("alpha")], [Hypo("beta")]]
                all_hyps = None
                return (hyps, all_hyps)

        stage = InferenceAsrNemoStage(model_name="dummy-model", asr_model=DummyModel())
        outputs = stage.transcribe(["/a.wav", "/b.wav"])
        assert outputs == ["alpha", "beta"]

    def test_transcribe_nested_list_of_strings(self) -> None:
        """Transcribe handles list[list[str]] by taking the first element from each inner list."""

        class DummyModel:
            def transcribe(self, _files: list[str]) -> list[list[str]]:
                return [["foo"], ["bar"]]

        stage = InferenceAsrNemoStage(model_name="dummy-model", asr_model=DummyModel())
        outputs = stage.transcribe(["/a.wav", "/b.wav"])
        assert outputs == ["foo", "bar"]

    def test_transcribe_list_of_objects_with_text(self) -> None:
        """Transcribe handles list[obj] where each obj has a `text` attribute."""

        class Hypo:
            def __init__(self, text: str) -> None:
                self.text = text

        class DummyModel:
            def transcribe(self, _files: list[str]) -> list[Hypo]:
                return [Hypo("x"), Hypo("y")]

        stage = InferenceAsrNemoStage(model_name="dummy-model", asr_model=DummyModel())
        outputs = stage.transcribe(["/a.wav", "/b.wav"])
        assert outputs == ["x", "y"]
