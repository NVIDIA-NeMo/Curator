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

import pytest

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

    @pytest.mark.skip("Import NeMo without apex")
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
