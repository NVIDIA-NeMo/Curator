"""Test suite for InferenceAsrNemoStage."""

import os
from unittest.mock import patch

from ray_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from ray_curator.tasks import DocumentObject


def get_e2e_test_data_path() -> str:
    """Returns path to e2e test data"""
    test_data_root = os.getenv("TEST_DATA_ROOT")
    if test_data_root:  # assume it's present locally
        return test_data_root
    else:
        raise ValueError


class TestAsrNeMoStage:
    """Test suite for TestAsrInference."""

    test_data_root = get_e2e_test_data_path()

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

            path_list = [
                DocumentObject(data={"audio_filepath": "/test/audio1.wav"}),
                DocumentObject(data={"audio_filepath": "/test/audio2.mp3"}),
            ]

            stage.setup_on_node()
            stage.setup()
            result = stage.process_batch(path_list)

            assert len(result) == 2
            assert all(isinstance(task, DocumentObject) for task in result)
            assert result[0].task_id == "task_id_/test/audio1.wav"
            assert result[1].task_id == "task_id_/test/audio2.mp3"
            assert result[0].dataset_name == "nvidia/parakeet-tdt-0.6b-v2_inference"
            assert result[1].dataset_name == "nvidia/parakeet-tdt-0.6b-v2_inference"

            # Check that the audio objects are created correctly
            assert isinstance(result[0].data, dict)
            assert isinstance(result[1].data, dict)
            assert result[0].data[result[0].filepath_key] == "/test/audio1.wav"
            assert result[0].data["pred_text"] == "the cat"
            assert result[1].data[result[1].filepath_key] == "/test/audio2.mp3"
            assert result[1].data["pred_text"] == "set on a mat"
