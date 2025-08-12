"""Test suite for InferenceAsrNemoStage."""

import os

from ray_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage


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
        assert stage.name == "audio_inference"
        assert stage.inputs() == ([], [])
        assert stage.outputs() == (["data"], ["audio_filepath", "text"])

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
