"""Test suite for AudioReaderStage."""

import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from ray_curator.stages.audio.io.asr_inference import AsrNemoInference, AsrNemoInferenceStage
from ray_curator.stages.io.reader.file_partitioning import FilePartitioningStage
from ray_curator.tasks import EmptyTask, SpeechObject

if TYPE_CHECKING:
    from unittest.mock import MagicMock


def get_e2e_test_data_path() -> str:
    """Returns path to e2e test data"""
    test_data_root = os.getenv("TEST_DATA_ROOT")
    if test_data_root:  # assume it's present locally
        return test_data_root
    else:
        raise ValueError


class TestAsrInference:
    """Test suite for TestAsrInference."""

    test_data_root = get_e2e_test_data_path()

    def test_stage_properties(self) -> None:
        """Test stage properties."""
        stage = AsrNemoInference(model_name="nvidia/parakeet-tdt-0.6b-v2", input_audio_path="/test/path")
        assert stage.name == "audio_inference"
        assert stage.inputs() == ([], [])
        assert stage.outputs() == (["data"], ["audio_filepath", "text"])

    def test_stage_initialization(self) -> None:
        """Test stage initialization with different parameters."""
        # Test with input_audio_path
        stage = AsrNemoInference(model_name="nvidia/parakeet-tdt-0.6b-v2", input_audio_path="/test/path")
        assert stage.input_audio_path == "/test/path"
        assert stage.audio_limit is None

        # Test with audio_limit
        stage = AsrNemoInference(
            model_name="nvidia/parakeet-tdt-0.6b-v2", input_audio_path="/test/path", audio_limit=10
        )
        assert stage.audio_limit == 10

    def test_composite_stage_behavior(self) -> None:
        """Test that AsrInference behaves correctly as a CompositeStage."""
        stage = AsrNemoInference(model_name="nvidia/parakeet-tdt-0.6b-v2", input_audio_path="/test/path")

        # Should be a CompositeStage
        from ray_curator.stages.base import CompositeStage

        assert isinstance(stage, CompositeStage)

        # Should have the correct generic type annotations
        # (This is more of a static analysis check, but we can verify the structure)
        stages = stage.decompose()
        assert len(stages) > 0
        assert all(hasattr(s, "process") for s in stages)

    @patch.object(FilePartitioningStage, "_get_file_list", return_value=["/test/audio1.wav", "/test/audio2.mp3"])
    @patch.object(AsrNemoInferenceStage, "transcribe", return_value=["the cat", "set on a mat"])
    def test_process_success(self, mock_get_files: "MagicMock", mock_transcribe: "MagicMock") -> None:
        """Test process method with successful file discovery."""
        _ = mock_get_files or mock_transcribe

        stage = AsrNemoInference(
            model_name="nvidia/parakeet-tdt-0.6b-v2", input_audio_path="/test/path", file_extensions=[".wav", ".mp3"]
        )
        stages = stage.decompose()

        stages[0].setup_on_node()
        stages[0].setup()
        path_list = stages[0].process(EmptyTask)

        stages[1].setup_on_node()
        stages[1].setup()
        result = stages[1].process_batch(path_list)

        assert len(result) == 2
        assert all(isinstance(task, SpeechObject) for task in result)
        assert result[0].task_id == "/test/audio1.wav_task_id"
        assert result[1].task_id == "/test/audio2.mp3_task_id"
        assert result[0].dataset_name == "nvidia/parakeet-tdt-0.6b-v2_inference"
        assert result[1].dataset_name == "nvidia/parakeet-tdt-0.6b-v2_inference"

        # Check that the audio objects are created correctly
        assert isinstance(result[0].data, dict)
        assert isinstance(result[1].data, dict)
        assert result[0].data[result[0].filepath_key] == "/test/audio1.wav"
        assert result[0].data["text"] == "the cat"
        assert result[1].data[result[1].filepath_key] == "/test/audio2.mp3"
        assert result[1].data["text"] == "set on a mat"

    @pytest.mark.skip(reason="Works locally only with TEST_DATA_ROOT, CI/CD is not ready")
    def test_file_inference(self) -> None:
        """Test file inference method with successful file discovery."""

        english_files = str(Path(self.test_data_root) / "english/librispeech/LibriSpeech/dev-clean/84/121123/")

        stage = AsrNemoInference(
            model_name="nvidia/parakeet-tdt-0.6b-v2", input_audio_path=english_files, file_extensions=[".wav", ".flac"]
        )
        stages = stage.decompose()

        stages[0].setup_on_node()
        stages[0].setup()
        path_list = stages[0].process(EmptyTask)

        stages[1].setup_on_node()
        stages[1].setup()
        result = stages[1].process_batch(path_list)

        assert len(result) == 29
        assert stages[1]._metrics["elapsed"] < 20  # 9.579694032669067/16.63609218597412 on A6000

        assert all(isinstance(task, SpeechObject) for task in result)
        assert os.path.split(result[0].task_id)[1] == "84-121123-0000.flac_task_id"
        assert os.path.split(result[1].task_id)[1] == "84-121123-0001.flac_task_id"
        assert result[0].dataset_name == "nvidia/parakeet-tdt-0.6b-v2_inference"
        assert result[1].dataset_name == "nvidia/parakeet-tdt-0.6b-v2_inference"
        assert result[0].data["text"] == "Go. Do you hear?"
        assert (
            result[1].data["text"]
            == "But in less than five minutes the staircase groaned beneath an extraordinary weight,"
        )
