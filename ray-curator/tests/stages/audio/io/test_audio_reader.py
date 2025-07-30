"""Test suite for AudioReaderStage."""

import pathlib
from typing import TYPE_CHECKING
from unittest.mock import patch

from ray_curator.stages.audio.io.audio_reader import AudioReaderStage
from ray_curator.tasks import SpeechDataEntry, _EmptyTask

if TYPE_CHECKING:
    from unittest.mock import MagicMock


class TestAudioReaderStage:
    """Test suite for TestAudioReaderStage."""

    def test_stage_properties(self) -> None:
        """Test stage properties."""
        stage = AudioReaderStage(input_audio_path="/test/path")
        assert stage.name == "audio_reader"
        assert stage.inputs() == ([], [])
        assert stage.outputs() == (["data"], ["input_audio"])

    def test_stage_initialization(self) -> None:
        """Test stage initialization with different parameters."""
        # Test with input_audio_path
        stage = AudioReaderStage(input_audio_path="/test/path")
        assert stage.input_audio_path == "/test/path"
        assert stage.audio_limit == -1

        # Test with audio_limit
        stage = AudioReaderStage(input_audio_path="/test/path", audio_limit=10)
        assert stage.audio_limit == 10

    @patch("ray_curator.stages.audio.io.audio_reader.get_all_files_paths_under")
    def test_process_success(self, mock_get_files: "MagicMock") -> None:
        """Test process method with successful file discovery."""
        mock_get_files.return_value = ["/test/audio1.wav", "/test/audio2.mp3"]

        stage = AudioReaderStage(input_audio_path="/test/path", extensions=[".wav", ".mp3"])
        result = stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

        assert len(result) == 2
        assert all(isinstance(task, SpeechDataEntry) for task in result)
        assert result[0].task_id == "/test/audio1.wav_processed"
        assert result[1].task_id == "/test/audio2.mp3_processed"
        assert result[0].dataset_name == "/test/path"
        assert result[1].dataset_name == "/test/path"

        # Check that the audio objects are created correctly
        assert isinstance(result[0].data, dict)
        assert isinstance(result[1].data, dict)
        assert result[0].data[result[0].filepath_key] == pathlib.Path("/test/audio1.wav")
        assert result[1].data[result[1].filepath_key] == pathlib.Path("/test/audio2.mp3")

        mock_get_files.assert_called_once_with(
            "/test/path",
            recurse_subdirectories=True,
            keep_extensions=[".wav", ".mp3"],
        )
