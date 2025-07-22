"""Test suite for VideoReaderStage."""

import pathlib
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from ray_curator.stages.video.io.video_reader import VideoReaderStage
from ray_curator.tasks import Video, VideoTask, _EmptyTask

if TYPE_CHECKING:
    from unittest.mock import MagicMock


class TestVideoReaderStage:
    """Test suite for VideoReaderStage."""

    def test_stage_properties(self) -> None:
        """Test stage properties."""
        stage = VideoReaderStage(input_video_path="/test/path")
        assert stage.name == "video_reader"
        assert stage.inputs() == ([], [])
        assert stage.outputs() == (["data"], ["input_video"])

    def test_stage_initialization(self) -> None:
        """Test stage initialization with different parameters."""
        # Test with input_video_path
        stage = VideoReaderStage(input_video_path="/test/path")
        assert stage.input_video_path == "/test/path"
        assert stage.video_limit == -1

        # Test with video_limit
        stage = VideoReaderStage(input_video_path="/test/path", video_limit=10)
        assert stage.video_limit == 10

    @patch("ray_curator.stages.video.io.video_reader.get_all_files_paths_under")
    def test_process_success(self, mock_get_files: "MagicMock") -> None:
        """Test process method with successful file discovery."""
        mock_get_files.return_value = ["/test/video1.mp4", "/test/video2.avi"]

        stage = VideoReaderStage(input_video_path="/test/path")
        result = stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

        assert len(result) == 2
        assert all(isinstance(task, VideoTask) for task in result)
        assert result[0].task_id == "/test/video1.mp4_processed"
        assert result[1].task_id == "/test/video2.avi_processed"
        assert result[0].dataset_name == "/test/path"
        assert result[1].dataset_name == "/test/path"

        # Check that the video objects are created correctly
        assert isinstance(result[0].data, Video)
        assert isinstance(result[1].data, Video)
        assert result[0].data.input_video == pathlib.Path("/test/video1.mp4")
        assert result[1].data.input_video == pathlib.Path("/test/video2.avi")

        mock_get_files.assert_called_once_with(
            "/test/path",
            recurse_subdirectories=True,
            keep_extensions=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
        )

    @patch("ray_curator.stages.video.io.video_reader.get_all_files_paths_under")
    def test_process_with_video_limit(self, mock_get_files: "MagicMock") -> None:
        """Test process method with video limit applied."""
        mock_get_files.return_value = [
            "/test/video1.mp4",
            "/test/video2.avi",
            "/test/video3.mkv",
            "/test/video4.webm",
        ]

        stage = VideoReaderStage(input_video_path="/test/path", video_limit=2)
        result = stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

        assert len(result) == 2
        assert result[0].task_id == "/test/video1.mp4_processed"
        assert result[1].task_id == "/test/video2.avi_processed"

    @patch("ray_curator.stages.video.io.video_reader.get_all_files_paths_under")
    def test_process_no_files_found(self, mock_get_files: "MagicMock") -> None:
        """Test process method when no files are found."""
        mock_get_files.return_value = []

        stage = VideoReaderStage(input_video_path="/test/path")
        result = stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

        assert result == []

    def test_process_no_input_video_path(self) -> None:
        """Test process method raises error when input_video_path is None."""
        stage = VideoReaderStage(input_video_path=None)

        with pytest.raises(ValueError, match="input_video_path is not set"):
            stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

    @patch("ray_curator.stages.video.io.video_reader.get_all_files_paths_under")
    def test_process_file_extensions(self, mock_get_files: "MagicMock") -> None:
        """Test process method filters for correct file extensions."""
        mock_get_files.return_value = ["/test/video1.mp4"]

        stage = VideoReaderStage(input_video_path="/test/path")
        stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

        mock_get_files.assert_called_once_with(
            "/test/path",
            recurse_subdirectories=True,
            keep_extensions=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
        )

    def test_process_video_task_creation(self) -> None:
        """Test that VideoTask objects are created correctly."""
        video_path = pathlib.Path("/test/video.mp4")

        with patch("ray_curator.stages.video.io.video_reader.get_all_files_paths_under", return_value=[str(video_path)]):
            stage = VideoReaderStage(input_video_path="/test/path")
            result = stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

            assert len(result) == 1
            video_task = result[0]

            assert isinstance(video_task, VideoTask)
            assert video_task.task_id == "/test/video.mp4_processed"
            assert video_task.dataset_name == "/test/path"
            assert isinstance(video_task.data, Video)
            assert video_task.data.input_video == video_path

    @patch("ray_curator.stages.video.io.video_reader.get_all_files_paths_under")
    def test_process_string_to_pathlib_conversion(self, mock_get_files: "MagicMock") -> None:
        """Test that string file paths are converted to pathlib.Path objects."""
        mock_get_files.return_value = ["/test/video1.mp4", "/test/video2.avi"]

        stage = VideoReaderStage(input_video_path="/test/path")
        result = stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

        # Check that the paths are converted to pathlib.Path objects
        assert isinstance(result[0].data.input_video, pathlib.Path)
        assert isinstance(result[1].data.input_video, pathlib.Path)
        assert result[0].data.input_video == pathlib.Path("/test/video1.mp4")
        assert result[1].data.input_video == pathlib.Path("/test/video2.avi")

    @patch("ray_curator.stages.video.io.video_reader.get_all_files_paths_under")
    def test_process_unlimited_video_limit(self, mock_get_files: "MagicMock") -> None:
        """Test process method with unlimited video limit (default -1)."""
        mock_get_files.return_value = [
            "/test/video1.mp4",
            "/test/video2.avi",
            "/test/video3.mkv",
            "/test/video4.webm",
        ]

        stage = VideoReaderStage(input_video_path="/test/path", video_limit=-1)
        result = stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

        # Should process all files when video_limit is -1
        assert len(result) == 4

    @patch("ray_curator.stages.video.io.video_reader.get_all_files_paths_under")
    @patch("ray_curator.stages.video.io.video_reader.logger")
    def test_process_logging(self, mock_logger: "MagicMock", mock_get_files: "MagicMock") -> None:
        """Test process method logs appropriate messages."""
        mock_get_files.return_value = ["/test/video1.mp4", "/test/video2.avi"]

        stage = VideoReaderStage(input_video_path="/test/path")
        stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

        # Check that info is logged about files found
        mock_logger.info.assert_called_with("Found 2 files under /test/path")

    @patch("ray_curator.stages.video.io.video_reader.get_all_files_paths_under")
    @patch("ray_curator.stages.video.io.video_reader.logger")
    def test_process_logging_with_limit(self, mock_logger: "MagicMock", mock_get_files: "MagicMock") -> None:
        """Test process method logs appropriate messages when limit is applied."""
        mock_get_files.return_value = [
            "/test/video1.mp4",
            "/test/video2.avi",
            "/test/video3.mkv",
            "/test/video4.webm",
        ]

        stage = VideoReaderStage(input_video_path="/test/path", video_limit=2)
        stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

        # Check that info is logged about files found and limit applied
        mock_logger.info.assert_any_call("Found 4 files under /test/path")
        mock_logger.info.assert_any_call("Using first 2 files under /test/path since video_limit is set to 2")
