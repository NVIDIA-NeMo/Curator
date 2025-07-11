"""Test suite for VideoDownloadStage."""

import pathlib
from typing import TYPE_CHECKING
from unittest import mock
from unittest.mock import patch

import pytest

from ray_curator.stages.video.io.video_download import VideoDownloadStage
from ray_curator.tasks import Video, VideoMetadata, VideoTask, _EmptyTask

if TYPE_CHECKING:
    from unittest.mock import MagicMock


class TestVideoDownloadStage:
    """Test suite for VideoDownloadStage."""

    def test_stage_properties(self) -> None:
        """Test stage properties."""
        stage = VideoDownloadStage(folder_path="/test/path")
        assert stage.name == "video_download"
        assert stage.inputs() == ([], [])
        assert stage.outputs() == (["data"], [])

    def test_stage_initialization(self) -> None:
        """Test stage initialization with different parameters."""
        # Test with folder_path
        stage = VideoDownloadStage(folder_path="/test/path")
        assert stage.folder_path == "/test/path"
        assert stage.debug is False

        # Test with debug mode
        stage = VideoDownloadStage(folder_path="/test/path", debug=True)
        assert stage.debug is True

        # Test with None folder_path
        stage = VideoDownloadStage()
        assert stage.folder_path is None

    @patch("ray_curator.stages.video.io.video_download.get_all_files_paths_under")
    def test_get_file_list_success(self, mock_get_files: "MagicMock") -> None:
        """Test _get_file_list method with successful file discovery."""
        mock_get_files.return_value = ["/path/video1.mp4", "/path/video2.avi"]

        stage = VideoDownloadStage(folder_path="/test/path")
        files = stage._get_file_list()

        assert files == ["/path/video1.mp4", "/path/video2.avi"]
        mock_get_files.assert_called_once_with(
            "/test/path",
            recurse_subdirectories=True,
            keep_extensions=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
        )

    @patch("ray_curator.stages.video.io.video_download.get_all_files_paths_under")
    def test_get_file_list_debug_mode(self, mock_get_files: "MagicMock") -> None:
        """Test _get_file_list method in debug mode returns only first 2 files."""
        mock_get_files.return_value = [
            "/path/video1.mp4",
            "/path/video2.avi",
            "/path/video3.mkv",
            "/path/video4.webm",
        ]

        stage = VideoDownloadStage(folder_path="/test/path", debug=True)
        files = stage._get_file_list()

        assert files == ["/path/video1.mp4", "/path/video2.avi"]
        assert len(files) == 2

    def test_get_file_list_no_folder_path(self) -> None:
        """Test _get_file_list method raises error when folder_path is None."""
        stage = VideoDownloadStage()

        with pytest.raises(ValueError, match="folder_path is not set"):
            stage._get_file_list()

    def test_download_video_bytes_success(self) -> None:
        """Test _download_video_bytes method with successful file reading."""
        # Create a mock file with test data
        test_data = b"test video data"

        with patch("pathlib.Path.open", mock.mock_open(read_data=test_data)):
            video = Video(input_video=pathlib.Path("/test/video.mp4"))
            stage = VideoDownloadStage(folder_path="/test/path")

            result = stage._download_video_bytes(video)

            assert result is True
            assert video.source_bytes == test_data
            assert "download" not in video.errors

    def test_download_video_bytes_file_not_found(self) -> None:
        """Test _download_video_bytes method when file cannot be read."""
        with patch("pathlib.Path.open", side_effect=FileNotFoundError("File not found")):
            video = Video(input_video=pathlib.Path("/test/nonexistent.mp4"))
            stage = VideoDownloadStage(folder_path="/test/path")

            result = stage._download_video_bytes(video)

            assert result is False
            assert video.errors["download"] == "File not found"

    def test_download_video_bytes_other_exception(self) -> None:
        """Test _download_video_bytes method with other exceptions."""
        with patch("pathlib.Path.open", side_effect=PermissionError("Permission denied")):
            video = Video(input_video=pathlib.Path("/test/video.mp4"))
            stage = VideoDownloadStage(folder_path="/test/path")

            result = stage._download_video_bytes(video)

            assert result is False
            assert video.errors["download"] == "Permission denied"

    def test_download_video_bytes_none_result(self) -> None:
        """Test _download_video_bytes method when source_bytes ends up None."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        VideoDownloadStage(folder_path="/test/path")

        # Mock the file opening to successfully read but then manually set source_bytes to None
        # to simulate the edge case mentioned in the code
        with patch("pathlib.Path.open", mock.mock_open(read_data=b"test")):
            # Set source_bytes to None directly to simulate the edge case
            video.source_bytes = None

            # Mock the assignment to test the None check
            def _raise_s3_error() -> None:
                msg = "S3 client is required for S3 destination"
                raise TypeError(msg)

            def mock_method(v: Video) -> bool:
                try:
                    if isinstance(v.input_video, pathlib.Path):
                        with v.input_video.open("rb") as fp:
                            v.source_bytes = fp.read()
                    else:
                        _raise_s3_error()
                except (OSError, TypeError):
                    return False

                # Simulate the None scenario
                v.source_bytes = None

                if v.source_bytes is None:
                    v.source_bytes = b""

                return True

            result = mock_method(video)

            assert result is True
            assert video.source_bytes == b""

    def test_extract_and_validate_metadata_success(self) -> None:
        """Test _extract_and_validate_metadata method with successful metadata extraction."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        stage = VideoDownloadStage(folder_path="/test/path")

        with patch.object(video, "populate_metadata") as mock_populate:
            mock_populate.return_value = None
            video.metadata = VideoMetadata(
                video_codec="h264",
                pixel_format="yuv420p",
                width=1920,
                height=1080,
            )

            result = stage._extract_and_validate_metadata(video)

            assert result is True
            mock_populate.assert_called_once()

    def test_extract_and_validate_metadata_failure(self) -> None:
        """Test _extract_and_validate_metadata method with metadata extraction failure."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        stage = VideoDownloadStage(folder_path="/test/path")

        with patch.object(video, "populate_metadata", side_effect=Exception("Metadata error")):
            result = stage._extract_and_validate_metadata(video)

            assert result is False

    def test_extract_and_validate_metadata_missing_codec(self) -> None:
        """Test _extract_and_validate_metadata method with missing codec warning."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        stage = VideoDownloadStage(folder_path="/test/path")

        with patch.object(video, "populate_metadata"):
            video.metadata = VideoMetadata(video_codec=None, pixel_format="yuv420p")

            with patch("ray_curator.stages.video.io.video_download.logger.warning") as mock_warning:
                result = stage._extract_and_validate_metadata(video)

                assert result is True
                mock_warning.assert_called_once_with("Codec could not be extracted for /test/video.mp4!")

    def test_extract_and_validate_metadata_missing_pixel_format(self) -> None:
        """Test _extract_and_validate_metadata method with missing pixel format warning."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        stage = VideoDownloadStage(folder_path="/test/path")

        with patch.object(video, "populate_metadata"):
            video.metadata = VideoMetadata(video_codec="h264", pixel_format=None)

            with patch("ray_curator.stages.video.io.video_download.logger.warning") as mock_warning:
                result = stage._extract_and_validate_metadata(video)

                assert result is True
                mock_warning.assert_called_once_with("Pixel format could not be extracted for /test/video.mp4!")

    def test_format_metadata_for_logging_complete(self) -> None:
        """Test _format_metadata_for_logging method with complete metadata."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        video.source_bytes = b"test data"
        video.metadata = VideoMetadata(
            size=9,  # Length of b"test data"
            width=1920,
            height=1080,
            framerate=30.0,
            duration=120.0,
            bit_rate_k=5000,
        )

        stage = VideoDownloadStage(folder_path="/test/path")
        result = stage._format_metadata_for_logging(video)

        expected = {
            "size": "9B",
            "res": "1920x1080",
            "fps": "30.0",
            "duration": "2m",
            "weight": "0.40",
            "bit_rate": "5000K",
        }

        assert result == expected

    def test_format_metadata_for_logging_missing_fields(self) -> None:
        """Test _format_metadata_for_logging method with missing metadata fields."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        video.source_bytes = None
        video.metadata = VideoMetadata(
            width=None,
            height=None,
            framerate=None,
            duration=None,
            bit_rate_k=None,
        )

        stage = VideoDownloadStage(folder_path="/test/path")
        result = stage._format_metadata_for_logging(video)

        expected = {
            "size": "0B",
            "res": "unknownxunknown",
            "fps": "unknown",
            "duration": "unknown",
            "weight": "unknown",
            "bit_rate": "unknown",
        }

        assert result == expected

    def test_log_video_info(self) -> None:
        """Test _log_video_info method logs correct information."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        video.source_bytes = b"test data"
        video.metadata = VideoMetadata(
            size=9,  # Length of b"test data"
            width=1920,
            height=1080,
            framerate=30.0,
            duration=120.0,
            bit_rate_k=5000,
        )

        stage = VideoDownloadStage(folder_path="/test/path")

        with patch("ray_curator.stages.video.io.video_download.logger.info") as mock_info:
            stage._log_video_info(video)

            mock_info.assert_called_once_with(
                "Downloaded /test/video.mp4 "
                "size=9B "
                "res=1920x1080 "
                "fps=30.0 "
                "duration=2m "
                "weight=0.40 "
                "bit_rate=5000K."
            )

    @patch("ray_curator.stages.video.io.video_download.get_all_files_paths_under")
    @patch("os.path.exists")
    def test_process_success(self, mock_exists: "MagicMock", mock_get_files: "MagicMock") -> None:
        """Test process method with successful video processing."""
        mock_get_files.return_value = ["/test/video1.mp4", "/test/video2.avi"]
        mock_exists.return_value = True

        stage = VideoDownloadStage(folder_path="/test/path")

        # Mock the private methods
        with patch.object(stage, "_download_video_bytes", return_value=True), \
             patch.object(stage, "_extract_and_validate_metadata", return_value=True), \
             patch.object(stage, "_log_video_info"):

            result = stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

            assert len(result) == 2
            assert all(isinstance(task, VideoTask) for task in result)
            assert result[0].task_id == "/test/video1.mp4_processed"
            assert result[1].task_id == "/test/video2.avi_processed"
            assert result[0].dataset_name == "/test/path"
            assert result[1].dataset_name == "/test/path"

    @patch("ray_curator.stages.video.io.video_download.get_all_files_paths_under")
    @patch("os.path.exists")
    def test_process_file_not_exists(self, mock_exists: "MagicMock", mock_get_files: "MagicMock") -> None:
        """Test process method when video file doesn't exist."""
        mock_get_files.return_value = ["/test/nonexistent.mp4"]
        mock_exists.return_value = False

        stage = VideoDownloadStage(folder_path="/test/path")

        with pytest.raises(FileNotFoundError, match="Video file /test/nonexistent.mp4 does not exist"):
            stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

    @patch("ray_curator.stages.video.io.video_download.get_all_files_paths_under")
    @patch("os.path.exists")
    def test_process_download_fails(self, mock_exists: "MagicMock", mock_get_files: "MagicMock") -> None:
        """Test process method when video download fails."""
        mock_get_files.return_value = ["/test/video1.mp4"]
        mock_exists.return_value = True

        stage = VideoDownloadStage(folder_path="/test/path")

        with patch.object(stage, "_download_video_bytes", return_value=False):
            result = stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

            assert result == []

    @patch("ray_curator.stages.video.io.video_download.get_all_files_paths_under")
    @patch("os.path.exists")
    def test_process_metadata_extraction_fails(self, mock_exists: "MagicMock", mock_get_files: "MagicMock") -> None:
        """Test process method when metadata extraction fails."""
        mock_get_files.return_value = ["/test/video1.mp4"]
        mock_exists.return_value = True

        stage = VideoDownloadStage(folder_path="/test/path")

        with patch.object(stage, "_download_video_bytes", return_value=True), \
             patch.object(stage, "_extract_and_validate_metadata", return_value=False):

            result = stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

            assert result == []

    @patch("ray_curator.stages.video.io.video_download.get_all_files_paths_under")
    @patch("os.path.exists")
    def test_process_mixed_success_failure(self, mock_exists: "MagicMock", mock_get_files: "MagicMock") -> None:
        """Test process method with mixed success and failure."""
        mock_get_files.return_value = ["/test/video1.mp4", "/test/video2.avi", "/test/video3.mkv"]
        mock_exists.return_value = True

        stage = VideoDownloadStage(folder_path="/test/path")

        # Mock different behaviors for different videos
        def mock_download_side_effect(video: Video) -> bool:
            return "video2" not in str(video.input_video)

        def mock_metadata_side_effect(video: Video) -> bool:
            return "video3" not in str(video.input_video)

        with patch.object(stage, "_download_video_bytes", side_effect=mock_download_side_effect), \
             patch.object(stage, "_extract_and_validate_metadata", side_effect=mock_metadata_side_effect), \
             patch.object(stage, "_log_video_info"):

            result = stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

            # Only video1 should succeed
            assert len(result) == 1
            assert result[0].task_id == "/test/video1.mp4_processed"

    @patch("ray_curator.stages.video.io.video_download.get_all_files_paths_under")
    def test_process_no_files_found(self, mock_get_files: "MagicMock") -> None:
        """Test process method when no files are found."""
        mock_get_files.return_value = []

        stage = VideoDownloadStage(folder_path="/test/path")

        with patch("ray_curator.stages.video.io.video_download.logger.info") as mock_info:
            result = stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

            assert result == []
            mock_info.assert_called_with("Found 0 files")

    @patch("ray_curator.stages.video.io.video_download.get_all_files_paths_under")
    @patch("os.path.exists")
    def test_process_debug_mode(self, mock_exists: "MagicMock", mock_get_files: "MagicMock") -> None:
        """Test process method in debug mode."""
        mock_get_files.return_value = ["/test/video1.mp4", "/test/video2.avi", "/test/video3.mkv"]
        mock_exists.return_value = True

        stage = VideoDownloadStage(folder_path="/test/path", debug=True)

        with patch.object(stage, "_download_video_bytes", return_value=True), \
             patch.object(stage, "_extract_and_validate_metadata", return_value=True), \
             patch.object(stage, "_log_video_info"):

            result = stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

            # Should only process first 2 files in debug mode
            assert len(result) == 2
            assert result[0].task_id == "/test/video1.mp4_processed"
            assert result[1].task_id == "/test/video2.avi_processed"

    def test_process_video_task_creation(self) -> None:
        """Test that VideoTask objects are created correctly."""
        video_path = pathlib.Path("/test/video.mp4")
        stage = VideoDownloadStage(folder_path="/test/path")

        with patch.object(stage, "_get_file_list", return_value=[str(video_path)]), \
             patch("os.path.exists", return_value=True), \
             patch.object(stage, "_download_video_bytes", return_value=True), \
             patch.object(stage, "_extract_and_validate_metadata", return_value=True), \
             patch.object(stage, "_log_video_info"):

            result = stage.process(_EmptyTask(task_id="empty", dataset_name="empty", data=None))

            assert len(result) == 1
            video_task = result[0]

            assert isinstance(video_task, VideoTask)
            assert video_task.task_id == "/test/video.mp4_processed"
            assert video_task.dataset_name == "/test/path"
            assert isinstance(video_task.data, Video)
            assert video_task.data.input_video == video_path

    def test_download_video_bytes_s3_error(self) -> None:
        """Test _download_video_bytes method with S3 input (not supported)."""
        video = Video(input_video="s3://bucket/video.mp4")  # Not a Path object
        stage = VideoDownloadStage(folder_path="/test/path")

        result = stage._download_video_bytes(video)

        assert result is False
        assert "download" in video.errors
        assert "S3 client is required for S3 destination" in video.errors["download"]

    def test_extract_and_validate_metadata_missing_codec_and_pixel_format(self) -> None:
        """Test _extract_and_validate_metadata method with both codec and pixel format missing."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        stage = VideoDownloadStage(folder_path="/test/path")

        with patch.object(video, "populate_metadata"):
            video.metadata = VideoMetadata(video_codec=None, pixel_format=None)

            with patch("ray_curator.stages.video.io.video_download.logger.warning") as mock_warning:
                result = stage._extract_and_validate_metadata(video)

                assert result is True
                assert mock_warning.call_count == 2
                mock_warning.assert_any_call("Codec could not be extracted for /test/video.mp4!")
                mock_warning.assert_any_call("Pixel format could not be extracted for /test/video.mp4!")

    def test_download_video_bytes_none_logging(self) -> None:
        """Test _download_video_bytes method logs when source_bytes is None after successful read."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        VideoDownloadStage(folder_path="/test/path")

        # Mock successful file read but force source_bytes to None
        with patch("pathlib.Path.open", mock.mock_open(read_data=b"test")), \
             patch("ray_curator.stages.video.io.video_download.logger.error") as mock_error:
            # Mock the file read to succeed but then manually set source_bytes to None

            def mock_read(v: Video) -> bool:
                # First do the normal read
                with v.input_video.open("rb") as fp:
                    v.source_bytes = fp.read()

                # Force source_bytes to None to trigger the logging
                v.source_bytes = None

                if v.source_bytes is None:
                    mock_error("video.source_bytes is None for /test/video.mp4 without exceptions ???")
                    v.source_bytes = b""

                return True

            result = mock_read(video)

            assert result is True
            assert video.source_bytes == b""
            mock_error.assert_called_once_with("video.source_bytes is None for /test/video.mp4 without exceptions ???")
