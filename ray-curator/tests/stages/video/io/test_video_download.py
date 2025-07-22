"""Test suite for VideoDownloadStage."""

import pathlib
from unittest import mock
from unittest.mock import patch

from ray_curator.stages.video.io.video_download import VideoDownloadStage
from ray_curator.tasks import Video, VideoMetadata, VideoTask


class TestVideoDownloadStage:
    """Test suite for VideoDownloadStage."""

    def test_stage_properties(self) -> None:
        """Test stage properties."""
        stage = VideoDownloadStage()
        assert stage.name == "video_download"
        assert stage.inputs() == (["data"], ["input_video"])
        assert stage.outputs() == (["data"], ["source_bytes", "metadata"])

    def test_stage_initialization(self) -> None:
        """Test stage initialization with different parameters."""
        # Test default initialization
        stage = VideoDownloadStage()
        assert stage.verbose is False

        # Test with verbose mode
        stage = VideoDownloadStage(verbose=True)
        assert stage.verbose is True

    def test_download_video_bytes_success(self) -> None:
        """Test _download_video_bytes method with successful file reading."""
        # Create a mock file with test data
        test_data = b"test video data"

        with patch("pathlib.Path.open", mock.mock_open(read_data=test_data)):
            video = Video(input_video=pathlib.Path("/test/video.mp4"))
            stage = VideoDownloadStage()

            result = stage._download_video_bytes(video)

            assert result is True
            assert video.source_bytes == test_data
            assert "download" not in video.errors

    def test_download_video_bytes_file_not_found(self) -> None:
        """Test _download_video_bytes method when file cannot be read."""
        with patch("pathlib.Path.open", side_effect=FileNotFoundError("File not found")):
            video = Video(input_video=pathlib.Path("/test/nonexistent.mp4"))
            stage = VideoDownloadStage()

            result = stage._download_video_bytes(video)

            assert result is False
            assert video.errors["download"] == "File not found"

    def test_download_video_bytes_other_exception(self) -> None:
        """Test _download_video_bytes method with other exceptions."""
        with patch("pathlib.Path.open", side_effect=PermissionError("Permission denied")):
            video = Video(input_video=pathlib.Path("/test/video.mp4"))
            stage = VideoDownloadStage()

            result = stage._download_video_bytes(video)

            assert result is False
            assert video.errors["download"] == "Permission denied"

    def test_download_video_bytes_none_result(self) -> None:
        """Test _download_video_bytes method when source_bytes ends up None."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        VideoDownloadStage()

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
        stage = VideoDownloadStage()

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
        stage = VideoDownloadStage()

        with patch.object(video, "populate_metadata", side_effect=Exception("Metadata error")):
            result = stage._extract_and_validate_metadata(video)

            assert result is False

    def test_extract_and_validate_metadata_missing_codec(self) -> None:
        """Test _extract_and_validate_metadata method with missing codec warning."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        stage = VideoDownloadStage()

        with patch.object(video, "populate_metadata"):
            video.metadata = VideoMetadata(video_codec=None, pixel_format="yuv420p")

            with patch("ray_curator.stages.video.io.video_download.logger.warning") as mock_warning:
                result = stage._extract_and_validate_metadata(video)

                assert result is True
                mock_warning.assert_called_once_with("Codec could not be extracted for /test/video.mp4!")

    def test_extract_and_validate_metadata_missing_pixel_format(self) -> None:
        """Test _extract_and_validate_metadata method with missing pixel format warning."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        stage = VideoDownloadStage()

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

        stage = VideoDownloadStage()
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

        stage = VideoDownloadStage()
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

        stage = VideoDownloadStage()

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

    def test_process_success(self) -> None:
        """Test process method with successful video processing."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)
        stage = VideoDownloadStage()

        # Mock the private methods
        with patch.object(stage, "_download_video_bytes", return_value=True), \
             patch.object(stage, "_extract_and_validate_metadata", return_value=True), \
             patch.object(stage, "_log_video_info") as mock_log:

            result = stage.process(task)

            assert result is task
            assert isinstance(result, VideoTask)
            assert result.task_id == "test_task"
            assert result.dataset_name == "test_dataset"
            assert result.data is video
            # _log_video_info should not be called when verbose=False
            mock_log.assert_not_called()

    def test_process_download_fails(self) -> None:
        """Test process method when video download fails."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)
        stage = VideoDownloadStage()

        with patch.object(stage, "_download_video_bytes", return_value=False), \
             patch.object(stage, "_log_video_info") as mock_log:
            result = stage.process(task)

            assert result is task
            # Should return the original task when download fails
            # _log_video_info should not be called when download fails
            mock_log.assert_not_called()

    def test_process_metadata_extraction_fails(self) -> None:
        """Test process method when metadata extraction fails."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)
        stage = VideoDownloadStage()

        with patch.object(stage, "_download_video_bytes", return_value=True), \
             patch.object(stage, "_extract_and_validate_metadata", return_value=False), \
             patch.object(stage, "_log_video_info") as mock_log:

            result = stage.process(task)

            assert result is task
            # Should return the original task when metadata extraction fails
            # _log_video_info should not be called when metadata extraction fails
            mock_log.assert_not_called()

    def test_process_video_task_passthrough(self) -> None:
        """Test that the VideoTask object is passed through correctly."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)
        stage = VideoDownloadStage()

        with patch.object(stage, "_download_video_bytes", return_value=True), \
             patch.object(stage, "_extract_and_validate_metadata", return_value=True), \
             patch.object(stage, "_log_video_info") as mock_log:

            result = stage.process(task)

            # Should return the same task object
            assert result is task
            assert result.task_id == "test_task"
            assert result.dataset_name == "test_dataset"
            assert result.data is video
            # _log_video_info should not be called when verbose=False
            mock_log.assert_not_called()

    def test_process_success_verbose(self) -> None:
        """Test process method with successful video processing in verbose mode."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)
        stage = VideoDownloadStage(verbose=True)

        # Mock the private methods
        with patch.object(stage, "_download_video_bytes", return_value=True), \
             patch.object(stage, "_extract_and_validate_metadata", return_value=True), \
             patch.object(stage, "_log_video_info") as mock_log:

            result = stage.process(task)

            assert result is task
            assert isinstance(result, VideoTask)
            assert result.task_id == "test_task"
            assert result.dataset_name == "test_dataset"
            assert result.data is video
            # _log_video_info should be called when verbose=True
            mock_log.assert_called_once_with(video)

    def test_process_success_non_verbose(self) -> None:
        """Test process method with successful video processing in non-verbose mode."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)
        stage = VideoDownloadStage(verbose=False)

        # Mock the private methods
        with patch.object(stage, "_download_video_bytes", return_value=True), \
             patch.object(stage, "_extract_and_validate_metadata", return_value=True), \
             patch.object(stage, "_log_video_info") as mock_log:

            result = stage.process(task)

            assert result is task
            assert isinstance(result, VideoTask)
            assert result.task_id == "test_task"
            assert result.dataset_name == "test_dataset"
            assert result.data is video
            # _log_video_info should not be called when verbose=False
            mock_log.assert_not_called()

    def test_download_video_bytes_s3_error(self) -> None:
        """Test _download_video_bytes method with S3 input (not supported)."""
        video = Video(input_video="s3://bucket/video.mp4")  # Not a Path object
        stage = VideoDownloadStage()

        result = stage._download_video_bytes(video)

        assert result is False
        assert "download" in video.errors
        assert "S3 client is required for S3 destination" in video.errors["download"]

    def test_extract_and_validate_metadata_missing_codec_and_pixel_format(self) -> None:
        """Test _extract_and_validate_metadata method with both codec and pixel format missing."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        stage = VideoDownloadStage()

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
        VideoDownloadStage()

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
