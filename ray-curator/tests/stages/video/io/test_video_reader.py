"""Test suite for VideoReaderStage."""

import pathlib
from unittest import mock
from unittest.mock import patch

import pytest

from ray_curator.stages.video.io.video_reader import VideoReaderStage
from ray_curator.tasks.file_group import FileGroupTask
from ray_curator.tasks.video import Video, VideoMetadata, VideoTask


class TestVideoReaderStage:
    """Test suite for VideoReaderStage."""

    def test_stage_properties(self) -> None:
        """Test stage properties."""
        stage = VideoReaderStage()
        assert stage.name == "video_reader"
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], ["source_bytes", "metadata"])

    def test_stage_initialization(self) -> None:
        """Test stage initialization with different parameters."""
        # Test default initialization
        stage = VideoReaderStage()
        assert stage.verbose is False

        # Test with verbose mode
        stage = VideoReaderStage(verbose=True)
        assert stage.verbose is True

    def test_download_video_bytes_success(self) -> None:
        """Test _download_video_bytes method with successful file reading."""
        # Create a mock file with test data
        test_data = b"test video data"

        with patch("pathlib.Path.open", mock.mock_open(read_data=test_data)):
            video = Video(input_video=pathlib.Path("/test/video.mp4"))
            stage = VideoReaderStage()

            result = stage._download_video_bytes(video)

            assert result is True
            assert video.source_bytes == test_data
            assert "download" not in video.errors

    def test_download_video_bytes_file_not_found(self) -> None:
        """Test _download_video_bytes method when file cannot be read."""
        with patch("pathlib.Path.open", side_effect=FileNotFoundError("File not found")):
            video = Video(input_video=pathlib.Path("/test/nonexistent.mp4"))
            stage = VideoReaderStage()

            result = stage._download_video_bytes(video)

            assert result is False
            assert video.source_bytes is None
            assert "download" in video.errors
            assert "File not found" in video.errors["download"]

    def test_download_video_bytes_none_bytes_fallback(self) -> None:
        """Test _download_video_bytes handles None source_bytes case."""
        with patch("pathlib.Path.open", mock.mock_open(read_data=b"")):
            video = Video(input_video=pathlib.Path("/test/video.mp4"))
            # Simulate the actual behavior where source_bytes could become None
            stage = VideoReaderStage()

            # First call the method
            result = stage._download_video_bytes(video)
            # Then manually set to None to test the fallback
            video.source_bytes = None
            result = stage._download_video_bytes(video)

            assert result is True
            assert video.source_bytes == b""

    def test_download_video_bytes_s3_error(self) -> None:
        """Test _download_video_bytes raises error for S3 paths."""
        video = Video(input_video="s3://bucket/video.mp4")
        stage = VideoReaderStage()

        result = stage._download_video_bytes(video)

        assert result is False
        assert "S3 client is required" in video.errors["download"]

    def test_extract_and_validate_metadata_success(self) -> None:
        """Test _extract_and_validate_metadata with successful metadata extraction."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        video.metadata = VideoMetadata(
            video_codec="h264",
            pixel_format="yuv420p",
            width=1920,
            height=1080,
            framerate=30.0,
            duration=60.0,
            bit_rate_k=5000
        )

        # Mock populate_metadata to set metadata
        with patch.object(video, "populate_metadata", return_value=None):
            stage = VideoReaderStage()
            result = stage._extract_and_validate_metadata(video)

            assert result is True

    def test_extract_and_validate_metadata_exception(self) -> None:
        """Test _extract_and_validate_metadata handles exceptions gracefully."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))

        # Mock populate_metadata to raise an exception
        with (
            patch.object(video, "populate_metadata", side_effect=Exception("Metadata error")),
            patch("ray_curator.stages.video.io.video_reader.logger.warning") as mock_warn
        ):
            stage = VideoReaderStage()
            result = stage._extract_and_validate_metadata(video)

            assert result is False
            # The actual implementation logs a warning, doesn't store error in video.errors
            mock_warn.assert_called_with("Failed to extract metadata for /test/video.mp4: Metadata error")

    def test_log_video_info(self) -> None:
        """Test _log_video_info method logs video information."""
        video = Video(
            input_video=pathlib.Path("/test/video.mp4"),
            source_bytes=b"test data",
            metadata=VideoMetadata(
                video_codec="h264",
                pixel_format="yuv420p",
                width=1920,
                height=1080,
                framerate=30.0,
                duration=60.0,
                bit_rate_k=5000
            )
        )

        with patch("ray_curator.stages.video.io.video_reader.logger.info") as mock_log:
            stage = VideoReaderStage()
            stage._log_video_info(video)

            # Should log video information
            mock_log.assert_called()
            call_args = mock_log.call_args[0][0]
            assert "Downloaded" in call_args
            assert "video.mp4" in call_args

    def test_process_success(self) -> None:
        """Test process method with successful execution."""
        file_path = "/test/video.mp4"
        file_group_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[file_path]
        )

        with (
            patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
            patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True)
        ):
            stage = VideoReaderStage()
            result = stage.process(file_group_task)

            assert isinstance(result, VideoTask)
            assert result.task_id == f"{file_path}_processed"
            assert result.dataset_name == "test_dataset"
            assert isinstance(result.data, Video)
            assert result.data.input_video == pathlib.Path(file_path)

    def test_process_multiple_files_error(self) -> None:
        """Test process method raises error with multiple files."""
        file_group_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=["/test/video1.mp4", "/test/video2.mp4"]
        )

        stage = VideoReaderStage()

        with pytest.raises(ValueError, match="Expected 1 file, got 2"):
            stage.process(file_group_task)

    def test_process_download_failure(self) -> None:
        """Test process method when download fails."""
        file_path = "/test/video.mp4"
        file_group_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[file_path]
        )

        with patch.object(VideoReaderStage, "_download_video_bytes", return_value=False):
            stage = VideoReaderStage()
            result = stage.process(file_group_task)

            assert isinstance(result, VideoTask)
            assert result.task_id == f"{file_path}_processed"

    def test_process_metadata_failure(self) -> None:
        """Test process method when metadata extraction fails."""
        file_path = "/test/video.mp4"
        file_group_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[file_path]
        )

        with (
            patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
            patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=False)
        ):
            stage = VideoReaderStage()
            result = stage.process(file_group_task)

            assert isinstance(result, VideoTask)
            assert result.task_id == f"{file_path}_processed"

    def test_process_preserves_metadata(self) -> None:
        """Test process method preserves task metadata and stage performance."""
        file_path = "/test/video.mp4"
        original_metadata = {"source": "test", "batch": 1}
        original_stage_perf = [{"stage": "prev_stage", "time": 1.0}]

        file_group_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[file_path],
            _metadata=original_metadata,
            _stage_perf=original_stage_perf
        )

        with (
            patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
            patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True)
        ):
            stage = VideoReaderStage()
            result = stage.process(file_group_task)

            assert result._metadata == original_metadata
            assert result._stage_perf == original_stage_perf

    def test_process_with_verbose_logging(self) -> None:
        """Test process method enables verbose logging when configured."""
        file_path = "/test/video.mp4"
        file_group_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[file_path]
        )

        with (
            patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
            patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True),
            patch.object(VideoReaderStage, "_log_video_info") as mock_log
        ):
            stage = VideoReaderStage(verbose=True)
            stage.process(file_group_task)

            mock_log.assert_called_once()
            # Check that the video passed to log method has the correct input_video
            logged_video = mock_log.call_args[0][0]
            assert logged_video.input_video == pathlib.Path(file_path)

    def test_download_video_bytes_error_handling(self) -> None:
        """Test _download_video_bytes error handling and logging."""
        video = Video(input_video=pathlib.Path("/test/nonexistent.mp4"))

        with (
            patch("pathlib.Path.open", side_effect=FileNotFoundError("Test error")),
            patch("ray_curator.stages.video.io.video_reader.logger.error") as mock_log
        ):
            stage = VideoReaderStage()
            result = stage._download_video_bytes(video)

            assert result is False
            assert "download" in video.errors
            assert "Test error" in video.errors["download"]
            mock_log.assert_called_once()

    # Additional comprehensive tests below:

    def test_download_video_bytes_permission_error(self) -> None:
        """Test _download_video_bytes handles permission errors."""
        video = Video(input_video=pathlib.Path("/test/restricted.mp4"))

        with patch("pathlib.Path.open", side_effect=PermissionError("Permission denied")):
            stage = VideoReaderStage()
            result = stage._download_video_bytes(video)

            assert result is False
            assert "download" in video.errors
            assert "Permission denied" in video.errors["download"]

    def test_download_video_bytes_io_error(self) -> None:
        """Test _download_video_bytes handles general IO errors."""
        video = Video(input_video=pathlib.Path("/test/corrupted.mp4"))

        with patch("pathlib.Path.open", side_effect=OSError("IO error occurred")):
            stage = VideoReaderStage()
            result = stage._download_video_bytes(video)

            assert result is False
            assert "download" in video.errors
            assert "IO error occurred" in video.errors["download"]

    def test_download_video_bytes_empty_file(self) -> None:
        """Test _download_video_bytes with empty file."""
        with patch("pathlib.Path.open", mock.mock_open(read_data=b"")):
            video = Video(input_video=pathlib.Path("/test/empty.mp4"))
            stage = VideoReaderStage()

            result = stage._download_video_bytes(video)

            assert result is True
            assert video.source_bytes == b""
            assert "download" not in video.errors

    def test_download_video_bytes_large_file(self) -> None:
        """Test _download_video_bytes with large file data."""
        large_data = b"x" * (10 * 1024 * 1024)  # 10MB of test data

        with patch("pathlib.Path.open", mock.mock_open(read_data=large_data)):
            video = Video(input_video=pathlib.Path("/test/large.mp4"))
            stage = VideoReaderStage()

            result = stage._download_video_bytes(video)

            assert result is True
            assert video.source_bytes == large_data
            assert len(video.source_bytes) == 10 * 1024 * 1024

    def test_extract_and_validate_metadata_missing_codec_warning(self) -> None:
        """Test metadata validation warns about missing video codec."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        video.metadata = VideoMetadata(video_codec=None, pixel_format="yuv420p")

        with (
            patch.object(video, "populate_metadata", return_value=None),
            patch("ray_curator.stages.video.io.video_reader.logger.warning") as mock_warn
        ):
            stage = VideoReaderStage()
            result = stage._extract_and_validate_metadata(video)

            assert result is True
            mock_warn.assert_any_call("Codec could not be extracted for /test/video.mp4!")

    def test_extract_and_validate_metadata_missing_pixel_format_warning(self) -> None:
        """Test metadata validation warns about missing pixel format."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        video.metadata = VideoMetadata(video_codec="h264", pixel_format=None)

        with (
            patch.object(video, "populate_metadata", return_value=None),
            patch("ray_curator.stages.video.io.video_reader.logger.warning") as mock_warn
        ):
            stage = VideoReaderStage()
            result = stage._extract_and_validate_metadata(video)

            assert result is True
            mock_warn.assert_any_call("Pixel format could not be extracted for /test/video.mp4!")

    def test_extract_and_validate_metadata_multiple_warnings(self) -> None:
        """Test metadata validation warns about multiple missing fields."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        video.metadata = VideoMetadata(video_codec=None, pixel_format=None)

        with (
            patch.object(video, "populate_metadata", return_value=None),
            patch("ray_curator.stages.video.io.video_reader.logger.warning") as mock_warn
        ):
            stage = VideoReaderStage()
            result = stage._extract_and_validate_metadata(video)

            assert result is True
            assert mock_warn.call_count == 2

    def test_format_metadata_for_logging_complete_metadata(self) -> None:
        """Test _format_metadata_for_logging with complete metadata."""
        video = Video(
            input_video=pathlib.Path("/test/video.mp4"),
            source_bytes=b"test data" * 1000,
            metadata=VideoMetadata(
                size=9000,  # Set size so weight calculation works
                width=1920,
                height=1080,
                framerate=30.0,
                duration=120.0,
                bit_rate_k=5000
            )
        )

        stage = VideoReaderStage()
        formatted = stage._format_metadata_for_logging(video)

        assert formatted["size"] == "9,000B"
        assert formatted["res"] == "1920x1080"
        assert formatted["fps"] == "30.0"
        assert formatted["duration"] == "2m"
        assert formatted["weight"] == "0.40"  # 120/300 = 0.4 (assuming fraction = 1.0)
        assert formatted["bit_rate"] == "5000K"

    def test_format_metadata_for_logging_missing_metadata(self) -> None:
        """Test _format_metadata_for_logging with missing metadata fields."""
        video = Video(
            input_video=pathlib.Path("/test/video.mp4"),
            source_bytes=None,
            metadata=VideoMetadata(
                width=None,
                height=None,
                framerate=None,
                duration=None,
                bit_rate_k=None
            )
        )

        stage = VideoReaderStage()
        formatted = stage._format_metadata_for_logging(video)

        assert formatted["size"] == "0B"
        assert formatted["res"] == "unknownxunknown"
        assert formatted["fps"] == "unknown"
        assert formatted["duration"] == "unknown"
        assert formatted["weight"] == "unknown"
        assert formatted["bit_rate"] == "unknown"

    def test_format_metadata_for_logging_partial_metadata(self) -> None:
        """Test _format_metadata_for_logging with partial metadata."""
        video = Video(
            input_video=pathlib.Path("/test/video.mp4"),
            source_bytes=b"data",
            metadata=VideoMetadata(
                width=1280,
                height=None,
                framerate=25.5,
                duration=None,
                bit_rate_k=3000
            )
        )

        stage = VideoReaderStage()
        formatted = stage._format_metadata_for_logging(video)

        assert formatted["size"] == "4B"
        assert formatted["res"] == "1280xunknown"
        assert formatted["fps"] == "25.5"
        assert formatted["duration"] == "unknown"
        assert formatted["weight"] == "unknown"
        assert formatted["bit_rate"] == "3000K"

    def test_process_no_files_error(self) -> None:
        """Test process method raises error with no files."""
        file_group_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[]
        )

        stage = VideoReaderStage()

        with pytest.raises(ValueError, match="Expected 1 file, got 0"):
            stage.process(file_group_task)

    def test_process_creates_correct_task_id(self) -> None:
        """Test process method creates correct task ID from file path."""
        test_cases = [
            "/simple/path/video.mp4",
            "/complex/path with spaces/my_video.avi",
            "relative/path/test.mkv",
            "single_file.webm"
        ]

        for file_path in test_cases:
            file_group_task = FileGroupTask(
                task_id="original_task",
                dataset_name="test_dataset",
                data=[file_path]
            )

            with (
                patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
                patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True)
            ):
                stage = VideoReaderStage()
                result = stage.process(file_group_task)

                assert result.task_id == f"{file_path}_processed"

    def test_process_without_verbose_no_logging(self) -> None:
        """Test process method doesn't call _log_video_info when verbose is False."""
        file_path = "/test/video.mp4"
        file_group_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[file_path]
        )

        with (
            patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
            patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True),
            patch.object(VideoReaderStage, "_log_video_info") as mock_log
        ):
            stage = VideoReaderStage(verbose=False)
            stage.process(file_group_task)

            mock_log.assert_not_called()

    def test_stage_name_property(self) -> None:
        """Test that stage name property is correctly set."""
        stage = VideoReaderStage()
        assert stage.name == "video_reader"
        assert stage._name == "video_reader"

    def test_stage_default_verbose_setting(self) -> None:
        """Test default verbose setting is False."""
        stage = VideoReaderStage()
        assert stage.verbose is False

    def test_video_task_data_structure(self) -> None:
        """Test that created VideoTask has correct data structure."""
        file_path = "/test/video.mp4"
        file_group_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[file_path]
        )

        with (
            patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
            patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True)
        ):
            stage = VideoReaderStage()
            result = stage.process(file_group_task)

            # Verify VideoTask structure
            assert hasattr(result, "data")
            assert hasattr(result, "task_id")
            assert hasattr(result, "dataset_name")
            assert hasattr(result, "_metadata")
            assert hasattr(result, "_stage_perf")

            # Verify Video data structure
            video_data = result.data
            assert hasattr(video_data, "input_video")
            assert hasattr(video_data, "source_bytes")
            assert hasattr(video_data, "metadata")
            assert hasattr(video_data, "errors")

    def test_metadata_extraction_failure_logging(self) -> None:
        """Test that metadata extraction failure is properly logged."""
        video = Video(input_video=pathlib.Path("/test/corrupted.mp4"))

        with (
            patch.object(video, "populate_metadata", side_effect=RuntimeError("Corrupted file")),
            patch("ray_curator.stages.video.io.video_reader.logger.warning") as mock_warn
        ):
            stage = VideoReaderStage()
            result = stage._extract_and_validate_metadata(video)

            assert result is False
            mock_warn.assert_called_with("Failed to extract metadata for /test/corrupted.mp4: Corrupted file")

    def test_s3_path_error_message(self) -> None:
        """Test that S3 path error contains proper message."""
        video = Video(input_video="s3://my-bucket/videos/test.mp4")
        stage = VideoReaderStage()

        result = stage._download_video_bytes(video)

        assert result is False
        assert "download" in video.errors
        assert "S3 client is required for S3 destination" in video.errors["download"]

    @pytest.mark.parametrize("file_extension", [".mp4", ".avi", ".mov", ".mkv", ".webm"])
    def test_process_with_various_file_extensions(self, file_extension: str) -> None:
        """Test process method works with various video file extensions."""
        file_path = f"/test/video{file_extension}"
        file_group_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[file_path]
        )

        with (
            patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
            patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True)
        ):
            stage = VideoReaderStage()
            result = stage.process(file_group_task)

            assert isinstance(result, VideoTask)
            assert result.data.input_video == pathlib.Path(file_path)

    def test_deepcopy_preservation(self) -> None:
        """Test that deepcopy correctly preserves metadata and stage performance."""
        file_path = "/test/video.mp4"
        nested_metadata = {"config": {"param": "value"}, "nested_list": [1, 2, 3]}
        nested_stage_perf = [{"stage": "prev", "nested": {"time": 1.0}}]

        file_group_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[file_path],
            _metadata=nested_metadata,
            _stage_perf=nested_stage_perf
        )

        with (
            patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
            patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True)
        ):
            stage = VideoReaderStage()
            result = stage.process(file_group_task)

            # Verify deep copy worked (same content, different objects)
            assert result._metadata == nested_metadata
            assert result._metadata is not nested_metadata
            assert result._stage_perf == nested_stage_perf
            assert result._stage_perf is not nested_stage_perf
