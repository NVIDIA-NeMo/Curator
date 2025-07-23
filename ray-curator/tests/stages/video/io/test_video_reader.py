"""Test suite for VideoReaderStage."""

import pathlib
from unittest import mock
from unittest.mock import patch

from ray_curator.stages.video.io.video_reader import VideoReaderStage
from ray_curator.tasks import Video, VideoMetadata, VideoTask


class TestVideoReaderStage:
    """Test suite for VideoReaderStage."""

    def test_stage_properties(self) -> None:
        """Test stage properties."""
        stage = VideoReaderStage()
        assert stage.name == "video_reader"
        assert stage.inputs() == (["data"], ["input_video"])
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
        with patch.object(video, "populate_metadata", side_effect=Exception("Metadata error")):
            stage = VideoReaderStage()
            result = stage._extract_and_validate_metadata(video)

            assert result is False

    def test_extract_and_validate_metadata_warns_missing_codec(self) -> None:
        """Test _extract_and_validate_metadata logs warning for missing codec."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        video.metadata = VideoMetadata(
            video_codec=None,  # Missing codec
            pixel_format="yuv420p",
            width=1920,
            height=1080,
            framerate=30.0,
            duration=60.0,
            bit_rate_k=5000
        )

        with patch.object(video, "populate_metadata", return_value=None):
            with patch("ray_curator.stages.video.io.video_reader.logger.warning") as mock_warning:
                stage = VideoReaderStage()
                result = stage._extract_and_validate_metadata(video)

                assert result is True
                mock_warning.assert_called_once()
                assert "Codec could not be extracted" in str(mock_warning.call_args)

    def test_extract_and_validate_metadata_warns_missing_pixel_format(self) -> None:
        """Test _extract_and_validate_metadata logs warning for missing pixel format."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        video.metadata = VideoMetadata(
            video_codec="h264",
            pixel_format=None,  # Missing pixel format
            width=1920,
            height=1080,
            framerate=30.0,
            duration=60.0,
            bit_rate_k=5000
        )

        with patch.object(video, "populate_metadata", return_value=None):
            with patch("ray_curator.stages.video.io.video_reader.logger.warning") as mock_warning:
                stage = VideoReaderStage()
                result = stage._extract_and_validate_metadata(video)

                assert result is True
                mock_warning.assert_called_once()
                assert "Pixel format could not be extracted" in str(mock_warning.call_args)

    def test_process_success(self) -> None:
        """Test process method with successful execution."""
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
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        with patch.object(VideoReaderStage, "_download_video_bytes", return_value=True):
            with patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True):
                stage = VideoReaderStage()
                result = stage.process(task)

                assert result == task

    def test_process_download_failure(self) -> None:
        """Test process method when download fails."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        with patch.object(VideoReaderStage, "_download_video_bytes", return_value=False):
            stage = VideoReaderStage()
            result = stage.process(task)

            assert result == task

    def test_process_metadata_failure(self) -> None:
        """Test process method when metadata extraction fails."""
        video = Video(input_video=pathlib.Path("/test/video.mp4"))
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        with patch.object(VideoReaderStage, "_download_video_bytes", return_value=True):
            with patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=False):
                stage = VideoReaderStage()
                result = stage.process(task)

                assert result == task

    def test_log_video_info_verbose_enabled(self) -> None:
        """Test _log_video_info when verbose logging is enabled."""
        video = Video(
            input_video=pathlib.Path("/test/video.mp4"),
            source_bytes=b"test data" * 1000,
            metadata=VideoMetadata(
                video_codec="h264",
                pixel_format="yuv420p",
                width=1920,
                height=1080,
                framerate=30.0,
                duration=120.0,
                bit_rate_k=5000
            )
        )

        with patch("ray_curator.stages.video.io.video_reader.logger.info") as mock_info:
            stage = VideoReaderStage(verbose=True)
            stage._log_video_info(video)

            mock_info.assert_called_once()
            log_message = str(mock_info.call_args[0][0])
            assert "/test/video.mp4" in log_message
            assert "size=" in log_message
            assert "res=1920x1080" in log_message
            assert "fps=30.0" in log_message
            assert "duration=2m" in log_message
            assert "bit_rate=5000K" in log_message

    def test_log_video_info_verbose_disabled(self) -> None:
        """Test _log_video_info when verbose logging is disabled."""
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
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        with patch.object(VideoReaderStage, "_download_video_bytes", return_value=True):
            with patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True):
                with patch("ray_curator.stages.video.io.video_reader.logger.info") as mock_info:
                    stage = VideoReaderStage(verbose=False)
                    stage.process(task)

                    # _log_video_info should not be called when verbose=False
                    mock_info.assert_not_called()

    def test_format_metadata_for_logging_all_fields(self) -> None:
        """Test _format_metadata_for_logging with all metadata fields present."""
        video = Video(
            input_video=pathlib.Path("/test/video.mp4"),
            source_bytes=b"test data" * 1000,
            metadata=VideoMetadata(
                video_codec="h264",
                pixel_format="yuv420p",
                width=1920,
                height=1080,
                framerate=29.97,
                duration=125.5,
                bit_rate_k=8500
            )
        )

        stage = VideoReaderStage()
        formatted = stage._format_metadata_for_logging(video)

        assert formatted["size"] == "9,000B"
        assert formatted["res"] == "1920x1080"
        assert formatted["fps"] == "30.0"
        assert formatted["duration"] == "2m"
        assert formatted["weight"] == f"{video.weight:.2f}"
        assert formatted["bit_rate"] == "8500K"

    def test_format_metadata_for_logging_missing_fields(self) -> None:
        """Test _format_metadata_for_logging with missing metadata fields."""
        video = Video(
            input_video=pathlib.Path("/test/video.mp4"),
            source_bytes=None,
            metadata=VideoMetadata(
                video_codec="h264",
                pixel_format="yuv420p",
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

    def test_process_with_verbose_logging(self) -> None:
        """Test process method enables verbose logging when configured."""
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
        task = VideoTask(task_id="test_task", dataset_name="test_dataset", data=video)

        with patch.object(VideoReaderStage, "_download_video_bytes", return_value=True):
            with patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True):
                with patch.object(VideoReaderStage, "_log_video_info") as mock_log:
                    stage = VideoReaderStage(verbose=True)
                    stage.process(task)

                    mock_log.assert_called_once_with(video)

    def test_download_video_bytes_error_handling(self) -> None:
        """Test _download_video_bytes error handling and logging."""
        video = Video(input_video=pathlib.Path("/test/nonexistent.mp4"))

        with patch("pathlib.Path.open", side_effect=FileNotFoundError("Test error")):
            with patch("ray_curator.stages.video.io.video_reader.logger.warning") as mock_warning:
                stage = VideoReaderStage()
                result = stage._download_video_bytes(video)

                assert result is False
                assert "download" in video.errors
                assert "Test error" in video.errors["download"]

    def test_s3_error_path(self) -> None:
        """Test that S3 paths raise appropriate error."""
        with patch("ray_curator.stages.video.io.video_reader.logger.error") as mock_error:
            stage = VideoReaderStage()
            video = Video(input_video="s3://bucket/key")
            result = stage._download_video_bytes(video)

            assert result is False
            mock_error.assert_called_once() 