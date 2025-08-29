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

import pathlib
from unittest import mock
from unittest.mock import patch

import pytest

from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.video.io.video_reader import VideoReader, VideoReaderStage
from nemo_curator.tasks.file_group import FileGroupTask
from nemo_curator.tasks.video import Video, VideoMetadata, VideoTask


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
            bit_rate_k=5000,
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
            patch("nemo_curator.stages.video.io.video_reader.logger.warning") as mock_warn,
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
                bit_rate_k=5000,
            ),
        )

        with patch("nemo_curator.stages.video.io.video_reader.logger.info") as mock_log:
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
        file_group_task = FileGroupTask(task_id="test_task", dataset_name="test_dataset", data=[file_path])

        with (
            patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
            patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True),
        ):
            stage = VideoReaderStage()
            result = stage.process(file_group_task)

            assert isinstance(result, VideoTask)
            assert result.task_id == f"{file_path}_processed"
            assert result.dataset_name == "test_dataset"
            assert isinstance(result.data, Video)
            assert result.data.input_video == file_path

    def test_process_download_failure(self) -> None:
        """Test process method when download fails."""
        file_path = "/test/video.mp4"
        file_group_task = FileGroupTask(task_id="test_task", dataset_name="test_dataset", data=[file_path])

        with patch.object(VideoReaderStage, "_download_video_bytes", return_value=False):
            stage = VideoReaderStage()
            result = stage.process(file_group_task)

            assert isinstance(result, VideoTask)
            assert result.task_id == f"{file_path}_processed"

    def test_process_metadata_failure(self) -> None:
        """Test process method when metadata extraction fails."""
        file_path = "/test/video.mp4"
        file_group_task = FileGroupTask(task_id="test_task", dataset_name="test_dataset", data=[file_path])

        with (
            patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
            patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=False),
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
            _stage_perf=original_stage_perf,
        )

        with (
            patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
            patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True),
        ):
            stage = VideoReaderStage()
            result = stage.process(file_group_task)

            assert result._metadata == original_metadata
            assert result._stage_perf == original_stage_perf

    def test_process_with_verbose_logging(self) -> None:
        """Test process method enables verbose logging when configured."""
        file_path = "/test/video.mp4"
        file_group_task = FileGroupTask(task_id="test_task", dataset_name="test_dataset", data=[file_path])

        with (
            patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
            patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True),
            patch.object(VideoReaderStage, "_log_video_info") as mock_log,
        ):
            stage = VideoReaderStage(verbose=True)
            stage.process(file_group_task)

            mock_log.assert_called_once()
            # Check that the video passed to log method has the correct input_video
            logged_video = mock_log.call_args[0][0]
            assert logged_video.input_video == file_path

    def test_download_video_bytes_error_handling(self) -> None:
        """Test _download_video_bytes error handling and logging."""
        video = Video(input_video=pathlib.Path("/test/nonexistent.mp4"))

        with (
            patch("pathlib.Path.open", side_effect=FileNotFoundError("Test error")),
            patch("nemo_curator.stages.video.io.video_reader.logger.error") as mock_log,
        ):
            stage = VideoReaderStage()
            result = stage._download_video_bytes(video)

            assert result is False
            assert "download" in video.errors
            assert "Test error" in video.errors["download"]
            mock_log.assert_called_once()

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
            patch("nemo_curator.stages.video.io.video_reader.logger.warning") as mock_warn,
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
            patch("nemo_curator.stages.video.io.video_reader.logger.warning") as mock_warn,
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
            patch("nemo_curator.stages.video.io.video_reader.logger.warning") as mock_warn,
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
                bit_rate_k=5000,
            ),
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
            metadata=VideoMetadata(width=None, height=None, framerate=None, duration=None, bit_rate_k=None),
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
            metadata=VideoMetadata(width=1280, height=None, framerate=25.5, duration=None, bit_rate_k=3000),
        )

        stage = VideoReaderStage()
        formatted = stage._format_metadata_for_logging(video)

        assert formatted["size"] == "4B"
        assert formatted["res"] == "1280xunknown"
        assert formatted["fps"] == "25.5"
        assert formatted["duration"] == "unknown"
        assert formatted["weight"] == "unknown"
        assert formatted["bit_rate"] == "3000K"

    def test_process_creates_correct_task_id(self) -> None:
        """Test process method creates correct task ID from file path."""
        test_cases = [
            "/simple/path/video.mp4",
            "/complex/path with spaces/my_video.avi",
            "relative/path/test.mkv",
            "single_file.webm",
        ]

        for file_path in test_cases:
            file_group_task = FileGroupTask(task_id="original_task", dataset_name="test_dataset", data=[file_path])

            with (
                patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
                patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True),
            ):
                stage = VideoReaderStage()
                result = stage.process(file_group_task)

                assert result.task_id == f"{file_path}_processed"

    def test_process_without_verbose_no_logging(self) -> None:
        """Test process method doesn't call _log_video_info when verbose is False."""
        file_path = "/test/video.mp4"
        file_group_task = FileGroupTask(task_id="test_task", dataset_name="test_dataset", data=[file_path])

        with (
            patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
            patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True),
            patch.object(VideoReaderStage, "_log_video_info") as mock_log,
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
        file_group_task = FileGroupTask(task_id="test_task", dataset_name="test_dataset", data=[file_path])

        with (
            patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
            patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True),
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
            patch("nemo_curator.stages.video.io.video_reader.logger.warning") as mock_warn,
        ):
            stage = VideoReaderStage()
            result = stage._extract_and_validate_metadata(video)

            assert result is False
            mock_warn.assert_called_with("Failed to extract metadata for /test/corrupted.mp4: Corrupted file")

    @pytest.mark.parametrize("file_extension", [".mp4", ".avi", ".mov", ".mkv", ".webm"])
    def test_process_with_various_file_extensions(self, file_extension: str) -> None:
        """Test process method works with various video file extensions."""
        file_path = f"/test/video{file_extension}"
        file_group_task = FileGroupTask(task_id="test_task", dataset_name="test_dataset", data=[file_path])

        with (
            patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
            patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True),
        ):
            stage = VideoReaderStage()
            result = stage.process(file_group_task)

            assert isinstance(result, VideoTask)
            assert result.data.input_video == file_path

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
            _stage_perf=nested_stage_perf,
        )

        with (
            patch.object(VideoReaderStage, "_download_video_bytes", return_value=True),
            patch.object(VideoReaderStage, "_extract_and_validate_metadata", return_value=True),
        ):
            stage = VideoReaderStage()
            result = stage.process(file_group_task)

            # Verify deep copy worked (same content, different objects)
            assert result._metadata == nested_metadata
            assert result._metadata is not nested_metadata
            assert result._stage_perf == nested_stage_perf
            assert result._stage_perf is not nested_stage_perf


class TestVideoReader:
    """Test suite for VideoReader composite functionality."""

    def test_stage_initialization_default_values(self) -> None:
        """Test VideoReader initialization with default values."""
        stage = VideoReader(input_video_path="/test/videos")

        assert stage.input_video_path == "/test/videos"
        assert stage.video_limit is None
        assert stage.verbose is False

    def test_stage_initialization_custom_values(self) -> None:
        """Test VideoReader initialization with custom values."""
        stage = VideoReader(input_video_path="/custom/path", video_limit=100, verbose=True)

        assert stage.input_video_path == "/custom/path"
        assert stage.video_limit == 100
        assert stage.verbose is True

    def test_stage_properties(self) -> None:
        """Test stage properties are correctly defined."""
        stage = VideoReader(input_video_path="/test/videos")

        # Test stage name
        assert stage.name == "video_reader"

        # Test that it's a composite stage (should raise error when trying to process)
        from nemo_curator.tasks import _EmptyTask

        empty_task = _EmptyTask(task_id="test", dataset_name="test", data=None)
        with pytest.raises(RuntimeError, match="Composite stage 'video_reader' should not be executed directly"):
            stage.process(empty_task)

    def test_decompose_basic(self) -> None:
        """Test decomposition into constituent stages with basic parameters."""
        stage = VideoReader(input_video_path="/test/videos", video_limit=50, verbose=True)

        stages = stage.decompose()

        # Should return exactly 2 stages
        assert len(stages) == 2

        # Check stage types and order
        assert isinstance(stages[0], FilePartitioningStage)
        assert isinstance(stages[1], VideoReaderStage)

        # Check FilePartitioningStage configuration
        file_stage = stages[0]
        assert file_stage.file_paths == "/test/videos"
        assert file_stage.files_per_partition == 1
        assert file_stage.file_extensions == [".mp4", ".mov", ".avi", ".mkv", ".webm"]
        assert file_stage.limit == 50

        # Check VideoReaderStage configuration
        reader_stage = stages[1]
        assert reader_stage.verbose is True

    def test_decompose_unlimited_videos(self) -> None:
        """Test decomposition with unlimited video processing."""
        stage = VideoReader(input_video_path="/unlimited/videos", video_limit=None, verbose=False)

        stages = stage.decompose()
        file_stage = stages[0]
        reader_stage = stages[1]

        # With None limit, should pass None to file partitioning stage
        assert file_stage.limit is None
        assert reader_stage.verbose is False

    def test_decompose_different_paths(self) -> None:
        """Test decomposition with different input paths."""
        test_paths = ["/home/user/videos", "/mnt/storage/media", "relative/path/videos", "single_video.mp4"]

        for path in test_paths:
            stage = VideoReader(input_video_path=path)
            stages = stage.decompose()

            file_stage = stages[0]
            assert file_stage.file_paths == path

    def test_get_description_unlimited(self) -> None:
        """Test get_description method with unlimited videos."""
        stage = VideoReader(input_video_path="/test/videos", video_limit=None)

        description = stage.get_description()
        expected = (
            "Reads video files from '/test/videos' "
            "(limit: unlimited) "
            "and downloads/processes them with metadata extraction"
        )
        assert description == expected

    def test_get_description_limited(self) -> None:
        """Test get_description method with limited videos."""
        stage = VideoReader(input_video_path="/test/videos", video_limit=25)

        description = stage.get_description()
        expected = (
            "Reads video files from '/test/videos' (limit: 25) and downloads/processes them with metadata extraction"
        )
        assert description == expected

    def test_inputs_outputs_delegation(self) -> None:
        """Test that inputs/outputs are properly delegated to constituent stages."""
        stage = VideoReader(input_video_path="/test/videos")

        # Should delegate to first stage for inputs
        inputs = stage.inputs()
        # FilePartitioningStage inputs should be empty
        assert inputs == ([], [])

        # Should delegate to last stage for outputs
        outputs = stage.outputs()
        # VideoReaderStage outputs
        assert outputs == (["data"], ["source_bytes", "metadata"])

    def test_post_init_calls_super(self) -> None:
        """Test that __post_init__ properly calls parent initialization."""
        with patch("nemo_curator.stages.base.CompositeStage.__init__") as mock_super_init:
            VideoReader(input_video_path="/test/videos")

            # Should have called parent __init__
            mock_super_init.assert_called_once()

    def test_decompose_stage_independence(self) -> None:
        """Test that each call to decompose returns independent stage instances."""
        stage = VideoReader(input_video_path="/test/videos", video_limit=10, verbose=True)

        # Get two decompositions
        stages1 = stage.decompose()
        stages2 = stage.decompose()

        # Should be different instances
        assert stages1[0] is not stages2[0]
        assert stages1[1] is not stages2[1]

        # But should have same configuration
        assert stages1[0].file_paths == stages2[0].file_paths
        assert stages1[0].limit == stages2[0].limit
        assert stages1[1].verbose == stages2[1].verbose

    def test_decompose_preserves_parameters(self) -> None:
        """Test that decompose preserves all input parameters correctly."""
        stage = VideoReader(input_video_path="/complex/path/with spaces", video_limit=999, verbose=True)

        stages = stage.decompose()
        file_stage, reader_stage = stages

        # Ensure all parameters are correctly passed through
        assert file_stage.file_paths == "/complex/path/with spaces"
        assert file_stage.files_per_partition == 1
        assert file_stage.limit == 999
        assert set(file_stage.file_extensions) == {".mp4", ".mov", ".avi", ".mkv", ".webm"}

        assert reader_stage.verbose is True

    def test_composite_stage_behavior(self) -> None:
        """Test that VideoReader behaves correctly as a CompositeStage."""
        stage = VideoReader(input_video_path="/test/videos")

        # Should be a CompositeStage
        from nemo_curator.stages.base import CompositeStage

        assert isinstance(stage, CompositeStage)

        # Should have the correct generic type annotations
        # (This is more of a static analysis check, but we can verify the structure)
        stages = stage.decompose()
        assert len(stages) > 0
        assert all(hasattr(s, "process") for s in stages)

    def test_description_path_handling(self) -> None:
        """Test description method handles various path formats correctly."""
        test_cases = [
            ("/simple/path", "'/simple/path'"),
            ("/path with spaces/videos", "'/path with spaces/videos'"),
            ("relative/path", "'relative/path'"),
            ("file.mp4", "'file.mp4'"),
        ]

        for input_path, expected_path_in_desc in test_cases:
            stage = VideoReader(input_video_path=input_path)
            description = stage.get_description()
            assert expected_path_in_desc in description
