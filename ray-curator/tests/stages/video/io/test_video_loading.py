"""Test suite for VideoLoadingStage."""

from unittest.mock import patch

import pytest

from ray_curator.stages.io.reader.file_partitioning import FilePartitioningStage
from ray_curator.stages.video.io.video_loading import VideoLoadingStage
from ray_curator.stages.video.io.video_reader import VideoReaderStage


class TestVideoLoadingStage:
    """Test suite for VideoLoadingStage composite functionality."""

    def test_stage_initialization_default_values(self) -> None:
        """Test VideoLoadingStage initialization with default values."""
        stage = VideoLoadingStage(input_video_path="/test/videos")

        assert stage.input_video_path == "/test/videos"
        assert stage.video_limit == -1
        assert stage.verbose is False

    def test_stage_initialization_custom_values(self) -> None:
        """Test VideoLoadingStage initialization with custom values."""
        stage = VideoLoadingStage(
            input_video_path="/custom/path",
            video_limit=100,
            verbose=True
        )

        assert stage.input_video_path == "/custom/path"
        assert stage.video_limit == 100
        assert stage.verbose is True

    def test_stage_properties(self) -> None:
        """Test stage properties are correctly defined."""
        stage = VideoLoadingStage(input_video_path="/test/videos")

        # Test stage name
        assert stage.name == "video_loading"

        # Test that it's a composite stage (should raise error when trying to process)
        from ray_curator.tasks import _EmptyTask
        empty_task = _EmptyTask(task_id="test", dataset_name="test", data=None)
        with pytest.raises(RuntimeError, match="Composite stage 'video_loading' should not be executed directly"):
            stage.process(empty_task)

    def test_decompose_basic(self) -> None:
        """Test decomposition into constituent stages with basic parameters."""
        stage = VideoLoadingStage(
            input_video_path="/test/videos",
            video_limit=50,
            verbose=True
        )

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
        stage = VideoLoadingStage(
            input_video_path="/unlimited/videos",
            video_limit=-1,
            verbose=False
        )

        stages = stage.decompose()
        file_stage = stages[0]
        reader_stage = stages[1]

        # With -1 limit, should pass -1 to file partitioning stage
        assert file_stage.limit == -1
        assert reader_stage.verbose is False

    def test_decompose_different_paths(self) -> None:
        """Test decomposition with different input paths."""
        test_paths = [
            "/home/user/videos",
            "/mnt/storage/media",
            "relative/path/videos",
            "single_video.mp4"
        ]

        for path in test_paths:
            stage = VideoLoadingStage(input_video_path=path)
            stages = stage.decompose()

            file_stage = stages[0]
            assert file_stage.file_paths == path

    def test_get_description_unlimited(self) -> None:
        """Test get_description method with unlimited videos."""
        stage = VideoLoadingStage(
            input_video_path="/test/videos",
            video_limit=-1
        )

        description = stage.get_description()
        expected = (
            "Reads video files from '/test/videos' "
            "(limit: unlimited) "
            "and downloads/processes them with metadata extraction"
        )
        assert description == expected

    def test_get_description_limited(self) -> None:
        """Test get_description method with limited videos."""
        stage = VideoLoadingStage(
            input_video_path="/test/videos",
            video_limit=25
        )

        description = stage.get_description()
        expected = (
            "Reads video files from '/test/videos' "
            "(limit: 25) "
            "and downloads/processes them with metadata extraction"
        )
        assert description == expected

    def test_get_description_zero_limit(self) -> None:
        """Test get_description method with zero limit."""
        stage = VideoLoadingStage(
            input_video_path="/test/videos",
            video_limit=0
        )

        description = stage.get_description()
        expected = (
            "Reads video files from '/test/videos' "
            "(limit: unlimited) "
            "and downloads/processes them with metadata extraction"
        )
        assert description == expected

    def test_inputs_outputs_delegation(self) -> None:
        """Test that inputs/outputs are properly delegated to constituent stages."""
        stage = VideoLoadingStage(input_video_path="/test/videos")

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
        with patch("ray_curator.stages.base.CompositeStage.__init__") as mock_super_init:
            VideoLoadingStage(input_video_path="/test/videos")

            # Should have called parent __init__
            mock_super_init.assert_called_once()

    def test_decompose_stage_independence(self) -> None:
        """Test that each call to decompose returns independent stage instances."""
        stage = VideoLoadingStage(
            input_video_path="/test/videos",
            video_limit=10,
            verbose=True
        )

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
        stage = VideoLoadingStage(
            input_video_path="/complex/path/with spaces",
            video_limit=999,
            verbose=True
        )

        stages = stage.decompose()
        file_stage, reader_stage = stages

        # Ensure all parameters are correctly passed through
        assert file_stage.file_paths == "/complex/path/with spaces"
        assert file_stage.files_per_partition == 1
        assert file_stage.limit == 999
        assert set(file_stage.file_extensions) == {".mp4", ".mov", ".avi", ".mkv", ".webm"}

        assert reader_stage.verbose is True

    @pytest.mark.parametrize(("video_limit", "expected_limit"), [
        (-1, -1),
        (0, 0),
        (1, 1),
        (100, 100),
        (999999, 999999)
    ])
    def test_decompose_video_limit_values(self, video_limit: int, expected_limit: int) -> None:
        """Test decompose with various video limit values."""
        stage = VideoLoadingStage(
            input_video_path="/test/videos",
            video_limit=video_limit
        )

        stages = stage.decompose()
        file_stage = stages[0]

        assert file_stage.limit == expected_limit

    @pytest.mark.parametrize("verbose", [True, False])
    def test_decompose_verbose_flag(self, verbose: bool) -> None:
        """Test decompose with different verbose flag values."""
        stage = VideoLoadingStage(
            input_video_path="/test/videos",
            verbose=verbose
        )

        stages = stage.decompose()
        reader_stage = stages[1]

        assert reader_stage.verbose is verbose

    def test_composite_stage_behavior(self) -> None:
        """Test that VideoLoadingStage behaves correctly as a CompositeStage."""
        stage = VideoLoadingStage(input_video_path="/test/videos")

        # Should be a CompositeStage
        from ray_curator.stages.base import CompositeStage
        assert isinstance(stage, CompositeStage)

        # Should have the correct generic type annotations
        # (This is more of a static analysis check, but we can verify the structure)
        stages = stage.decompose()
        assert len(stages) > 0
        assert all(hasattr(s, "process") for s in stages)

    def test_file_extensions_configuration(self) -> None:
        """Test that the correct video file extensions are configured."""
        stage = VideoLoadingStage(input_video_path="/test/videos")
        stages = stage.decompose()
        file_stage = stages[0]

        expected_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
        assert file_stage.file_extensions == expected_extensions

    def test_files_per_partition_configuration(self) -> None:
        """Test that files_per_partition is correctly set to 1."""
        stage = VideoLoadingStage(input_video_path="/test/videos")
        stages = stage.decompose()
        file_stage = stages[0]

        # Should process one video file per partition for video processing
        assert file_stage.files_per_partition == 1

    def test_description_path_handling(self) -> None:
        """Test description method handles various path formats correctly."""
        test_cases = [
            ("/simple/path", "'/simple/path'"),
            ("/path with spaces/videos", "'/path with spaces/videos'"),
            ("relative/path", "'relative/path'"),
            ("file.mp4", "'file.mp4'"),
        ]

        for input_path, expected_path_in_desc in test_cases:
            stage = VideoLoadingStage(input_video_path=input_path)
            description = stage.get_description()
            assert expected_path_in_desc in description
