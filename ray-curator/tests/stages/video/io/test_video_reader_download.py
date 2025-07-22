import pytest

from ray_curator.stages.video.io.video_download import VideoDownloadStage
from ray_curator.stages.video.io.video_reader import VideoReaderStage
from ray_curator.stages.video.io.video_reader_download import VideoReaderDownloadStage
from ray_curator.tasks import _EmptyTask


class TestVideoReaderDownloadStage:
    """Test suite for VideoReaderDownloadStage composite stage."""

    def test_initialization_default_values(self):
        """Test initialization with default parameter values."""
        stage = VideoReaderDownloadStage(input_video_path="/test/path")

        assert stage.input_video_path == "/test/path"
        assert stage.video_limit == -1
        assert stage.verbose is False
        assert hasattr(stage, "_with_operations")  # Should be initialized by __post_init__
        assert stage._with_operations == []

    def test_initialization_custom_values(self):
        """Test initialization with custom parameter values."""
        stage = VideoReaderDownloadStage(
            input_video_path="/custom/path",
            video_limit=100,
            verbose=True
        )

        assert stage.input_video_path == "/custom/path"
        assert stage.video_limit == 100
        assert stage.verbose is True
        assert hasattr(stage, "_with_operations")
        assert stage._with_operations == []

    def test_name_property(self):
        """Test that the name property returns the correct value."""
        stage = VideoReaderDownloadStage(input_video_path="/test/path")
        assert stage.name == "video_reader_download"

    def test_decompose(self):
        """Test that decompose returns the correct constituent stages."""
        stage = VideoReaderDownloadStage(
            input_video_path="/test/videos",
            video_limit=50,
            verbose=True
        )

        stages = stage.decompose()

        # Should return exactly 2 stages
        assert len(stages) == 2

        # First stage should be VideoReaderStage
        reader_stage = stages[0]
        assert isinstance(reader_stage, VideoReaderStage)
        assert reader_stage.input_video_path == "/test/videos"
        assert reader_stage.video_limit == 50

        # Second stage should be VideoDownloadStage
        download_stage = stages[1]
        assert isinstance(download_stage, VideoDownloadStage)
        assert download_stage.verbose is True

    def test_decompose_default_values(self):
        """Test decompose with default parameter values."""
        stage = VideoReaderDownloadStage(input_video_path="/default/path")

        stages = stage.decompose()

        reader_stage = stages[0]
        assert reader_stage.input_video_path == "/default/path"
        assert reader_stage.video_limit == -1

        download_stage = stages[1]
        assert download_stage.verbose is False

    def test_inputs(self):
        """Test that inputs() method delegates to the first constituent stage."""
        stage = VideoReaderDownloadStage(input_video_path="/test/path")

        # Should delegate to VideoReaderStage.inputs()
        top_level_attrs, data_attrs = stage.inputs()

        # VideoReaderStage returns ([], [])
        assert top_level_attrs == []
        assert data_attrs == []

    def test_outputs(self):
        """Test that outputs() method delegates to the last constituent stage."""
        stage = VideoReaderDownloadStage(input_video_path="/test/path")

        # Should delegate to VideoDownloadStage.outputs()
        top_level_attrs, data_attrs = stage.outputs()

        # VideoDownloadStage returns (["data"], ["source_bytes", "metadata"])
        assert top_level_attrs == ["data"]
        assert data_attrs == ["source_bytes", "metadata"]

    def test_get_description_unlimited(self):
        """Test get_description with unlimited video limit."""
        stage = VideoReaderDownloadStage(
            input_video_path="/my/videos",
            video_limit=-1
        )

        description = stage.get_description()
        expected = (
            "Reads video files from '/my/videos' "
            "(limit: unlimited) "
            "and downloads/processes them with metadata extraction"
        )
        assert description == expected

    def test_get_description_limited(self):
        """Test get_description with specific video limit."""
        stage = VideoReaderDownloadStage(
            input_video_path="/my/videos",
            video_limit=25
        )

        description = stage.get_description()
        expected = (
            "Reads video files from '/my/videos' "
            "(limit: 25) "
            "and downloads/processes them with metadata extraction"
        )
        assert description == expected

    def test_with_method_single_stage_config(self):
        """Test with_() method for configuring a single constituent stage."""
        stage = VideoReaderDownloadStage(input_video_path="/test/path")

        configured_stage = stage.with_({
            "video_reader": {
                "name": "custom_reader",
                "batch_size": 4
            }
        })

        # Should return the same instance
        assert configured_stage is stage

        # Should store the configuration
        assert len(stage._with_operations) == 1
        assert stage._with_operations[0] == {
            "video_reader": {
                "name": "custom_reader",
                "batch_size": 4
            }
        }

    def test_with_method_multiple_stage_config(self):
        """Test with_() method for configuring multiple constituent stages."""
        stage = VideoReaderDownloadStage(input_video_path="/test/path")

        configured_stage = stage.with_({
            "video_reader": {
                "name": "custom_reader",
                "batch_size": 4
            },
            "video_download": {
                "name": "custom_download",
                "batch_size": 2
            }
        })

        assert configured_stage is stage
        assert len(stage._with_operations) == 1
        assert stage._with_operations[0]["video_reader"]["name"] == "custom_reader"
        assert stage._with_operations[0]["video_download"]["name"] == "custom_download"

    def test_with_method_chaining(self):
        """Test chaining multiple with_() calls."""
        stage = VideoReaderDownloadStage(input_video_path="/test/path")

        stage.with_({
            "video_reader": {"batch_size": 4}
        }).with_({
            "video_download": {"batch_size": 2}
        })

        # Should have two with operations
        assert len(stage._with_operations) == 2
        assert stage._with_operations[0] == {"video_reader": {"batch_size": 4}}
        assert stage._with_operations[1] == {"video_download": {"batch_size": 2}}

    def test_decompose_and_apply_with(self):
        """Test decompose_and_apply_with method applies configurations correctly."""
        stage = VideoReaderDownloadStage(input_video_path="/test/path")

        # Configure both stages
        stage.with_({
            "video_reader": {
                "name": "configured_reader",
                "batch_size": 8
            },
            "video_download": {
                "name": "configured_download",
                "batch_size": 4
            }
        })

        # Apply configurations
        configured_stages = stage.decompose_and_apply_with()

        assert len(configured_stages) == 2

        # Check reader stage configuration
        reader_stage = configured_stages[0]
        assert reader_stage.name == "configured_reader"
        assert reader_stage.batch_size == 8

        # Check download stage configuration
        download_stage = configured_stages[1]
        assert download_stage.name == "configured_download"
        assert download_stage.batch_size == 4

    def test_process_raises_runtime_error(self):
        """Test that process() raises RuntimeError as composite stages shouldn't execute directly."""
        stage = VideoReaderDownloadStage(input_video_path="/test/path")

        # Create a dummy task
        task = _EmptyTask(task_id="test", dataset_name="test", data=None)

        with pytest.raises(RuntimeError) as exc_info:
            stage.process(task)

        error_msg = str(exc_info.value)
        assert "Composite stage 'video_reader_download' should not be executed directly" in error_msg
        assert "It should be decomposed into execution stages during planning" in error_msg

    def test_with_invalid_stage_name(self):
        """Test that with_() with invalid stage name raises error during decompose_and_apply_with."""
        stage = VideoReaderDownloadStage(input_video_path="/test/path")

        stage.with_({
            "invalid_stage_name": {
                "name": "test"
            }
        })

                # Should raise ValueError when trying to apply invalid configuration
        with pytest.raises(ValueError, match="Stage invalid_stage_name not found in composite stage"):
            stage.decompose_and_apply_with()

    def test_post_init_called(self):
        """Test that __post_init__ properly initializes parent CompositeStage."""
        stage = VideoReaderDownloadStage(input_video_path="/test/path")

        # _with_operations should be initialized as empty list
        assert hasattr(stage, "_with_operations")
        assert isinstance(stage._with_operations, list)
        assert len(stage._with_operations) == 0
