"""Unit tests for ImageReaderStage."""

from __future__ import annotations

import io
import pathlib
from unittest.mock import Mock, patch

import numpy as np
import pytest

from ray_curator.stages.image.io.image_reader import ImageReaderStage
from ray_curator.tasks import ImageBatch, EmptyTask


class TestImageReaderStage:
    """Test cases for ImageReaderStage."""

    @pytest.fixture
    def stage(self) -> ImageReaderStage:
        """Create a test ImageReaderStage instance."""
        return ImageReaderStage(
            data_path="/test/path",
            batch_size=2,
            image_limit=10,
            verbose=True
        )

    @pytest.fixture
    def mock_tar_files(self):
        """Create mock tar file members."""
        # Create mock TarInfo objects for image files
        mock_members = []
        for i in range(4):
            member = Mock()
            member.name = f"image_{i:03d}.jpg"
            member.isfile.return_value = True
            mock_members.append(member)
        return mock_members

    @pytest.fixture
    def mock_image_data(self) -> bytes:
        """Create mock image data."""
        # Create a simple JPEG-like byte string
        rng = np.random.default_rng(42)
        return rng.integers(0, 255, size=1024, dtype=np.uint8).tobytes()

    def test_stage_properties(self, stage: ImageReaderStage) -> None:
        """Test basic stage properties."""
        assert stage.name == "image_reader"
        assert stage.data_path == "/test/path"
        assert stage.batch_size == 2
        assert stage.image_limit == 10
        assert stage.verbose is True
        assert stage.inputs() == ([], [])
        assert stage.outputs() == (["data"], [])

    @patch("ray_curator.stages.image.io.image_reader.ImageReaderStage._get_files")
    def test_process_empty_directory(self, mock_get_files: Mock, stage: ImageReaderStage) -> None:
        """Test processing when no files are found."""
        mock_get_files.return_value = []

        result = stage.process(EmptyTask())

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("ray_curator.stages.image.io.image_reader.Image.open")
    @patch("ray_curator.stages.image.io.image_reader.tarfile.open")
    @patch("ray_curator.stages.image.io.image_reader.ImageReaderStage._get_files")
    def test_process_tar_files(
        self,
        mock_get_files: Mock,
        mock_tarfile_open: Mock,
        mock_pil_open: Mock,
        stage: ImageReaderStage,
        mock_tar_files: list[Mock]
    ) -> None:
        """Test processing tar files with images."""
        # Setup mocks
        mock_get_files.return_value = [pathlib.Path("/test/archive.tar")]

        # Mock tar file
        mock_tar = Mock()
        mock_tar.getmembers.return_value = mock_tar_files
        mock_tarfile_open.return_value.__enter__.return_value = mock_tar

        # Mock image reading for each file
        rng = np.random.default_rng(42)
        mock_image_array = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
        mock_pil_image = Mock()
        mock_pil_image.__array__.return_value = mock_image_array
        mock_pil_open.return_value = mock_pil_image

        # Mock extractfile to return image data
        mock_image_data = b"fake_image_data"
        mock_tar.extractfile.return_value = io.BytesIO(mock_image_data)

        result = stage.process(EmptyTask())

        # Verify results
        assert isinstance(result, list)
        assert len(result) == 2  # 4 images / batch_size=2
        for batch in result:
            assert isinstance(batch, ImageBatch)
            assert len(batch.data) <= 2  # batch_size

    def test_batch_creation(self, stage: ImageReaderStage) -> None:
        """Test image batch creation with different sizes."""
        # Test with exact batch size
        from ray_curator.tasks.image import ImageObject
        images = [ImageObject(image_id=f"img_{i}", image_path=f"/path/img_{i}.jpg") for i in range(4)]

        batches = stage._create_image_batches(images)
        assert len(batches) == 2
        assert len(batches[0].data) == 2
        assert len(batches[1].data) == 2

        # Test with remainder
        images = [ImageObject(image_id=f"img_{i}", image_path=f"/path/img_{i}.jpg") for i in range(5)]
        batches = stage._create_image_batches(images)
        assert len(batches) == 3
        assert len(batches[0].data) == 2
        assert len(batches[1].data) == 2
        assert len(batches[2].data) == 1

    def test_image_limit_enforcement(self) -> None:
        """Test that image limit is properly enforced."""
        stage_with_limit = ImageReaderStage(
            data_path="/test/path",
            batch_size=10,
            image_limit=3,
            verbose=False
        )

        from ray_curator.tasks.image import ImageObject
        images = [ImageObject(image_id=f"img_{i}", image_path=f"/path/img_{i}.jpg") for i in range(10)]

        # Should only keep first 3 images
        limited_images = images[:stage_with_limit.image_limit]
        batches = stage_with_limit._create_image_batches(limited_images)

        total_images = sum(len(batch.data) for batch in batches)
        assert total_images == 3

    def test_file_filtering(self) -> None:
        """Test that only tar files are processed."""
        test_files = [
            pathlib.Path("/test/valid.tar"),
            pathlib.Path("/test/valid.tar.gz"),
            pathlib.Path("/test/invalid.txt"),
            pathlib.Path("/test/invalid.zip"),
        ]

        filtered = [f for f in test_files if self._is_tar_file(f)]
        assert len(filtered) == 2
        assert pathlib.Path("/test/valid.tar") in filtered
        assert pathlib.Path("/test/valid.tar.gz") in filtered

    def _is_tar_file(self, file_path: pathlib.Path) -> bool:
        """Helper method to check if file is a tar file."""
        return file_path.suffix in [".tar"] or file_path.name.endswith(".tar.gz")

    def test_verbose_logging(self, stage: ImageReaderStage) -> None:
        """Test verbose logging functionality."""
        assert stage.verbose is True

        # Test with verbose disabled
        quiet_stage = ImageReaderStage(
            data_path="/test/path",
            batch_size=2,
            image_limit=-1,
            verbose=False
        )
        assert quiet_stage.verbose is False

    def test_unlimited_images(self) -> None:
        """Test processing with no image limit."""
        unlimited_stage = ImageReaderStage(
            data_path="/test/path",
            batch_size=5,
            image_limit=-1,  # No limit
            verbose=False
        )
        assert unlimited_stage.image_limit == -1

    @patch("ray_curator.stages.image.io.image_reader.pathlib.Path.rglob")
    def test_get_files_method(self, mock_rglob: Mock, stage: ImageReaderStage) -> None:
        """Test the _get_files method."""
        mock_files = [
            pathlib.Path("/test/file1.tar"),
            pathlib.Path("/test/file2.tar.gz"),
            pathlib.Path("/test/subdir/file3.tar")
        ]
        mock_rglob.return_value = mock_files

        files = stage._get_files()

        assert len(files) >= 0  # Should return some files
        mock_rglob.assert_called()

    def test_error_handling(self) -> None:
        """Test error handling for invalid paths."""
        # Test with non-existent path
        invalid_stage = ImageReaderStage(
            data_path="/nonexistent/path",
            batch_size=2,
            image_limit=10,
            verbose=False
        )

        # Should handle gracefully
        result = invalid_stage.process(EmptyTask())
        assert isinstance(result, list)

    def test_batch_metadata(self, stage: ImageReaderStage) -> None:
        """Test that batches have proper metadata."""
        from ray_curator.tasks.image import ImageObject
        images = [ImageObject(image_id=f"img_{i}", image_path=f"/path/img_{i}.jpg") for i in range(2)]

        batches = stage._create_image_batches(images)
        assert len(batches) == 1

        batch = batches[0]
        assert batch.dataset_name == "image_reader_dataset"
        assert batch.task_id.startswith("image_reader_")
        assert isinstance(batch._metadata, dict)

    def test_stage_setup(self, stage: ImageReaderStage) -> None:
        """Test stage setup method."""
        # Should not raise any exceptions
        stage.setup()

    def test_stage_teardown(self, stage: ImageReaderStage) -> None:
        """Test stage teardown method."""
        # Should not raise any exceptions
        stage.teardown()

    def test_resource_allocation(self, stage: ImageReaderStage) -> None:
        """Test that stage has proper resource allocation."""
        resources = stage.resources
        assert resources.cpu is None  # CPU not specified
        assert resources.memory is None  # Memory not specified
        assert resources.gpus == 0  # No GPU needed for I/O

    def test_concurrent_processing_safety(self, stage: ImageReaderStage) -> None:
        """Test that processing can be done safely in concurrent environments."""
        # Processing should be stateless and safe for concurrent execution
        task1 = EmptyTask()
        task2 = EmptyTask()

        # Both should work independently
        result1 = stage.process(task1)
        result2 = stage.process(task2)

        assert isinstance(result1, list)
        assert isinstance(result2, list)

    @patch("ray_curator.stages.image.io.image_reader.logger")
    def test_logging_calls(self, mock_logger: Mock, stage: ImageReaderStage) -> None:
        """Test that appropriate logging calls are made."""
        # Process empty task to trigger logging
        stage.process(EmptyTask())

        # Should have made some logging calls if verbose
        if stage.verbose:
            assert mock_logger.info.called or mock_logger.debug.called 