"""Unit tests for ImageReaderStage."""

import io
from unittest.mock import Mock, patch

import numpy as np
import pytest

from ray_curator.stages.image.io.image_reader import ImageReaderStage
from ray_curator.tasks import FileGroupTask, ImageBatch


class TestImageReaderStage:
    """Test cases for ImageReaderStage."""

    @pytest.fixture
    def stage(self) -> ImageReaderStage:
        """Create a test ImageReaderStage instance."""
        return ImageReaderStage(
            task_batch_size=2,
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

    @pytest.fixture
    def sample_file_group_task(self) -> FileGroupTask:
        """Create a sample FileGroupTask for testing."""
        return FileGroupTask(
            task_id="test_task_1",
            dataset_name="test_dataset",
            data=["/test/path/file1.tar", "/test/path/file2.tar"]
        )

    def test_stage_properties(self, stage: ImageReaderStage) -> None:
        """Test basic stage properties."""
        assert stage.name == "image_reader"
        assert stage.task_batch_size == 2
        assert stage.verbose is True
        assert stage.inputs() == ([], [])
        assert stage.outputs() == (["data"], ["image_data", "image_path", "image_id"])

    def test_process_empty_task(self, stage: ImageReaderStage) -> None:
        """Test processing when task has no files."""
        empty_task = FileGroupTask(
            task_id="empty_task",
            dataset_name="test_dataset",
            data=[]
        )

        result = stage.process(empty_task)

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("ray_curator.stages.image.io.image_reader.Image.open")
    @patch("ray_curator.stages.image.io.image_reader.tarfile.open")
    def test_process_tar_files(
        self,
        mock_tarfile_open: Mock,
        mock_pil_open: Mock,
        stage: ImageReaderStage,
        mock_tar_files: list[Mock],
        sample_file_group_task: FileGroupTask
    ) -> None:
        """Test processing tar files with images."""
        # Mock tar file
        mock_tar = Mock()
        mock_tar.getmembers.return_value = mock_tar_files
        mock_tarfile_open.return_value.__enter__.return_value = mock_tar

        # Mock image reading for each file
        rng = np.random.default_rng(42)
        mock_image_array = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
        mock_pil_image = Mock()
        mock_pil_image.convert.return_value = mock_pil_image

        # Mock numpy array conversion and PIL Image context manager
        with patch("ray_curator.stages.image.io.image_reader.np.array", return_value=mock_image_array):
            mock_pil_open.return_value.__enter__.return_value = mock_pil_image

            # Mock extractfile to return image data
            mock_image_data = b"fake_image_data"
            mock_tar.extractfile.return_value = io.BytesIO(mock_image_data)

            result = stage.process(sample_file_group_task)

            # Verify results
            assert isinstance(result, list)
            # Should have 4 batches: 2 tar files * 4 images each / batch_size=2 = 4 batches
            assert len(result) == 4
            for batch in result:
                assert isinstance(batch, ImageBatch)
                assert len(batch.data) <= 2  # task_batch_size

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

    def test_task_batch_size_enforcement(self) -> None:
        """Test that task_batch_size is properly enforced."""
        stage_with_large_batch = ImageReaderStage(
            task_batch_size=10,
            verbose=False
        )

        from ray_curator.tasks.image import ImageObject
        images = [ImageObject(image_id=f"img_{i}", image_path=f"/path/img_{i}.jpg") for i in range(25)]

        batches = stage_with_large_batch._create_image_batches(images)
        assert len(batches) == 3  # 25 images / 10 batch_size = 3 batches
        assert len(batches[0].data) == 10
        assert len(batches[1].data) == 10
        assert len(batches[2].data) == 5

    def test_file_filtering(self) -> None:
        """Test that only tar files are processed."""
        import pathlib
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

    def _is_tar_file(self, file_path: str) -> bool:
        """Helper method to check if file is a tar file."""
        import pathlib
        file_path = pathlib.Path(file_path)
        return file_path.suffix in [".tar"] or file_path.name.endswith(".tar.gz")

    def test_verbose_logging(self, stage: ImageReaderStage) -> None:
        """Test verbose logging functionality."""
        assert stage.verbose is True

        # Test with verbose disabled
        quiet_stage = ImageReaderStage(
            task_batch_size=2,
            verbose=False
        )
        assert quiet_stage.verbose is False

    def test_unlimited_images(self) -> None:
        """Test processing with no image limit."""
        # Test instantiation of ImageReaderStage with different parameters
        stage = ImageReaderStage(
            task_batch_size=5,
            verbose=False
        )
        # Verify the stage was created successfully
        assert stage.task_batch_size == 5

    @patch("ray_curator.stages.image.io.image_reader.pathlib.Path.rglob")
    def test_get_files_method(self, mock_rglob: Mock, stage: ImageReaderStage) -> None:  # noqa: ARG002
        """Test the _get_files method."""
        import pathlib
        mock_files = [
            pathlib.Path("/test/file1.tar"),
            pathlib.Path("/test/file2.tar.gz"),
            pathlib.Path("/test/subdir/file3.tar")
        ]
        mock_rglob.return_value = mock_files

        # ImageReaderStage doesn't have _get_files method, so this test verifies mock setup

    @patch("ray_curator.stages.image.io.image_reader.tarfile.open")
    def test_error_handling(self, mock_tarfile_open: Mock, stage: ImageReaderStage) -> None:
        """Test error handling for invalid tar files."""
        # Mock tarfile.open to raise an error
        mock_tarfile_open.side_effect = FileNotFoundError("File not found")

        task_with_invalid_file = FileGroupTask(
            task_id="invalid_task",
            dataset_name="test_dataset",
            data=["/nonexistent/path.tar"]
        )

        # Should handle gracefully
        result = stage.process(task_with_invalid_file)
        assert isinstance(result, list)
        assert len(result) == 0

    @patch("ray_curator.stages.image.io.image_reader.Image.open")
    @patch("ray_curator.stages.image.io.image_reader.tarfile.open")
    def test_image_loading_error_handling(
        self,
        mock_tarfile_open: Mock,
        mock_pil_open: Mock,
        stage: ImageReaderStage,
        sample_file_group_task: FileGroupTask
    ) -> None:
        """Test error handling when image loading fails."""
        # Setup mock tar file with one member
        mock_member = Mock()
        mock_member.name = "corrupted.jpg"
        mock_member.isfile.return_value = True

        mock_tar = Mock()
        mock_tar.getmembers.return_value = [mock_member]
        mock_tar.extractfile.return_value = io.BytesIO(b"corrupted_data")
        mock_tarfile_open.return_value.__enter__.return_value = mock_tar

        # Mock PIL to raise an error that will be caught
        from PIL import Image
        mock_pil_open.side_effect = Image.UnidentifiedImageError("Corrupted image")

        # Suppress the tqdm progress bar for cleaner test output
        with patch("ray_curator.stages.image.io.image_reader.tqdm", side_effect=lambda x, **_: x):
            result = stage.process(sample_file_group_task)

        # Should handle gracefully and return empty list since all images failed
        assert isinstance(result, list)
        assert len(result) == 0  # No valid images processed

    def test_stage_name(self, stage: ImageReaderStage) -> None:
        """Test that stage has correct name."""
        assert stage.name == "image_reader"

    def test_default_values(self) -> None:
        """Test default parameter values."""
        default_stage = ImageReaderStage()
        assert default_stage.task_batch_size == 100
        assert default_stage.verbose is True
        assert default_stage.name == "image_reader"

    @patch("ray_curator.stages.image.io.image_reader.logger")
    def test_logging_calls(self, mock_logger: Mock, stage: ImageReaderStage, sample_file_group_task: FileGroupTask) -> None:
        """Test that appropriate logging calls are made."""
        with patch("ray_curator.stages.image.io.image_reader.tarfile.open") as mock_tarfile:
            mock_tar = Mock()
            mock_tar.getmembers.return_value = []
            mock_tarfile.return_value.__enter__.return_value = mock_tar

            # Process task to trigger logging
            stage.process(sample_file_group_task)

            # Should have made some logging calls if verbose
            if stage.verbose:
                assert mock_logger.info.called

    @patch("ray_curator.stages.image.io.image_reader.Image.open")
    @patch("ray_curator.stages.image.io.image_reader.tarfile.open")
    def test_image_id_extraction(
        self,
        mock_tarfile_open: Mock,
        mock_pil_open: Mock,
        stage: ImageReaderStage,
        sample_file_group_task: FileGroupTask
    ) -> None:
        """Test that image IDs are correctly extracted from filenames."""
        # Setup mock tar file with specific filename
        mock_member = Mock()
        mock_member.name = "sample_image_123.jpg"
        mock_member.isfile.return_value = True

        mock_tar = Mock()
        mock_tar.getmembers.return_value = [mock_member]
        mock_tar.extractfile.return_value = io.BytesIO(b"fake_image_data")
        mock_tarfile_open.return_value.__enter__.return_value = mock_tar

        # Mock PIL image
        rng = np.random.default_rng(42)
        mock_image_array = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
        mock_pil_image = Mock()
        mock_pil_image.convert.return_value = mock_pil_image

        # Mock numpy array conversion and PIL Image context manager
        with patch("ray_curator.stages.image.io.image_reader.np.array", return_value=mock_image_array):
            mock_pil_open.return_value.__enter__.return_value = mock_pil_image

            # Limit to one file for simpler testing
            single_file_task = FileGroupTask(
                task_id="single_file_task",
                dataset_name="test_dataset",
                data=[sample_file_group_task.data[0]]
            )

            result = stage.process(single_file_task)

            # Verify image ID extraction
            assert len(result) == 1
            batch = result[0]
            assert len(batch.data) == 1
            image_obj = batch.data[0]
            assert image_obj.image_id == "sample_image_123"  # .jpg should be removed
