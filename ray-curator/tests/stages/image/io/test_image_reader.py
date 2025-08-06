"""Unit tests for ImageReaderStage."""

import io
import pathlib
import tarfile
import tempfile
from unittest.mock import Mock, patch, mock_open
import pytest
import numpy as np
from PIL import Image

from ray_curator.stages.image.io.image_reader import ImageReaderStage
from ray_curator.tasks import ImageBatch, ImageObject, EmptyTask


class TestImageReaderStage:
    """Test cases for ImageReaderStage."""

    @pytest.fixture
    def stage(self):
        """Create a test ImageReaderStage instance."""
        return ImageReaderStage(
            input_dataset_path="test_dataset",
            image_limit=10,
            batch_size=2,
            verbose=True
        )

    @pytest.fixture
    def mock_tar_files(self):
        """Create mock tar file paths."""
        return [
            pathlib.Path("test_dataset/shard_001.tar"),
            pathlib.Path("test_dataset/shard_002.tar")
        ]

    @pytest.fixture
    def mock_image_data(self):
        """Create mock RGB image data."""
        # Create a simple 10x10 RGB image
        return np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

    @pytest.fixture
    def mock_pil_image(self, mock_image_data):
        """Create a mock PIL Image."""
        mock_img = Mock()
        mock_img.convert.return_value = mock_img
        # When np.array is called on the PIL image, return our mock data
        with patch('numpy.array', return_value=mock_image_data):
            yield mock_img

    def test_stage_properties(self, stage):
        """Test stage properties and configuration."""
        assert stage.input_dataset_path == "test_dataset"
        assert stage.image_limit == 10
        assert stage.batch_size == 2
        assert stage.verbose is True
        assert stage._name == "image_reader"

    def test_inputs_outputs(self, stage):
        """Test inputs and outputs specifications."""
        inputs = stage.inputs()
        outputs = stage.outputs()
        
        assert inputs == ([], [])
        assert outputs == (["data"], ["image_data", "image_path", "image_id"])

    @patch('ray_curator.stages.image.io.image_reader.get_all_files_paths_under')
    def test_no_tar_files_found(self, mock_get_files, stage):
        """Test behavior when no tar files are found."""
        mock_get_files.return_value = []
        
        result = stage.process(EmptyTask)
        
        assert result == []
        mock_get_files.assert_called_once_with(
            "test_dataset",
            recurse_subdirectories=True,
            keep_extensions=[".tar"]
        )

    @patch('ray_curator.stages.image.io.image_reader.get_all_files_paths_under')
    @patch('tarfile.open')
    @patch('PIL.Image.open')
    def test_successful_image_loading(self, mock_pil_open, mock_tarfile_open, mock_get_files, 
                                    stage, mock_tar_files, mock_pil_image, mock_image_data):
        """Test successful loading of images from tar files."""
        mock_get_files.return_value = mock_tar_files[:1]  # Use only one tar file
        
        # Mock tar file contents
        mock_tar = Mock()
        mock_tarfile_open.return_value.__enter__.return_value = mock_tar
        
        # Create mock tar members (image files)
        mock_member1 = Mock()
        mock_member1.name = "image_001.jpg"
        mock_member1.isfile.return_value = True
        
        mock_member2 = Mock()
        mock_member2.name = "image_002.jpg"
        mock_member2.isfile.return_value = True
        
        mock_tar.getmembers.return_value = [mock_member1, mock_member2]
        
        # Mock image file extraction
        mock_image_file1 = io.BytesIO(b"fake_image_data_1")
        mock_image_file2 = io.BytesIO(b"fake_image_data_2")
        mock_tar.extractfile.side_effect = [mock_image_file1, mock_image_file2]
        
        # Mock PIL Image.open
        mock_pil_open.return_value.__enter__.return_value = mock_pil_image
        
        # Mock numpy array conversion
        with patch('numpy.array', return_value=mock_image_data):
            result = stage.process(EmptyTask)
        
        # Verify results
        assert len(result) == 1  # One batch with batch_size=2
        assert isinstance(result[0], ImageBatch)
        assert len(result[0].data) == 2
        
        # Check first image object
        img_obj1 = result[0].data[0]
        assert img_obj1.image_id == "image_001"
        assert "image_001.jpg" in img_obj1.image_path
        assert np.array_equal(img_obj1.image_data, mock_image_data)
        
        # Check second image object
        img_obj2 = result[0].data[1]
        assert img_obj2.image_id == "image_002"
        assert "image_002.jpg" in img_obj2.image_path
        assert np.array_equal(img_obj2.image_data, mock_image_data)

    @patch('ray_curator.stages.image.io.image_reader.get_all_files_paths_under')
    @patch('tarfile.open')
    @patch('PIL.Image.open')
    def test_image_limit_enforcement(self, mock_pil_open, mock_tarfile_open, mock_get_files,
                                   mock_tar_files, mock_pil_image, mock_image_data):
        """Test that image limit is properly enforced."""
        # Set up stage with limit of 3 images
        stage = ImageReaderStage(
            input_dataset_path="test_dataset",
            image_limit=3,
            batch_size=2,
            verbose=True
        )
        
        mock_get_files.return_value = mock_tar_files[:1]
        
        # Mock tar file with 5 images
        mock_tar = Mock()
        mock_tarfile_open.return_value.__enter__.return_value = mock_tar
        
        mock_members = []
        for i in range(5):
            member = Mock()
            member.name = f"image_{i:03d}.jpg"
            member.isfile.return_value = True
            mock_members.append(member)
        
        mock_tar.getmembers.return_value = mock_members
        mock_tar.extractfile.return_value = io.BytesIO(b"fake_image_data")
        
        mock_pil_open.return_value.__enter__.return_value = mock_pil_image
        
        with patch('numpy.array', return_value=mock_image_data):
            result = stage.process(EmptyTask)
        
        # Should have 2 batches: first with 2 images, second with 1 image
        assert len(result) == 2
        assert len(result[0].data) == 2  # First batch: 2 images
        assert len(result[1].data) == 1  # Second batch: 1 image (limited to 3 total)
        
        # Verify only 3 images were loaded despite 5 being available
        total_images = sum(len(batch.data) for batch in result)
        assert total_images == 3

    @patch('ray_curator.stages.image.io.image_reader.get_all_files_paths_under')
    @patch('tarfile.open')
    def test_batch_creation(self, mock_tarfile_open, mock_get_files, mock_tar_files):
        """Test proper creation of ImageBatch objects."""
        stage = ImageReaderStage(
            input_dataset_path="test_dataset",
            batch_size=3,
            verbose=True
        )
        
        mock_get_files.return_value = mock_tar_files[:1]
        
        # Mock tar file with 7 images
        mock_tar = Mock()
        mock_tarfile_open.return_value.__enter__.return_value = mock_tar
        
        mock_members = []
        for i in range(7):
            member = Mock()
            member.name = f"image_{i:03d}.jpg"
            member.isfile.return_value = True
            mock_members.append(member)
        
        mock_tar.getmembers.return_value = mock_members
        mock_tar.extractfile.return_value = io.BytesIO(b"fake_image_data")
        
        mock_image_data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        
        with patch('PIL.Image.open') as mock_pil_open:
            mock_pil_image = Mock()
            mock_pil_image.convert.return_value = mock_pil_image
            mock_pil_open.return_value.__enter__.return_value = mock_pil_image
            
            with patch('numpy.array', return_value=mock_image_data):
                result = stage.process(EmptyTask)
        
        # Should have 3 batches: [3, 3, 1] images
        assert len(result) == 3
        assert len(result[0].data) == 3
        assert len(result[1].data) == 3
        assert len(result[2].data) == 1
        
        # Check batch task IDs
        assert result[0].task_id == "image_batch_0"
        assert result[1].task_id == "image_batch_1"
        assert result[2].task_id == "image_batch_2"
        
        # Check dataset name
        for batch in result:
            assert batch.dataset_name == "test_dataset"

    @patch('ray_curator.stages.image.io.image_reader.get_all_files_paths_under')
    @patch('tarfile.open')
    def test_error_handling_corrupted_tar(self, mock_tarfile_open, mock_get_files, mock_tar_files):
        """Test error handling when tar file is corrupted."""
        mock_get_files.return_value = mock_tar_files[:1]
        
        # Mock tarfile.open to raise an exception
        mock_tarfile_open.side_effect = tarfile.ReadError("Corrupted tar file")
        
        stage = ImageReaderStage(
            input_dataset_path="test_dataset",
            verbose=True
        )
        
        result = stage.process(EmptyTask)
        
        # Should return empty list when tar file is corrupted
        assert result == []

    @patch('ray_curator.stages.image.io.image_reader.get_all_files_paths_under')
    @patch('tarfile.open')
    @patch('PIL.Image.open')
    def test_error_handling_corrupted_image(self, mock_pil_open, mock_tarfile_open, 
                                          mock_get_files, mock_tar_files):
        """Test error handling when individual image is corrupted."""
        mock_get_files.return_value = mock_tar_files[:1]
        
        # Mock tar file
        mock_tar = Mock()
        mock_tarfile_open.return_value.__enter__.return_value = mock_tar
        
        # Create two mock members: one good, one bad
        mock_member1 = Mock()
        mock_member1.name = "good_image.jpg"
        mock_member1.isfile.return_value = True
        
        mock_member2 = Mock()
        mock_member2.name = "bad_image.jpg"
        mock_member2.isfile.return_value = True
        
        mock_tar.getmembers.return_value = [mock_member1, mock_member2]
        mock_tar.extractfile.side_effect = [io.BytesIO(b"good_data"), io.BytesIO(b"bad_data")]
        
        mock_image_data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        
        # Create a proper context manager mock for successful image
        mock_good_image = Mock()
        mock_good_image.convert.return_value = mock_good_image
        mock_good_context = Mock()
        mock_good_context.__enter__.return_value = mock_good_image
        mock_good_context.__exit__.return_value = None
        
        # Mock PIL to succeed for first image, fail for second
        mock_pil_open.side_effect = [
            mock_good_context,  # Success for first image
            Exception("Corrupted image")  # Failure for second image
        ]
        
        stage = ImageReaderStage(
            input_dataset_path="test_dataset",
            verbose=True
        )
        
        with patch('numpy.array', return_value=mock_image_data):
            result = stage.process(EmptyTask)
        
        # Should have successfully loaded only the good image
        assert len(result) == 1
        assert len(result[0].data) == 1
        assert result[0].data[0].image_id == "good_image"

    @patch('ray_curator.stages.image.io.image_reader.get_all_files_paths_under')
    @patch('tarfile.open')
    def test_extractfile_returns_none(self, mock_tarfile_open, mock_get_files, mock_tar_files):
        """Test handling when tar.extractfile returns None."""
        mock_get_files.return_value = mock_tar_files[:1]
        
        mock_tar = Mock()
        mock_tarfile_open.return_value.__enter__.return_value = mock_tar
        
        mock_member = Mock()
        mock_member.name = "image_001.jpg"
        mock_member.isfile.return_value = True
        
        mock_tar.getmembers.return_value = [mock_member]
        mock_tar.extractfile.return_value = None  # Simulate extraction failure
        
        stage = ImageReaderStage(
            input_dataset_path="test_dataset",
            verbose=True
        )
        
        result = stage.process(EmptyTask)
        
        # Should return empty list when extraction fails
        assert result == []

    @patch('ray_curator.stages.image.io.image_reader.get_all_files_paths_under')
    @patch('tarfile.open')
    def test_non_image_files_filtered(self, mock_tarfile_open, mock_get_files, mock_tar_files):
        """Test that non-image files are filtered out."""
        mock_get_files.return_value = mock_tar_files[:1]
        
        mock_tar = Mock()
        mock_tarfile_open.return_value.__enter__.return_value = mock_tar
        
        # Mix of image and non-image files
        mock_member1 = Mock()
        mock_member1.name = "image_001.jpg"
        mock_member1.isfile.return_value = True
        
        mock_member2 = Mock()
        mock_member2.name = "metadata.json"  # Non-image file
        mock_member2.isfile.return_value = True
        
        mock_member3 = Mock()
        mock_member3.name = "image_002.jpg"
        mock_member3.isfile.return_value = True
        
        mock_member4 = Mock()
        mock_member4.name = "readme.txt"  # Non-image file
        mock_member4.isfile.return_value = True
        
        mock_tar.getmembers.return_value = [mock_member1, mock_member2, mock_member3, mock_member4]
        mock_tar.extractfile.return_value = io.BytesIO(b"fake_image_data")
        
        mock_image_data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        
        with patch('PIL.Image.open') as mock_pil_open:
            mock_pil_image = Mock()
            mock_pil_image.convert.return_value = mock_pil_image
            mock_pil_open.return_value.__enter__.return_value = mock_pil_image
            
            with patch('numpy.array', return_value=mock_image_data):
                stage = ImageReaderStage(
                    input_dataset_path="test_dataset",
                    verbose=True
                )
                
                result = stage.process(EmptyTask)
        
        # Should only load the 2 .jpg files
        assert len(result) == 1
        assert len(result[0].data) == 2
        assert result[0].data[0].image_id == "image_001"
        assert result[0].data[1].image_id == "image_002"

    def test_no_input_dataset_path(self):
        """Test error when input_dataset_path is None."""
        stage = ImageReaderStage(input_dataset_path=None)
        
        with pytest.raises(ValueError, match="input_dataset_path is not set"):
            stage.process(EmptyTask)

    @patch('ray_curator.stages.image.io.image_reader.get_all_files_paths_under')
    @patch('tarfile.open')
    @patch('PIL.Image.open')
    def test_string_path_conversion(self, mock_pil_open, mock_tarfile_open, mock_get_files):
        """Test that string paths are properly converted to pathlib.Path objects."""
        # Return string paths instead of pathlib.Path objects
        mock_get_files.return_value = ["test_dataset/shard_001.tar"]
        
        mock_tar = Mock()
        mock_tarfile_open.return_value.__enter__.return_value = mock_tar
        
        mock_member = Mock()
        mock_member.name = "image_001.jpg"
        mock_member.isfile.return_value = True
        
        mock_tar.getmembers.return_value = [mock_member]
        mock_tar.extractfile.return_value = io.BytesIO(b"fake_image_data")
        
        mock_image_data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        mock_pil_image = Mock()
        mock_pil_image.convert.return_value = mock_pil_image
        mock_pil_open.return_value.__enter__.return_value = mock_pil_image
        
        stage = ImageReaderStage(
            input_dataset_path="test_dataset",
            verbose=True
        )
        
        with patch('numpy.array', return_value=mock_image_data):
            result = stage.process(EmptyTask)
        
        # Should successfully process string paths
        assert len(result) == 1
        assert len(result[0].data) == 1
        assert result[0].data[0].image_id == "image_001"

    @patch('ray_curator.stages.image.io.image_reader.get_all_files_paths_under')
    @patch('tarfile.open')
    @patch('PIL.Image.open')
    def test_verbose_logging(self, mock_pil_open, mock_tarfile_open, mock_get_files):
        """Test verbose logging functionality."""
        mock_get_files.return_value = ["test_dataset/shard_001.tar"]
        
        mock_tar = Mock()
        mock_tarfile_open.return_value.__enter__.return_value = mock_tar
        mock_tar.getmembers.return_value = []  # Empty tar file
        
        # Test with verbose=True
        stage_verbose = ImageReaderStage(
            input_dataset_path="test_dataset",
            verbose=True
        )
        
        # Test with verbose=False
        stage_quiet = ImageReaderStage(
            input_dataset_path="test_dataset",
            verbose=False
        )
        
        # Both should work without errors
        result_verbose = stage_verbose.process(EmptyTask)
        result_quiet = stage_quiet.process(EmptyTask)
        
        assert result_verbose == []
        assert result_quiet == []

    @patch('ray_curator.stages.image.io.image_reader.get_all_files_paths_under')
    @patch('tarfile.open')
    @patch('PIL.Image.open')
    def test_multiple_tar_files(self, mock_pil_open, mock_tarfile_open, mock_get_files, mock_tar_files):
        """Test processing multiple tar files."""
        mock_get_files.return_value = mock_tar_files  # Both tar files
        
        mock_image_data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        
        # Set up different mock tars for each file
        def create_mock_tar(tar_index):
            mock_tar = Mock()
            mock_member = Mock()
            mock_member.name = f"image_{tar_index:03d}.jpg"
            mock_member.isfile.return_value = True
            mock_tar.getmembers.return_value = [mock_member]
            mock_tar.extractfile.return_value = io.BytesIO(b"fake_image_data")
            return mock_tar
        
        mock_tarfile_open.return_value.__enter__.side_effect = [
            create_mock_tar(1),
            create_mock_tar(2)
        ]
        
        mock_pil_image = Mock()
        mock_pil_image.convert.return_value = mock_pil_image
        mock_pil_open.return_value.__enter__.return_value = mock_pil_image
        
        stage = ImageReaderStage(
            input_dataset_path="test_dataset",
            batch_size=1,
            verbose=True
        )
        
        with patch('numpy.array', return_value=mock_image_data):
            result = stage.process(EmptyTask)
        
        # Should have processed images from both tar files
        assert len(result) == 2  # 2 batches with batch_size=1
        assert len(result[0].data) == 1
        assert len(result[1].data) == 1
        
        # Images should be from different tar files
        image_ids = {result[0].data[0].image_id, result[1].data[0].image_id}
        assert image_ids == {"image_001", "image_002"} 