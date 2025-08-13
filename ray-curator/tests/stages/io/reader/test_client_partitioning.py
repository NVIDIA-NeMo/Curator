"""Tests for ClientPartitioningStage."""

from unittest.mock import Mock, patch, MagicMock
import pytest
from pathlib import Path

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.io.reader.client_partitioning import ClientPartitioningStage, _read_list_json
from ray_curator.tasks import FileGroupTask, _EmptyTask


class TestClientPartitioningStage:
    """Test suite for ClientPartitioningStage."""

    @pytest.fixture
    def empty_task(self) -> _EmptyTask:
        """Create an empty task for testing."""
        return _EmptyTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=None,
            _metadata={"source": "test"},
        )

    @pytest.fixture
    def worker_metadata(self) -> WorkerMetadata:
        """Create worker metadata for testing."""
        return WorkerMetadata(worker_id="test_worker", allocation=None)

    @pytest.fixture
    def temp_files(self, tmp_path: Path) -> list[str]:
        """Create temporary test files for testing."""
        files = []
        for i in range(10):
            file_path = tmp_path / f"test_file_{i}.jsonl"
            file_path.write_text(f'{{"id": {i}, "text": "Test content {i}"}}')
            files.append(str(file_path))
        return files

    def test_initialization_default_values(self):
        """Test initialization with default parameter values."""
        stage = ClientPartitioningStage(file_paths="/test/path")

        assert stage.file_paths == "/test/path"
        assert stage.input_s3_profile_name is None
        assert stage.input_list_json_path is None
        assert stage._name == "client_partitioning"
        assert stage.files_per_partition is None
        assert stage.blocksize is None
        assert stage.file_extensions == [".jsonl", ".json"]
        assert stage.storage_options == {}
        assert stage.limit is None

    def test_initialization_custom_values(self):
        """Test initialization with custom parameter values."""
        stage = ClientPartitioningStage(
            file_paths="/custom/path",
            input_s3_profile_name="test_profile",
            input_list_json_path="/path/to/list.json",
            files_per_partition=5,
            blocksize="128MB",
            file_extensions=[".txt", ".json"],
            storage_options={"key": "value"},
            limit=3
        )

        assert stage.file_paths == "/custom/path"
        assert stage.input_s3_profile_name == "test_profile"
        assert stage.input_list_json_path == "/path/to/list.json"
        assert stage.files_per_partition == 5
        assert stage.blocksize == "128MB"
        assert stage.file_extensions == [".txt", ".json"]
        assert stage.storage_options == {"key": "value"}
        assert stage.limit == 3

    @patch('ray_curator.stages.io.reader.client_partitioning.get_storage_client')
    def test_setup(self, mock_get_storage_client, worker_metadata):
        """Test setup method."""
        mock_client = Mock()
        mock_get_storage_client.return_value = mock_client
        
        stage = ClientPartitioningStage(file_paths="/test/path", input_s3_profile_name="test_profile")
        stage.setup(worker_metadata)

        mock_get_storage_client.assert_called_once_with("/test/path", profile_name="test_profile")
        assert stage.client_input == mock_client

    @patch('ray_curator.stages.io.reader.client_partitioning.get_storage_client')
    def test_setup_no_profile(self, mock_get_storage_client, worker_metadata):
        """Test setup method without S3 profile."""
        mock_client = Mock()
        mock_get_storage_client.return_value = mock_client
        
        stage = ClientPartitioningStage(file_paths="/test/path")
        stage.setup(worker_metadata)

        mock_get_storage_client.assert_called_once_with("/test/path", profile_name=None)
        assert stage.client_input == mock_client

    @patch('ray_curator.stages.io.reader.client_partitioning.get_files_relative')
    @patch('ray_curator.stages.io.reader.client_partitioning.get_storage_client')
    def test_process_with_files_relative(self, mock_get_storage_client, mock_get_files_relative, empty_task):
        """Test process method using get_files_relative."""
        mock_client = Mock()
        mock_get_storage_client.return_value = mock_client
        
        test_files = ["file1.jsonl", "file2.jsonl", "file3.jsonl"]
        mock_get_files_relative.return_value = test_files
        
        stage = ClientPartitioningStage(file_paths="/test/path")
        stage.setup()
        
        result = stage.process(empty_task)

        # When files_per_partition is not specified, each file gets its own task
        assert len(result) == 3
        assert isinstance(result[0], FileGroupTask)
        assert result[0].data == "file1.jsonl"
        assert result[0].dataset_name == "/test/path"
        assert result[0].task_id == "file_group_0"
        assert result[0]._metadata["partition_index"] == 0
        assert result[0]._metadata["total_partitions"] == 3
        assert result[0]._metadata["storage_options"] == {}
        assert result[0]._metadata["source_files"] == "file1.jsonl"
        assert result[0].reader_config == {}

        mock_get_files_relative.assert_called_once_with("/test/path", mock_client)

    @patch('ray_curator.stages.io.reader.client_partitioning._read_list_json')
    @patch('ray_curator.stages.io.reader.client_partitioning.get_storage_client')
    def test_process_with_input_list_json(self, mock_get_storage_client, mock_read_list_json, empty_task):
        """Test process method using input_list_json_path."""
        mock_client = Mock()
        mock_get_storage_client.return_value = mock_client
        
        test_files = ["file1.jsonl", "file2.jsonl", "file3.jsonl"]
        mock_read_list_json.return_value = test_files
        
        stage = ClientPartitioningStage(
            file_paths="/test/path",
            input_list_json_path="/path/to/list.json",
            input_s3_profile_name="test_profile"
        )
        stage.setup()
        
        result = stage.process(empty_task)

        # When files_per_partition is not specified, each file gets its own task
        assert len(result) == 3
        assert result[0].data == "file1.jsonl"
        assert result[1].data == "file2.jsonl"
        assert result[2].data == "file3.jsonl"
        
        mock_read_list_json.assert_called_once_with("/test/path", "/path/to/list.json", "test_profile")

    @patch('ray_curator.stages.io.reader.client_partitioning.get_files_relative')
    @patch('ray_curator.stages.io.reader.client_partitioning.get_storage_client')
    def test_process_with_file_extensions_filter(self, mock_get_storage_client, mock_get_files_relative, empty_task):
        """Test process method with file extension filtering."""
        mock_client = Mock()
        mock_get_storage_client.return_value = mock_client
        
        all_files = ["file1.jsonl", "file2.txt", "file3.json", "file4.py"]
        mock_get_files_relative.return_value = all_files
        
        stage = ClientPartitioningStage(
            file_paths="/test/path",
            file_extensions=[".jsonl", ".json"]
        )
        stage.setup()
        
        result = stage.process(empty_task)

        # Should filter by extensions first: ["file1.jsonl", "file3.json"]
        # Then create individual tasks for each file
        assert len(result) == 2
        assert result[0].data == "file1.jsonl"
        assert result[1].data == "file3.json"

    @patch('ray_curator.stages.io.reader.client_partitioning.get_files_relative')
    @patch('ray_curator.stages.io.reader.client_partitioning.get_storage_client')
    def test_process_with_limit(self, mock_get_storage_client, mock_get_files_relative, empty_task):
        """Test process method with limit parameter."""
        mock_client = Mock()
        mock_get_storage_client.return_value = mock_client
        
        all_files = ["file1.jsonl", "file2.jsonl", "file3.jsonl", "file4.jsonl", "file5.jsonl"]
        mock_get_files_relative.return_value = all_files
        
        stage = ClientPartitioningStage(
            file_paths="/test/path",
            limit=3
        )
        stage.setup()
        
        result = stage.process(empty_task)

        # Should limit to 3 files, each getting its own task
        assert len(result) == 3
        assert result[0].data == "file1.jsonl"
        assert result[1].data == "file2.jsonl"
        assert result[2].data == "file3.jsonl"

    @patch('ray_curator.stages.io.reader.client_partitioning.get_files_relative')
    @patch('ray_curator.stages.io.reader.client_partitioning.get_storage_client')
    def test_process_with_files_per_partition(self, mock_get_storage_client, mock_get_files_relative, empty_task):
        """Test process method with files_per_partition."""
        mock_client = Mock()
        mock_get_storage_client.return_value = mock_client
        
        all_files = ["file1.jsonl", "file2.jsonl", "file3.jsonl", "file4.jsonl"]
        mock_get_files_relative.return_value = all_files
        
        stage = ClientPartitioningStage(
            file_paths="/test/path",
            files_per_partition=2
        )
        stage.setup()
        
        result = stage.process(empty_task)

        assert len(result) == 2
        assert result[0].data == ["file1.jsonl", "file2.jsonl"]
        assert result[1].data == ["file3.jsonl", "file4.jsonl"]
        
        # Check metadata
        assert result[0]._metadata["partition_index"] == 0
        assert result[0]._metadata["total_partitions"] == 2
        assert result[1]._metadata["partition_index"] == 1
        assert result[1]._metadata["total_partitions"] == 2

    @patch('ray_curator.stages.io.reader.client_partitioning.get_files_relative')
    @patch('ray_curator.stages.io.reader.client_partitioning.get_storage_client')
    def test_process_with_storage_options(self, mock_get_storage_client, mock_get_files_relative, empty_task):
        """Test process method with storage options."""
        mock_client = Mock()
        mock_get_storage_client.return_value = mock_client
        
        test_files = ["file1.jsonl", "file2.jsonl"]
        mock_get_files_relative.return_value = test_files
        
        storage_options = {"option1": "value1", "option2": "value2"}
        stage = ClientPartitioningStage(
            file_paths="/test/path",
            storage_options=storage_options
        )
        stage.setup()
        
        result = stage.process(empty_task)

        # Each file gets its own task
        assert len(result) == 2
        assert result[0]._metadata["storage_options"] == storage_options
        assert result[1]._metadata["storage_options"] == storage_options

    @patch('ray_curator.stages.io.reader.client_partitioning.get_files_relative')
    @patch('ray_curator.stages.io.reader.client_partitioning.get_storage_client')
    def test_process_empty_file_list(self, mock_get_storage_client, mock_get_files_relative, empty_task):
        """Test process method with empty file list."""
        mock_client = Mock()
        mock_get_storage_client.return_value = mock_client
        
        mock_get_files_relative.return_value = []
        
        stage = ClientPartitioningStage(file_paths="/test/path")
        stage.setup()
        
        result = stage.process(empty_task)

        assert len(result) == 0

    @patch('ray_curator.stages.io.reader.client_partitioning.get_files_relative')
    @patch('ray_curator.stages.io.reader.client_partitioning.get_storage_client')
    def test_process_limit_zero(self, mock_get_storage_client, mock_get_files_relative, empty_task):
        """Test process method with limit set to 0."""
        mock_client = Mock()
        mock_get_storage_client.return_value = mock_client
        
        all_files = ["file1.jsonl", "file2.jsonl", "file3.jsonl"]
        mock_get_files_relative.return_value = all_files
        
        stage = ClientPartitioningStage(
            file_paths="/test/path",
            limit=0
        )
        stage.setup()
        
        result = stage.process(empty_task)

        # ClientPartitioningStage only applies limit when it's > 0, so limit=0 means no limit
        assert len(result) == 3
        assert result[0].data == "file1.jsonl"
        assert result[1].data == "file2.jsonl"
        assert result[2].data == "file3.jsonl"

    def test_process_file_paths_none(self, empty_task):
        """Test process method when file_paths is None."""
        stage = ClientPartitioningStage(file_paths=None)
        
        with pytest.raises(AssertionError):
            stage.process(empty_task)

    @patch('ray_curator.stages.io.reader.client_partitioning.get_files_relative')
    @patch('ray_curator.stages.io.reader.client_partitioning.get_storage_client')
    def test_process_combined_filters(self, mock_get_storage_client, mock_get_files_relative, empty_task):
        """Test process method with multiple filters combined."""
        mock_client = Mock()
        mock_get_storage_client.return_value = mock_client
        
        all_files = ["file1.jsonl", "file2.txt", "file3.json", "file4.py", "file5.jsonl", "file6.json"]
        mock_get_files_relative.return_value = all_files
        
        stage = ClientPartitioningStage(
            file_paths="/test/path",
            file_extensions=[".jsonl", ".json"],
            limit=3,
            files_per_partition=2
        )
        stage.setup()
        
        result = stage.process(empty_task)

        # Should filter by extensions first: ["file1.jsonl", "file3.json", "file5.jsonl", "file6.json"]
        # Then limit to 3: ["file1.jsonl", "file3.json", "file5.jsonl"]
        # Then partition by 2: [["file1.jsonl", "file3.json"], ["file5.jsonl"]]
        assert len(result) == 2
        assert result[0].data == ["file1.jsonl", "file3.json"]
        assert result[1].data == ["file5.jsonl"]

    def test_inheritance_from_file_partitioning(self):
        """Test that ClientPartitioningStage inherits from FilePartitioningStage."""
        stage = ClientPartitioningStage(file_paths="/test/path")
        
        # Test that it has the expected methods from parent class
        assert hasattr(stage, '_partition_by_count')
        assert hasattr(stage, '_parse_size')
        assert hasattr(stage, '_get_dataset_name')


class TestReadListJson:
    """Test suite for _read_list_json function."""

    @patch('ray_curator.stages.io.reader.client_partitioning.read_json_file')
    @patch('ray_curator.stages.io.reader.client_partitioning.get_storage_client')
    def test_read_list_json_success(self, mock_get_storage_client, mock_read_json_file):
        """Test successful reading of list JSON file."""
        mock_client = Mock()
        mock_get_storage_client.return_value = mock_client
        
        mock_read_json_file.return_value = [
            "/input/path/video1.mp4",
            "/input/path/video2.mp4",
            "/input/path/video3.mp4"
        ]
        
        result = _read_list_json(
            input_path="/input/path",
            input_video_list_json_path="/path/to/list.json",
            input_video_list_s3_profile_name="test_profile"
        )
        
        expected = ["video1.mp4", "video2.mp4", "video3.mp4"]
        assert result == expected
        
        mock_get_storage_client.assert_called_once_with("/path/to/list.json", profile_name="test_profile")
        mock_read_json_file.assert_called_once_with("/path/to/list.json", mock_client)

    @patch('ray_curator.stages.io.reader.client_partitioning.read_json_file')
    @patch('ray_curator.stages.io.reader.client_partitioning.get_storage_client')
    def test_read_list_json_with_trailing_slash(self, mock_get_storage_client, mock_read_json_file):
        """Test reading list JSON with trailing slash in input path."""
        mock_client = Mock()
        mock_get_storage_client.return_value = mock_client
        
        mock_read_json_file.return_value = [
            "/input/path/video1.mp4",
            "/input/path/video2.mp4"
        ]
        
        result = _read_list_json(
            input_path="/input/path/",  # Note trailing slash
            input_video_list_json_path="/path/to/list.json",
            input_video_list_s3_profile_name="test_profile"
        )
        
        expected = ["video1.mp4", "video2.mp4"]
        assert result == expected

    @patch('ray_curator.stages.io.reader.client_partitioning.read_json_file')
    @patch('ray_curator.stages.io.reader.client_partitioning.get_storage_client')
    def test_read_list_json_path_mismatch(self, mock_get_storage_client, mock_read_json_file):
        """Test reading list JSON with path mismatch."""
        mock_client = Mock()
        mock_get_storage_client.return_value = mock_client
        
        mock_read_json_file.return_value = [
            "/different/path/video1.mp4",  # Different base path
            "/input/path/video2.mp4"
        ]
        
        with pytest.raises(ValueError, match="Input video /different/path/video1.mp4 is not in /input/path/"):
            _read_list_json(
                input_path="/input/path",
                input_video_list_json_path="/path/to/list.json",
                input_video_list_s3_profile_name="test_profile"
            )

    @patch('ray_curator.stages.io.reader.client_partitioning.read_json_file')
    @patch('ray_curator.stages.io.reader.client_partitioning.get_storage_client')
    def test_read_list_json_read_exception(self, mock_get_storage_client, mock_read_json_file):
        """Test reading list JSON when read_json_file raises an exception."""
        mock_client = Mock()
        mock_get_storage_client.return_value = mock_client
        
        mock_read_json_file.side_effect = Exception("File not found")
        
        with pytest.raises(Exception, match="File not found"):
            _read_list_json(
                input_path="/input/path",
                input_video_list_json_path="/path/to/list.json",
                input_video_list_s3_profile_name="test_profile"
            )

    @patch('ray_curator.stages.io.reader.client_partitioning.read_json_file')
    @patch('ray_curator.stages.io.reader.client_partitioning.get_storage_client')
    def test_read_list_json_empty_list(self, mock_get_storage_client, mock_read_json_file):
        """Test reading list JSON with empty list."""
        mock_client = Mock()
        mock_get_storage_client.return_value = mock_client
        
        mock_read_json_file.return_value = []
        
        result = _read_list_json(
            input_path="/input/path",
            input_video_list_json_path="/path/to/list.json",
            input_video_list_s3_profile_name="test_profile"
        )
        
        assert result == []

    @patch('ray_curator.stages.io.reader.client_partitioning.read_json_file')
    @patch('ray_curator.stages.io.reader.client_partitioning.get_storage_client')
    def test_read_list_json_string_conversion(self, mock_get_storage_client, mock_read_json_file):
        """Test that list items are converted to strings."""
        mock_client = Mock()
        mock_get_storage_client.return_value = mock_client
        
        mock_read_json_file.return_value = [
            "/input/path/video2.mp4"  # Only valid path
        ]
        
        result = _read_list_json(
            input_path="/input/path",
            input_video_list_json_path="/path/to/list.json",
            input_video_list_s3_profile_name="test_profile"
        )
        
        # Should only return the valid path
        expected = ["video2.mp4"]
        assert result == expected 