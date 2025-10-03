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

from unittest.mock import MagicMock, Mock, patch

import pytest

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.client_partitioning import ClientPartitioningStage, _read_list_json_rel
from nemo_curator.tasks import FileGroupTask, _EmptyTask


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

    def test_initialization(self):
        """Test initialization with default and custom values."""
        # Test default values
        stage = ClientPartitioningStage(file_paths="/test/path")
        assert stage.file_paths == "/test/path"
        assert stage.input_list_json_path is None
        assert stage._name == "client_partitioning"
        assert stage.files_per_partition is None
        assert stage.file_extensions == [".jsonl", ".json", ".parquet"]
        assert stage.storage_options == {}
        assert stage.limit is None

        # Test custom values
        stage = ClientPartitioningStage(
            file_paths="/custom/path",
            input_list_json_path="/path/to/list.json",
            files_per_partition=5,
            file_extensions=[".txt", ".json"],
            storage_options={"key": "value"},
            limit=3,
        )
        assert stage.file_paths == "/custom/path"
        assert stage.input_list_json_path == "/path/to/list.json"
        assert stage.files_per_partition == 5
        assert stage.file_extensions == [".txt", ".json"]
        assert stage.storage_options == {"key": "value"}
        assert stage.limit == 3

    @patch("nemo_curator.stages.client_partitioning.url_to_fs")
    def test_setup(self, mock_url_to_fs: Mock) -> None:
        """Test setup method with and without storage options."""
        mock_fs = Mock()
        mock_root = "/test/path"
        mock_url_to_fs.return_value = (mock_fs, mock_root)

        # Test basic setup
        stage = ClientPartitioningStage(file_paths="/test/path")
        stage.setup()
        mock_url_to_fs.assert_called_with("/test/path")
        assert stage._fs == mock_fs
        assert stage._root == mock_root

        # Test setup with storage options
        mock_url_to_fs.reset_mock()
        storage_options = {"key": "value"}
        stage = ClientPartitioningStage(file_paths="/test/path", storage_options=storage_options)
        stage.setup()
        mock_url_to_fs.assert_called_with("/test/path", **storage_options)

    @patch("nemo_curator.stages.client_partitioning.ClientPartitioningStage._list_relative")
    @patch("nemo_curator.stages.client_partitioning.url_to_fs")
    def test_process_basic_functionality(
        self, mock_url_to_fs: Mock, mock_list_relative: Mock, empty_task: _EmptyTask
    ) -> None:
        """Test basic process functionality including file filtering."""
        mock_fs = Mock()
        mock_root = "/test/path"
        mock_url_to_fs.return_value = (mock_fs, mock_root)

        # Test basic processing - each file gets its own task
        test_files = ["file1.jsonl", "file2.jsonl", "file3.jsonl"]
        mock_list_relative.return_value = test_files

        stage = ClientPartitioningStage(file_paths="/test/path")
        stage.setup()
        result = stage.process(empty_task)

        assert len(result) == 3
        assert isinstance(result[0], FileGroupTask)
        assert str(result[0].data[0]).endswith("file1.jsonl")
        assert result[0].task_id == "file_group_0"
        assert result[0]._metadata["partition_index"] == 0
        assert result[0]._metadata["total_partitions"] == 3

        # Test file extension filtering
        all_files = ["file1.jsonl", "file2.txt", "file3.json", "file4.py"]
        mock_list_relative.return_value = all_files
        stage = ClientPartitioningStage(file_paths="/test/path", file_extensions=[".jsonl", ".json"])
        stage.setup()
        result = stage.process(empty_task)

        assert len(result) == 2
        assert str(result[0].data[0]).endswith("file1.jsonl")
        assert str(result[1].data[0]).endswith("file3.json")

    @patch("nemo_curator.stages.client_partitioning.ClientPartitioningStage._list_relative")
    @patch("nemo_curator.stages.client_partitioning.url_to_fs")
    def test_process_partitioning_and_limits(
        self, mock_url_to_fs: Mock, mock_list_relative: Mock, empty_task: _EmptyTask
    ) -> None:
        """Test files_per_partition and limit functionality."""
        mock_fs = Mock()
        mock_root = "/test/path"
        mock_url_to_fs.return_value = (mock_fs, mock_root)

        # Test files_per_partition
        all_files = ["file1.jsonl", "file2.jsonl", "file3.jsonl", "file4.jsonl"]
        mock_list_relative.return_value = all_files

        stage = ClientPartitioningStage(file_paths="/test/path", files_per_partition=2)
        stage.setup()
        result = stage.process(empty_task)

        assert len(result) == 2
        assert len(result[0].data) == 2
        assert len(result[1].data) == 2
        assert result[0]._metadata["partition_index"] == 0
        assert result[0]._metadata["total_partitions"] == 2

        # Test limit functionality
        all_files = ["file1.jsonl", "file2.jsonl", "file3.jsonl", "file4.jsonl", "file5.jsonl"]
        mock_list_relative.return_value = all_files
        stage = ClientPartitioningStage(file_paths="/test/path", limit=3)
        stage.setup()
        result = stage.process(empty_task)

        assert len(result) == 3

    @patch("nemo_curator.stages.client_partitioning.ClientPartitioningStage._list_relative")
    @patch("nemo_curator.stages.client_partitioning.url_to_fs")
    def test_process_edge_cases(self, mock_url_to_fs: Mock, mock_list_relative: Mock, empty_task: _EmptyTask) -> None:
        """Test edge cases like empty file list and combined filters."""
        mock_fs = Mock()
        mock_root = "/test/path"
        mock_url_to_fs.return_value = (mock_fs, mock_root)

        # Test empty file list
        mock_list_relative.return_value = []
        stage = ClientPartitioningStage(file_paths="/test/path")
        stage.setup()
        result = stage.process(empty_task)
        assert len(result) == 0

        # Test combined filters (extensions + limit + partitioning)
        all_files = ["file1.jsonl", "file2.txt", "file3.json", "file4.py", "file5.jsonl", "file6.json"]
        mock_list_relative.return_value = all_files
        stage = ClientPartitioningStage(
            file_paths="/test/path", file_extensions=[".jsonl", ".json"], limit=3, files_per_partition=2
        )
        stage.setup()
        result = stage.process(empty_task)

        assert len(result) == 2
        assert len(result[0].data) == 2
        assert len(result[1].data) == 1

    def test_process_without_setup(self, empty_task: _EmptyTask) -> None:
        """Test process method when setup() hasn't been called."""
        stage = ClientPartitioningStage(file_paths=None)
        with pytest.raises(RuntimeError, match="Stage not initialized"):
            stage.process(empty_task)

    def test_inheritance_from_file_partitioning(self) -> None:
        """Test that ClientPartitioningStage inherits from FilePartitioningStage."""
        stage = ClientPartitioningStage(file_paths="/test/path")
        assert hasattr(stage, "_partition_by_count")
        assert hasattr(stage, "_parse_size")
        assert hasattr(stage, "_get_dataset_name")


class TestReadListJsonRel:
    """Test suite for _read_list_json_rel function."""

    @patch("nemo_curator.stages.client_partitioning.fsspec.open")
    def test_read_list_json_rel_success(self, mock_fsspec_open: Mock) -> None:
        """Test successful reading of list JSON file."""
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_fsspec_open.return_value = mock_file

        mock_file.read.return_value = b'["/input/path/video1.mp4", "/input/path/video2.mp4"]'

        result = _read_list_json_rel(
            root="/input/path", json_url="/path/to/list.json", storage_options={"profile_name": "test_profile"}
        )

        expected = ["video1.mp4", "video2.mp4"]
        assert result == expected
        mock_fsspec_open.assert_called_once_with("/path/to/list.json", "rb", profile_name="test_profile")

    @patch("nemo_curator.stages.client_partitioning.fsspec.open")
    def test_read_list_json_rel_error_cases(self, mock_fsspec_open: Mock) -> None:
        """Test error cases for reading list JSON."""
        # Test path mismatch
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_fsspec_open.return_value = mock_file

        mock_file.read.return_value = b'["/different/path/video1.mp4"]'

        with pytest.raises(ValueError, match="Input path .* is not under root"):
            _read_list_json_rel(root="/input/path", json_url="/path/to/list.json", storage_options={})

        # Test file read exception
        mock_fsspec_open.side_effect = Exception("File not found")
        with pytest.raises(Exception, match="File not found"):
            _read_list_json_rel(root="/input/path", json_url="/path/to/list.json", storage_options={})

    @patch("nemo_curator.stages.client_partitioning.fsspec.open")
    def test_read_list_json_rel_empty_list(self, mock_fsspec_open: Mock) -> None:
        """Test reading empty list JSON."""
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_fsspec_open.return_value = mock_file

        mock_file.read.return_value = b"[]"

        result = _read_list_json_rel(root="/input/path", json_url="/path/to/list.json", storage_options={})

        assert result == []
