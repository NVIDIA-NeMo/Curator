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

import os
from pathlib import Path

import pytest

from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import FileGroupTask, _EmptyTask


def _create_test_jsonl_files(base_dir: Path | str, num_files: int, subdir: str | None = None) -> list[str]:
    """Create num_files minimal JSONL files in base_dir[/subdir] and return their paths."""
    base = Path(base_dir)
    target_dir = base / subdir if subdir else base
    os.makedirs(target_dir, exist_ok=True)
    files: list[str] = []
    for i in range(num_files):
        file_path = target_dir / f"file{i}.jsonl"
        file_path.write_text("{}\n")
        files.append(str(file_path))
    return files


class TestFilePartitioningStage:
    """Test suite for FilePartitioningStage."""

    @pytest.fixture
    def temp_files(self, tmp_path: Path) -> list[str]:
        """Create temporary test files for testing."""
        files = []
        for i in range(10):
            file_path = tmp_path / f"test_file_{i}.jsonl"
            file_path.write_text(f'{{"id": {i}, "text": "Test content {i}"}}')
            files.append(str(file_path))
        return files

    @pytest.fixture
    def empty_task(self) -> _EmptyTask:
        """Create an empty task for testing."""
        return _EmptyTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=None,
            _metadata={"source": "test"},
        )

    def test_initialization_default_values(self):
        """Test initialization with default parameter values."""
        stage = FilePartitioningStage(file_paths="/test/path")

        assert stage.file_paths == "/test/path"
        assert stage.files_per_partition is None
        assert stage.blocksize is None
        assert stage.file_extensions == [".jsonl", ".json", ".parquet"]
        assert stage.storage_options == {}
        assert stage.limit is None
        assert stage._name == "file_partitioning"

    def test_initialization_custom_values(self):
        """Test initialization with custom parameter values."""
        stage = FilePartitioningStage(
            file_paths="/custom/path",
            files_per_partition=5,
            blocksize="128MB",
            file_extensions=[".txt", ".json"],
            storage_options={"key": "value"},
            limit=3,
        )

        assert stage.file_paths == "/custom/path"
        assert stage.files_per_partition == 5
        assert stage.blocksize == "128MB"
        assert stage.file_extensions == [".txt", ".json"]
        assert stage.storage_options == {"key": "value"}
        assert stage.limit == 3

    def test_inputs_outputs(self):
        """Test inputs and outputs methods."""
        stage = FilePartitioningStage(file_paths="/test/path")

        assert stage.inputs() == ([], [])
        assert stage.outputs() == ([], [])

    def test_resources(self):
        """Test resource requirements."""
        stage = FilePartitioningStage(file_paths="/test/path")

        assert stage.resources.cpus == 0.5

    def test_ray_stage_spec(self):
        """Test ray stage specification."""
        stage = FilePartitioningStage(file_paths="/test/path")

        spec = stage.ray_stage_spec()
        assert spec["is_fanout_stage"] is True

    def test_process_with_file_list(self, empty_task: _EmptyTask, tmp_path: Path):
        """Test processing with a list of files."""
        # Create these files in the tmp_path:
        test_files = _create_test_jsonl_files(tmp_path, num_files=3, subdir="path")
        stage = FilePartitioningStage(file_paths=test_files)

        result = stage.process(empty_task)

        assert len(result) == len(test_files)  # 3 files in 3 groups
        assert isinstance(result[0], FileGroupTask)
        assert result[0].data == [test_files[0]]
        assert result[0].dataset_name == "path"
        assert result[0].task_id == "file_group_0"

    def test_process_with_files_per_partition(self, empty_task: _EmptyTask, tmp_path: Path):
        """Test processing with files_per_partition setting."""
        test_files = _create_test_jsonl_files(tmp_path, num_files=4, subdir="path")
        stage = FilePartitioningStage(file_paths=test_files, files_per_partition=2)

        result = stage.process(empty_task)

        assert len(result) == 2  # 4 files / 2 per partition
        assert result[0].data == test_files[:2]
        assert result[1].data == test_files[2:]

    def test_process_with_limit(self, empty_task: _EmptyTask, tmp_path: Path):
        """Test processing with limit parameter - this is the main test for the limit functionality."""
        test_files = _create_test_jsonl_files(tmp_path, num_files=10, subdir="path")
        stage = FilePartitioningStage(
            file_paths=test_files,
            files_per_partition=2,  # This would normally create 5 groups
            limit=3,  # But limit to only 3 groups
        )

        result = stage.process(empty_task)

        # Should only return 3 file groups due to limit
        assert len(result) == 3
        assert result[0].data == test_files[:2]
        assert result[1].data == test_files[2:4]
        assert result[2].data == test_files[4:6]

        # Verify metadata
        for i, task in enumerate(result):
            assert task.task_id == f"file_group_{i}"
            assert task._metadata["partition_index"] == i
            assert task._metadata["total_partitions"] == 5  # Total partitions before limit

    def test_process_with_limit_single_partition(self, empty_task: _EmptyTask, tmp_path: Path):
        """Test limit when all files would be in a single partition."""
        test_files = _create_test_jsonl_files(tmp_path, num_files=5, subdir="path")
        stage = FilePartitioningStage(
            file_paths=test_files,
            limit=1,  # Limit to 1 group, TODO: Ask ayush why this is the behavior
        )
        result = stage.process(empty_task)

        assert len(result) == 1
        assert result[0].data == [test_files[0]]

    def test_process_with_limit_zero(self, empty_task: _EmptyTask, tmp_path: Path):
        """Test processing with limit set to 0."""
        test_files = _create_test_jsonl_files(tmp_path, num_files=5, subdir="path")
        stage = FilePartitioningStage(
            file_paths=test_files,
            files_per_partition=1,
            limit=0,  # No groups should be created
        )

        result = stage.process(empty_task)

        assert len(result) == 0

    def test_process_with_blocksize(self, empty_task: _EmptyTask, tmp_path: Path):
        """Test processing with blocksize setting."""
        test_files = _create_test_jsonl_files(tmp_path, num_files=6)
        stage = FilePartitioningStage(file_paths=test_files, blocksize="1B")

        result = stage.process(empty_task)

        # With default avg_file_size of 100MB and blocksize of ~52MB,
        # files_per_block should be max(1, 52MB // 100MB) = 1
        assert len(result) == 6
        for i, task in enumerate(result):
            assert len(task.data) == 1
            assert task.data[0] == test_files[i]

    def test_process_empty_file_list(self, empty_task: _EmptyTask):
        """Test processing with empty file list."""
        stage = FilePartitioningStage(file_paths=[])

        result = stage.process(empty_task)

        assert len(result) == 0

    def test_get_dataset_name(self, tmp_path: Path):
        """Test dataset name extraction."""
        stage = FilePartitioningStage(file_paths=[])

        # Test with files
        files = _create_test_jsonl_files(tmp_path, num_files=2, subdir="parent/dir")
        dataset_name = stage._get_dataset_name(files)
        assert dataset_name == "dir"

        # Test with empty files
        dataset_name = stage._get_dataset_name([])
        assert dataset_name == "dataset"

    def test_partition_by_count(self):
        """Test _partition_by_count method."""
        stage = FilePartitioningStage(file_paths=[])
        files = ["file1", "file2", "file3", "file4", "file5"]

        partitions = stage._partition_by_count(files, 2)

        assert len(partitions) == 3
        assert partitions[0] == ["file1", "file2"]
        assert partitions[1] == ["file3", "file4"]
        assert partitions[2] == ["file5"]

    def test_parse_size(self):
        """Test _parse_size method."""
        stage = FilePartitioningStage(file_paths=[])

        assert stage._parse_size("100B") == 100
        assert stage._parse_size("1KB") == 1000
        assert stage._parse_size("1KiB") == 1024
        assert stage._parse_size("1MB") == 1000 * 1000
        assert stage._parse_size("1MiB") == 1024 * 1024
        assert stage._parse_size("1GB") == 1000 * 1000 * 1000
        assert stage._parse_size("1GiB") == 1024 * 1024 * 1024
        assert stage._parse_size("2TB") == 2 * 1000 * 1000 * 1000 * 1000
        assert stage._parse_size("2TiB") == 2 * 1024 * 1024 * 1024 * 1024
        assert stage._parse_size("100") == 100  # No unit defaults to bytes

    def test_task_metadata(self, empty_task: _EmptyTask, tmp_path: Path):
        """Test that created tasks have proper metadata."""
        test_files = _create_test_jsonl_files(tmp_path, num_files=2, subdir="path")
        storage_options = {"option1": "value1"}
        stage = FilePartitioningStage(file_paths=test_files, storage_options=storage_options)

        result = stage.process(empty_task)

        assert len(result) == 2
        task = result[0]

        assert task._metadata["partition_index"] == 0
        assert task._metadata["total_partitions"] == 2
        assert task._metadata["source_files"] == [test_files[0]]
        assert task.reader_config == {}
