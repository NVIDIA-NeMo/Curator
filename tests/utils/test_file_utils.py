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

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.utils.file_utils import (
    get_all_file_paths_and_size_under,
    get_all_file_paths_under,
    infer_dataset_name_from_path,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestInferDatasetNameFromPath:
    """Test cases for infer_dataset_name_from_path function."""

    def test_local_paths(self):
        """Test local path dataset name inference."""
        assert infer_dataset_name_from_path("/home/user/my_dataset/file.txt") == "my_dataset"
        assert infer_dataset_name_from_path("./file.txt") == "file"
        assert infer_dataset_name_from_path("file.txt") == "file"

    def test_cloud_paths(self):
        """Test cloud storage path dataset name inference."""
        assert infer_dataset_name_from_path("s3://bucket/datasets/my_dataset/") == "my_dataset"
        assert infer_dataset_name_from_path("s3://bucket/datasets/my_dataset/data.parquet") == "data.parquet"
        assert infer_dataset_name_from_path("s3://my-bucket") == "my-bucket"
        assert infer_dataset_name_from_path("abfs://container@account.dfs.core.windows.net/dataset") == "dataset"

    def test_case_conversion(self):
        """Test that results are converted to lowercase."""
        assert infer_dataset_name_from_path("s3://bucket/MyDataSet") == "mydataset"
        assert infer_dataset_name_from_path("/home/user/MyDataSet/file.txt") == "mydataset"


def _write_test_file(path: Path, content: str = "test", size_bytes: int | None = None) -> None:
    """Helper to create test files with specific content or size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if size_bytes is not None:
        path.write_bytes(b"x" * size_bytes)
    else:
        path.write_text(content)


class TestGetAllFilePathsUnder:
    """Test cases for get_all_file_paths_under function."""

    def test_recursion_with_filtering(self, tmp_path: Path):
        """Test recursive directory traversal with extension filtering."""
        # Create directory structure with mixed file types
        root = tmp_path / "data"
        files_to_create = [
            (root / "file1.jsonl", "{}"),
            (root / "file2.json", "{}"),
            (root / "file3.parquet", "{}"),
            (root / "file4.txt", "text"),
            (root / "subdir" / "file5.jsonl", "{}"),
            (root / "subdir" / "nested" / "file6.json", "{}"),
            (root / "subdir" / "file7.txt", "text"),
        ]

        for file_path, content in files_to_create:
            _write_test_file(file_path, content)

        # Test with recursion and filtering
        result = get_all_file_paths_under(
            str(root),
            recurse_subdirectories=True,
            keep_extensions=[".jsonl", ".json"],
        )

        expected_files = [
            str(root / "file1.jsonl"),
            str(root / "file2.json"),
            str(root / "subdir" / "file5.jsonl"),
            str(root / "subdir" / "nested" / "file6.json"),
        ]
        assert sorted(result) == sorted(expected_files)

    def test_no_recursion(self, tmp_path: Path):
        """Test directory traversal without recursion."""
        root = tmp_path / "data"
        files_to_create = [
            (root / "file1.jsonl", "{}"),
            (root / "file2.txt", "text"),
            (root / "subdir" / "file3.jsonl", "{}"),
        ]

        for file_path, content in files_to_create:
            _write_test_file(file_path, content)

        result = get_all_file_paths_under(
            str(root),
            recurse_subdirectories=False,
            keep_extensions=[".jsonl"],
        )

        # Should only include top-level jsonl files
        assert result == [str(root / "file1.jsonl")]

    def test_single_file_input(self, tmp_path: Path):
        """Test with a single file as input."""
        test_file = tmp_path / "single.jsonl"
        _write_test_file(test_file, "{}")

        result = get_all_file_paths_under(
            str(test_file),
            recurse_subdirectories=False,
            keep_extensions=[".jsonl"],
        )

        assert result == [str(test_file)]

    def test_no_extension_filtering(self, tmp_path: Path):
        """Test without extension filtering."""
        root = tmp_path / "data"
        files_to_create = [
            (root / "file1.jsonl", "{}"),
            (root / "file2.txt", "text"),
            (root / "file_no_ext", "content"),
        ]

        for file_path, content in files_to_create:
            _write_test_file(file_path, content)

        result = get_all_file_paths_under(
            str(root),
            recurse_subdirectories=False,
            keep_extensions=None,
        )

        expected_files = [str(f[0]) for f in files_to_create]
        assert sorted(result) == sorted(expected_files)

    def test_empty_directory(self, tmp_path: Path):
        """Test with empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = get_all_file_paths_under(str(empty_dir))
        assert result == []


class TestGetAllFilePathsAndSizeUnder:
    """Test cases for get_all_file_paths_and_size_under function."""

    def test_files_with_sizes_sorted(self, tmp_path: Path):
        """Test that files are returned with sizes and sorted by size."""
        root = tmp_path / "sized_files"

        # Create files with different sizes
        files_with_sizes = [
            (root / "large.jsonl", 100),
            (root / "small.jsonl", 10),
            (root / "medium.jsonl", 50),
        ]

        for file_path, size in files_with_sizes:
            _write_test_file(file_path, size_bytes=size)

        result = get_all_file_paths_and_size_under(
            str(root),
            recurse_subdirectories=False,
            keep_extensions=[".jsonl"],
        )

        # Should be sorted by size (ascending)
        expected = [
            (str(root / "small.jsonl"), 10),
            (str(root / "medium.jsonl"), 50),
            (str(root / "large.jsonl"), 100),
        ]
        assert result == expected

    def test_with_recursion_and_filtering(self, tmp_path: Path):
        """Test recursive traversal with extension filtering and size info."""
        root = tmp_path / "nested_sized"

        files_to_create = [
            (root / "file1.jsonl", 20),
            (root / "file2.txt", 30),  # Should be filtered out
            (root / "subdir" / "file3.jsonl", 15),
            (root / "subdir" / "file4.parquet", 25),
        ]

        for file_path, size in files_to_create:
            _write_test_file(file_path, size_bytes=size)

        result = get_all_file_paths_and_size_under(
            str(root),
            recurse_subdirectories=True,
            keep_extensions=[".jsonl", ".parquet"],
        )

        expected = [
            (str(root / "subdir" / "file3.jsonl"), 15),
            (str(root / "file1.jsonl"), 20),
            (str(root / "subdir" / "file4.parquet"), 25),
        ]
        assert result == expected

    def test_single_file_with_size(self, tmp_path: Path):
        """Test single file input returns correct size."""
        test_file = tmp_path / "single.jsonl"
        _write_test_file(test_file, size_bytes=42)

        result = get_all_file_paths_and_size_under(str(test_file))
        assert result == [(str(test_file), 42)]


class TestFilePartitioningStageGetters:
    """Test cases for FilePartitioningStage private methods."""

    def test_get_file_list_directory_input(self, tmp_path: Path):
        """Test _get_file_list with directory input."""
        root = tmp_path / "stage_test"

        # Create files with default extensions
        files_to_create = [
            (root / "file1.jsonl", "{}"),
            (root / "file2.json", "{}"),
            (root / "file3.parquet", "{}"),
            (root / "file4.txt", "text"),  # Should be filtered out
            (root / "nested" / "file5.jsonl", "{}"),
        ]

        for file_path, content in files_to_create:
            _write_test_file(file_path, content)

        stage = FilePartitioningStage(file_paths=str(root))
        result = stage._get_file_list()

        # Should include jsonl, json, parquet files (default extensions)
        expected_files = [
            str(root / "file1.jsonl"),
            str(root / "file2.json"),
            str(root / "file3.parquet"),
            str(root / "nested" / "file5.jsonl"),
        ]
        assert sorted(result) == sorted(expected_files)

    def test_get_file_list_custom_extensions(self, tmp_path: Path):
        """Test _get_file_list with custom file extensions."""
        root = tmp_path / "custom_ext"

        files_to_create = [
            (root / "file1.txt", "text"),
            (root / "file2.jsonl", "NA"),  # Should be filtered out
            (root / "file3.log", "log"),
        ]

        for file_path, content in files_to_create:
            _write_test_file(file_path, content)

        stage = FilePartitioningStage(file_paths=str(root), file_extensions=[".txt", ".log"])
        result = stage._get_file_list()

        expected_files = [
            str(root / "file1.txt"),
            str(root / "file3.log"),
        ]
        assert sorted(result) == sorted(expected_files)

    def test_get_file_list_with_sizes_directory_sorted(self, tmp_path: Path):
        """Test _get_file_list_with_sizes with directory input."""
        root = tmp_path / "sized_stage_test"

        files_with_sizes = [
            (root / "large.jsonl", 80),
            (root / "small.json", 20),
            (root / "medium.parquet", 50),
            (root / "nested" / "tiny.jsonl", 5),
        ]

        for file_path, size in files_with_sizes:
            _write_test_file(file_path, size_bytes=size)

        stage = FilePartitioningStage(file_paths=str(root))
        result = stage._get_file_list_with_sizes()

        # Should be sorted by size (ascending)
        expected = [
            (str(root / "nested" / "tiny.jsonl"), 5),
            (str(root / "small.json"), 20),
            (str(root / "medium.parquet"), 50),
            (str(root / "large.jsonl"), 80),
        ]
        assert result == expected

    def test_get_file_list_with_sizes_list_input(self, tmp_path: Path):
        """Test _get_file_list_with_sizes with list input preserves order."""
        files_with_sizes = [
            (tmp_path / "file1.jsonl", 30),
            (tmp_path / "file2.jsonl", 10),
            (tmp_path / "file3.jsonl", 20),
        ]

        for file_path, size in files_with_sizes:
            _write_test_file(file_path, size_bytes=size)

        # Test with specific order
        file_paths = [str(files_with_sizes[1][0]), str(files_with_sizes[2][0]), str(files_with_sizes[0][0])]
        stage = FilePartitioningStage(file_paths=file_paths)
        result = stage._get_file_list_with_sizes()

        # Should preserve input order for list input
        expected = [
            (str(files_with_sizes[1][0]), 10),
            (str(files_with_sizes[2][0]), 20),
            (str(files_with_sizes[0][0]), 30),
        ]
        assert result == expected

    def test_get_file_list_invalid_input(self):
        """Test that invalid input types raise TypeError."""
        stage = FilePartitioningStage(file_paths=123)  # Invalid type

        with pytest.raises(TypeError, match="Invalid file paths"):
            stage._get_file_list()

        with pytest.raises(TypeError, match="Invalid file paths"):
            stage._get_file_list_with_sizes()
