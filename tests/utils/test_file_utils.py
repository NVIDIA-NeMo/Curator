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

import os
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import Mock

import fsspec
import pytest

from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.utils.file_utils import (
    fs_join,
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


class TestFSJoin:
    """Test suite for the fs_join utility function."""

    def test_fs_join_local_filesystem(self):
        """Test fs_join with local filesystem."""
        fs = fsspec.filesystem("file")
        result = fs_join("/tmp", "subdir", "file.txt", fs=fs)
        expected = "file:///tmp/subdir/file.txt"
        assert result == expected

    def test_fs_join_local_filesystem_auto_fs(self):
        """Test fs_join with auto-detected local filesystem."""
        result = fs_join("/tmp", "subdir", "file.txt")
        expected = "file:///tmp/subdir/file.txt"
        assert result == expected

    def test_fs_join_s3_filesystem(self):
        """Test fs_join with S3 filesystem."""
        # Mock S3 filesystem
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.sep = "/"
        mock_fs._strip_protocol.return_value = "bucket/path"
        mock_fs.unstrip_protocol.return_value = "s3://bucket/path/subdir/file.txt"

        result = fs_join("s3://bucket/path", "subdir", "file.txt", fs=mock_fs)
        
        mock_fs._strip_protocol.assert_called_once_with("s3://bucket/path")
        mock_fs.unstrip_protocol.assert_called_once_with("bucket/path/subdir/file.txt")
        assert result == "s3://bucket/path/subdir/file.txt"

    def test_fs_join_gcs_filesystem(self):
        """Test fs_join with GCS filesystem."""
        # Mock GCS filesystem
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.sep = "/"
        mock_fs._strip_protocol.return_value = "bucket/path"
        mock_fs.unstrip_protocol.return_value = "gs://bucket/path/data/file.json"

        result = fs_join("gs://bucket/path", "data", "file.json", fs=mock_fs)
        
        assert result == "gs://bucket/path/data/file.json"

    def test_fs_join_with_trailing_separators(self):
        """Test fs_join with trailing separators in path components."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.sep = "/"
        mock_fs._strip_protocol.return_value = "bucket/path"
        mock_fs.unstrip_protocol.return_value = "s3://bucket/path/subdir/file.txt"

        result = fs_join("s3://bucket/path/", "/subdir/", "/file.txt", fs=mock_fs)
        
        # Should strip separators from parts
        mock_fs.unstrip_protocol.assert_called_once_with("bucket/path/subdir/file.txt")
        assert result == "s3://bucket/path/subdir/file.txt"

    def test_fs_join_empty_parts(self):
        """Test fs_join with empty parts."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.sep = "/"
        mock_fs._strip_protocol.return_value = "bucket/path"
        mock_fs.unstrip_protocol.return_value = "s3://bucket/path"

        result = fs_join("s3://bucket/path", fs=mock_fs)
        
        assert result == "s3://bucket/path"

    def test_fs_join_download_path_construction_patterns(self):
        """Test that download components use cloud-compatible path construction."""
        # This test verifies that fs_join correctly handles cloud filesystem paths
        
        # Mock different filesystem types
        mock_s3_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_s3_fs.sep = "/"
        mock_s3_fs._strip_protocol.side_effect = lambda x: x.replace("s3://", "")
        mock_s3_fs.unstrip_protocol.side_effect = lambda x: f"s3://{x}"
        
        # Test S3 path construction
        base_path = "s3://commoncrawl"
        subpath = "crawl-data/file.txt"
        result = fs_join(base_path, subpath, fs=mock_s3_fs)
        
        expected = "s3://commoncrawl/crawl-data/file.txt"
        assert result == expected

    def test_fs_join_cache_directory_path_construction(self):
        """Test that cache directories use cloud-compatible paths."""
        # Test with temporary directory to simulate cache behavior
        with tempfile.TemporaryDirectory() as tmp_dir:
            fs = fsspec.filesystem("file")
            
            # Test constructing histogram cache path
            cache_base = tmp_dir
            cache_subdir = "histograms"
            
            result = fs_join(cache_base, cache_subdir, fs=fs)
            # For local filesystem, fsspec adds file:// protocol
            expected_path = f"file://{os.path.join(tmp_dir, 'histograms')}"
            
            assert result == expected_path
            
            # Verify the path is valid for the filesystem
            # Strip protocol for existence check
            check_path = fs._strip_protocol(result)
            assert not fs.exists(check_path)  # Should not exist yet
            
    def test_fs_join_output_filename_construction(self):
        """Test that output filenames are constructed in a cloud-compatible way."""
        # Mock filesystem for testing
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.sep = "/"
        mock_fs._strip_protocol.side_effect = lambda x: x.replace("s3://", "")
        mock_fs.unstrip_protocol.side_effect = lambda x: f"s3://{x}"
        
        # Test output file construction patterns
        download_dir = "s3://my-bucket/downloads"
        output_name = "document.txt"
        
        result = fs_join(download_dir, output_name, fs=mock_fs)
        expected = "s3://my-bucket/downloads/document.txt"
        
        assert result == expected

    def test_fs_join_arxiv_s3_path_construction(self):
        """Test ArXiv-specific S3 path construction patterns."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.sep = "/"
        mock_fs._strip_protocol.side_effect = lambda x: x.replace("s3://", "")
        mock_fs.unstrip_protocol.side_effect = lambda x: f"s3://{x}"
        
        # Test the pattern used in ArXiv downloader
        base = "s3://arxiv/src"
        url = "1901/1901.00001.tar"
        
        result = fs_join(base, url, fs=mock_fs)
        expected = "s3://arxiv/src/1901/1901.00001.tar"
        
        assert result == expected

    def test_fs_join_common_crawl_s3_path_construction(self):
        """Test Common Crawl S3 path construction patterns."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.sep = "/"
        mock_fs._strip_protocol.side_effect = lambda x: x.replace("s3://", "")
        mock_fs.unstrip_protocol.side_effect = lambda x: f"s3://{x}"
        
        # Test the pattern used in Common Crawl downloader
        base = "s3://commoncrawl"
        urlpath = "crawl-data/CC-MAIN-2023-14/segments/file.warc.gz"
        
        result = fs_join(base, urlpath, fs=mock_fs)
        expected = "s3://commoncrawl/crawl-data/CC-MAIN-2023-14/segments/file.warc.gz"
        
        assert result == expected

    def test_fs_join_mixed_filesystem_compatibility(self):
        """Test that fs_join works with different filesystem types."""
        
        # Test with actual local filesystem
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_fs = fsspec.filesystem("file")
            result = fs_join(tmp_dir, "subdir", "file.txt", fs=local_fs)
            
            # For local filesystem, fsspec adds file:// protocol
            expected = f"file://{os.path.join(tmp_dir, 'subdir', 'file.txt')}"
            assert result == expected
            
        # Test with mock remote filesystems
        for protocol in ["s3", "gs", "azure"]:
            mock_fs = Mock(spec=fsspec.AbstractFileSystem)
            mock_fs.sep = "/"
            mock_fs._strip_protocol.side_effect = lambda x: x.split("://", 1)[1] if "://" in x else x
            mock_fs.unstrip_protocol.side_effect = lambda x: f"{protocol}://{x}"
            
            result = fs_join(f"{protocol}://bucket/path", "subdir", "file.txt", fs=mock_fs)
            expected = f"{protocol}://bucket/path/subdir/file.txt"
            assert result == expected
