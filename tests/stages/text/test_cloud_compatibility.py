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

"""
Test cloud compatibility for text components to ensure fsspec-based operations
instead of OS-based operations.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import fsspec
import pytest

from nemo_curator.utils.client_utils import fs_join


class TestFSJoin:
    """Test suite for the fs_join utility function."""

    def test_fs_join_local_filesystem(self):
        """Test fs_join with local filesystem."""
        fs = fsspec.filesystem("file")
        result = fs_join(fs, "/tmp", "subdir", "file.txt")
        expected = "file:///tmp/subdir/file.txt"
        assert result == expected

    def test_fs_join_s3_filesystem(self):
        """Test fs_join with S3 filesystem."""
        # Mock S3 filesystem
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.sep = "/"
        mock_fs._strip_protocol.return_value = "bucket/path"
        mock_fs.unstrip_protocol.return_value = "s3://bucket/path/subdir/file.txt"

        result = fs_join(mock_fs, "s3://bucket/path", "subdir", "file.txt")
        
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

        result = fs_join(mock_fs, "gs://bucket/path", "data", "file.json")
        
        assert result == "gs://bucket/path/data/file.json"

    def test_fs_join_with_trailing_separators(self):
        """Test fs_join with trailing separators in path components."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.sep = "/"
        mock_fs._strip_protocol.return_value = "bucket/path"
        mock_fs.unstrip_protocol.return_value = "s3://bucket/path/subdir/file.txt"

        result = fs_join(mock_fs, "s3://bucket/path/", "/subdir/", "/file.txt")
        
        # Should strip separators from parts
        mock_fs.unstrip_protocol.assert_called_once_with("bucket/path/subdir/file.txt")
        assert result == "s3://bucket/path/subdir/file.txt"

    def test_fs_join_empty_parts(self):
        """Test fs_join with empty parts."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.sep = "/"
        mock_fs._strip_protocol.return_value = "bucket/path"
        mock_fs.unstrip_protocol.return_value = "s3://bucket/path"

        result = fs_join(mock_fs, "s3://bucket/path")
        
        assert result == "s3://bucket/path"


class TestCloudCompatibility:
    """Test cloud compatibility patterns across text components."""
    
    def test_download_path_construction_patterns(self):
        """Test that download components use cloud-compatible path construction."""
        # This test verifies that we avoid os.path.join in favor of fsspec patterns
        
        # Mock different filesystem types
        mock_s3_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_s3_fs.sep = "/"
        mock_s3_fs._strip_protocol.side_effect = lambda x: x.replace("s3://", "")
        mock_s3_fs.unstrip_protocol.side_effect = lambda x: f"s3://{x}"
        
        # Test S3 path construction
        base_path = "s3://commoncrawl"
        subpath = "crawl-data/file.txt"
        result = fs_join(mock_s3_fs, base_path, subpath)
        
        expected = "s3://commoncrawl/crawl-data/file.txt"
        assert result == expected

    def test_cache_directory_path_construction(self):
        """Test that cache directories use cloud-compatible paths."""
        # Test with temporary directory to simulate cache behavior
        with tempfile.TemporaryDirectory() as tmp_dir:
            fs = fsspec.filesystem("file")
            
            # Test constructing histogram cache path
            cache_base = tmp_dir
            cache_subdir = "histograms"
            
            result = fs_join(fs, cache_base, cache_subdir)
            # For local filesystem, fsspec adds file:// protocol
            expected_path = f"file://{os.path.join(tmp_dir, 'histograms')}"
            
            assert result == expected_path
            
            # Verify the path is valid for the filesystem
            # Strip protocol for existence check
            check_path = fs._strip_protocol(result)
            assert not fs.exists(check_path)  # Should not exist yet
            
    def test_output_filename_construction(self):
        """Test that output filenames are constructed in a cloud-compatible way."""
        # Mock filesystem for testing
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.sep = "/"
        mock_fs._strip_protocol.side_effect = lambda x: x.replace("s3://", "")
        mock_fs.unstrip_protocol.side_effect = lambda x: f"s3://{x}"
        
        # Test output file construction patterns
        download_dir = "s3://my-bucket/downloads"
        output_name = "document.txt"
        
        result = fs_join(mock_fs, download_dir, output_name)
        expected = "s3://my-bucket/downloads/document.txt"
        
        assert result == expected

    def test_arxiv_s3_path_construction(self):
        """Test ArXiv-specific S3 path construction patterns."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.sep = "/"
        mock_fs._strip_protocol.side_effect = lambda x: x.replace("s3://", "")
        mock_fs.unstrip_protocol.side_effect = lambda x: f"s3://{x}"
        
        # Test the pattern used in ArXiv downloader
        base = "s3://arxiv/src"
        url = "1901/1901.00001.tar"
        
        result = fs_join(mock_fs, base, url)
        expected = "s3://arxiv/src/1901/1901.00001.tar"
        
        assert result == expected

    def test_common_crawl_s3_path_construction(self):
        """Test Common Crawl S3 path construction patterns."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.sep = "/"
        mock_fs._strip_protocol.side_effect = lambda x: x.replace("s3://", "")
        mock_fs.unstrip_protocol.side_effect = lambda x: f"s3://{x}"
        
        # Test the pattern used in Common Crawl downloader
        base = "s3://commoncrawl"
        urlpath = "crawl-data/CC-MAIN-2023-14/segments/file.warc.gz"
        
        result = fs_join(mock_fs, base, urlpath)
        expected = "s3://commoncrawl/crawl-data/CC-MAIN-2023-14/segments/file.warc.gz"
        
        assert result == expected

    def test_mixed_filesystem_compatibility(self):
        """Test that fs_join works with different filesystem types."""
        
        # Test with actual local filesystem
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_fs = fsspec.filesystem("file")
            result = fs_join(local_fs, tmp_dir, "subdir", "file.txt")
            
            # For local filesystem, fsspec adds file:// protocol
            expected = f"file://{os.path.join(tmp_dir, 'subdir', 'file.txt')}"
            assert result == expected
            
        # Test with mock remote filesystems
        for protocol in ["s3", "gs", "azure"]:
            mock_fs = Mock(spec=fsspec.AbstractFileSystem)
            mock_fs.sep = "/"
            mock_fs._strip_protocol.side_effect = lambda x: x.split("://", 1)[1] if "://" in x else x
            mock_fs.unstrip_protocol.side_effect = lambda x: f"{protocol}://{x}"
            
            result = fs_join(mock_fs, f"{protocol}://bucket/path", "subdir", "file.txt")
            expected = f"{protocol}://bucket/path/subdir/file.txt"
            assert result == expected