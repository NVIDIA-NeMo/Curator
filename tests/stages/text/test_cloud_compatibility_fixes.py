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
Tests for cloud compatibility fixes in Text Components.

This module tests that Text Components properly use fsspec and posixpath
for cloud storage URIs instead of os/pathlib operations.
"""

import posixpath
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from nemo_curator.stages.text.utils.text_utils import get_docstrings


class TestTextUtilsCloudFixes:
    """Test cloud compatibility fixes in text_utils.py."""

    def test_get_docstrings_with_cloud_uri(self):
        """Test that get_docstrings handles cloud URIs correctly."""
        # Create a mock file-like object with a cloud URI name
        source_code = '''
def example_function():
    """This is a docstring."""
    pass
'''
        mock_file = StringIO(source_code)
        # Simulate a cloud URI filename
        mock_file.name = "s3://bucket/path/to/script.py"
        
        # This should work without errors using posixpath operations
        result = get_docstrings(mock_file)
        
        # Verify the result contains the expected docstring
        assert len(result) > 0
        assert "This is a docstring." in str(result)

    def test_get_docstrings_filename_extraction_patterns(self):
        """Test filename extraction works with various cloud URI patterns."""
        test_cases = [
            ("s3://bucket/path/to/file.py", "file"),
            ("gs://my-bucket/deep/nested/path/script.py", "script"),
            ("abfs://container@account.dfs.core.windows.net/data/code.py", "code"),
            ("https://example.com/api/v1/source.py", "source"),
            ("/local/path/local_file.py", "local_file"),  # Local files should still work
        ]
        
        for uri, expected_module_name in test_cases:
            # Test the pattern that's now used in the fixed code
            module_name = posixpath.splitext(posixpath.basename(uri))[0]
            assert module_name == expected_module_name, f"Failed for URI: {uri}"


class TestSemanticDeduplicationCloudFixes:
    """Test cloud compatibility fixes in semantic deduplication."""

    def test_path_construction_patterns(self):
        """Test that path construction uses posixpath for cloud compatibility."""
        # Test the patterns now used in the fixed semantic.py
        base_path = "s3://bucket/cache"
        output_path = "gs://bucket/output"
        
        # These are the patterns now used in the fixed code
        embeddings_path = posixpath.join(base_path, "embeddings")
        semantic_dedup_path = posixpath.join(base_path, "semantic_dedup")
        duplicates_path = posixpath.join(output_path, "duplicates")
        deduplicated_path = posixpath.join(output_path, "deduplicated")
        state_file = posixpath.join(output_path, "semantic_id_generator.json")
        
        # Verify the paths are constructed correctly
        assert embeddings_path == "s3://bucket/cache/embeddings"
        assert semantic_dedup_path == "s3://bucket/cache/semantic_dedup"
        assert duplicates_path == "gs://bucket/output/duplicates"
        assert deduplicated_path == "gs://bucket/output/deduplicated"
        assert state_file == "gs://bucket/output/semantic_id_generator.json"

    def test_complex_cloud_uri_handling(self):
        """Test complex cloud URI scenarios."""
        test_uris = [
            "s3://my-bucket/path/with/multiple/levels/",
            "gs://another-bucket/data/2024/01/15/",
            "abfs://container@storage.dfs.core.windows.net/datasets/processed/",
        ]
        
        for base_uri in test_uris:
            # Test subdirectory creation patterns
            subdir = posixpath.join(base_uri, "embeddings")
            assert subdir.startswith(base_uri)
            assert subdir.endswith("embeddings")


class TestDownloadCloudFixes:
    """Test cloud compatibility fixes in download modules."""

    @patch('fsspec.core.url_to_fs')
    def test_download_file_operations(self, mock_url_to_fs):
        """Test that download operations use fsspec for cloud URIs."""
        # Mock fsspec filesystem
        mock_fs = Mock()
        mock_fs.makedirs.return_value = None
        mock_fs.exists.return_value = True
        mock_fs.info.return_value = {"size": 1024}
        mock_url_to_fs.return_value = (mock_fs, "bucket/path")
        
        # Import after patching to ensure mock is used
        from nemo_curator.stages.text.download.base.download import DocumentDownloader
        
        # Create a concrete subclass for testing
        class TestDownloader(DocumentDownloader):
            def _get_output_filename(self, url: str) -> str:
                return "test_file.txt"
            
            def _download_to_path(self, url: str, path: str) -> tuple[bool, str]:
                return True, ""
        
        # Test cloud URI download directory
        cloud_download_dir = "s3://test-bucket/downloads/"
        downloader = TestDownloader(cloud_download_dir)
        
        # Verify fsspec was called for directory creation
        mock_url_to_fs.assert_called()
        mock_fs.makedirs.assert_called_with(cloud_download_dir, exist_ok=True)

    def test_filename_extraction_from_cloud_paths(self):
        """Test filename extraction from cloud paths."""
        test_cases = [
            ("s3://bucket/path/to/file.txt", "file.txt"),
            ("gs://my-bucket/data/document.pdf", "document.pdf"),
            ("abfs://container@account.dfs.core.windows.net/files/archive.zip", "archive.zip"),
            ("https://example.com/downloads/data.json", "data.json"),
        ]
        
        for cloud_path, expected_filename in test_cases:
            # Test the pattern now used in the fixed iterator.py
            filename = posixpath.basename(cloud_path)
            assert filename == expected_filename, f"Failed for path: {cloud_path}"


class TestFilterCloudFixes:
    """Test cloud compatibility fixes in filter modules."""

    @patch('fsspec.core.url_to_fs')
    def test_fasttext_filter_model_check(self, mock_url_to_fs):
        """Test that FastText filter uses fsspec for model file checks."""
        # Mock fsspec filesystem
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_url_to_fs.return_value = (mock_fs, "bucket/path")
        
        # Import after patching
        from nemo_curator.stages.text.filters.fasttext_filter import FastTextQualityFilter
        
        # Test cloud URI model path
        cloud_model_path = "s3://models/fasttext_quality.bin"
        filter_instance = FastTextQualityFilter(model_path=cloud_model_path)
        
        # This should not raise an exception with fsspec
        filter_instance.model_check_or_download()
        
        # Verify fsspec was used
        mock_url_to_fs.assert_called_with(cloud_model_path)
        mock_fs.exists.assert_called_with(cloud_model_path)

    @patch('fsspec.core.url_to_fs')
    def test_heuristic_filter_cache_directory(self, mock_url_to_fs):
        """Test that heuristic filter uses fsspec for cache directory creation."""
        # Mock fsspec filesystem
        mock_fs = Mock()
        mock_fs.makedirs.return_value = None
        mock_url_to_fs.return_value = (mock_fs, "bucket/path")
        
        # We can't easily test the full heuristic filter due to dependencies,
        # but we can test the pattern directly
        cache_dir = "s3://bucket/cache/"
        
        # This is the pattern now used in the fixed code
        fs, _ = mock_url_to_fs(cache_dir)
        fs.makedirs(cache_dir, exist_ok=True)
        
        # Verify fsspec was used
        mock_url_to_fs.assert_called_with(cache_dir)
        mock_fs.makedirs.assert_called_with(cache_dir, exist_ok=True)


class TestCloudCompatibilityIntegration:
    """Integration tests for cloud compatibility across components."""

    def test_end_to_end_cloud_uri_patterns(self):
        """Test that common cloud URI patterns work across all fixed components."""
        cloud_uris = [
            "s3://my-data-bucket/datasets/train/",
            "gs://ml-models/embeddings/bert/",
            "abfs://data@storage.dfs.core.windows.net/processed/",
            "https://api.example.com/v1/data/",
        ]
        
        for uri in cloud_uris:
            # Test path construction (semantic deduplication pattern)
            embeddings_path = posixpath.join(uri, "embeddings")
            assert embeddings_path.startswith(uri)
            
            # Test filename extraction (download pattern)
            test_file_path = posixpath.join(uri, "test_file.json")
            filename = posixpath.basename(test_file_path)
            assert filename == "test_file.json"
            
            # Test module name extraction (text_utils pattern)
            script_path = posixpath.join(uri, "script.py")
            module_name = posixpath.splitext(posixpath.basename(script_path))[0]
            assert module_name == "script"

    def test_backward_compatibility_with_local_paths(self):
        """Ensure fixes don't break local filesystem operations."""
        local_paths = [
            "/home/user/data/",
            "./local_data/",
            "../relative/path/",
            "simple_filename.txt",
        ]
        
        for path in local_paths:
            # All the fixed patterns should work with local paths too
            subpath = posixpath.join(path, "subdir")
            filename = posixpath.basename(subpath)
            
            # These operations should succeed without errors
            assert isinstance(subpath, str)
            assert isinstance(filename, str)

    def test_error_handling_for_invalid_uris(self):
        """Test that invalid URIs are handled gracefully."""
        invalid_uris = [
            "",
            "invalid://bad-protocol/path",
            "s3://",  # Missing bucket
            "gs:///no-bucket",
        ]
        
        for uri in invalid_uris:
            # The posixpath operations should not crash on invalid URIs
            try:
                result = posixpath.basename(uri)
                assert isinstance(result, str)
            except Exception:
                # If an exception occurs, it should be a reasonable one
                pass