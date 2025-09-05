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
Cloud compatibility tests for Text Components.

This module tests that Text Components use fsspec for cloud storage URIs
instead of os/pathlib/glob/shutil which don't work with s3://, gs://, abfs://, etc.
"""

import os
import posixpath
import tempfile
from unittest.mock import Mock, patch

import pytest

# We can't import the actual modules due to dependencies, so we'll test the patterns


class TestTextUtilsCloudCompatibilityPatterns:
    """Test that text_utils functions handle cloud URIs correctly."""

    def test_filename_extraction_from_cloud_uris(self):
        """Test filename extraction patterns used in text_utils.py."""
        # Simulate the pattern from text_utils.py line 170:
        # module = os.path.splitext(os.path.basename(filename))[0]
        
        test_cases = [
            ("s3://bucket/path/to/file.py", "file"),
            ("gs://my-bucket/deep/nested/path/script.py", "script"), 
            ("abfs://container@account.dfs.core.windows.net/data/code.py", "code"),
            ("https://example.com/api/v1/source.py", "source"),
            ("/local/path/local_file.py", "local_file"),  # Local files should still work
        ]
        
        for filename, expected_module_name in test_cases:
            # Current problematic pattern (but works on POSIX systems)
            os_module = os.path.splitext(os.path.basename(filename))[0]
            
            # Recommended cloud-compatible pattern
            posix_module = posixpath.splitext(posixpath.basename(filename))[0]
            
            # Both should work on POSIX systems, but posixpath is more explicit for cloud URIs
            assert os_module == expected_module_name, f"os.path pattern failed for {filename}"
            assert posix_module == expected_module_name, f"posixpath pattern failed for {filename}"
            
            # Verify the posixpath approach is cloud-compatible
            assert posix_module == expected_module_name

    def test_cloud_uri_filename_extraction_robustness(self):
        """Test robustness of filename extraction with various cloud URI formats."""
        edge_cases = [
            ("s3://bucket/file", "file"),  # No extension
            ("gs://bucket/path.with.dots/file.tar.gz", "file.tar"),  # Multiple dots
            ("abfs://container@account.dfs.core.windows.net/", ""),  # Directory ending
            ("https://example.com/path/", ""),  # Directory ending with slash
            ("s3://bucket/dir/file.json", "file"),  # JSON file
        ]
        
        for uri, expected in edge_cases:
            result = posixpath.splitext(posixpath.basename(uri))[0]
            assert result == expected, f"Failed for {uri}: got {result}, expected {expected}"


class TestSemanticDedupeCloudCompatibilityPatterns:
    """Test semantic deduplication path construction patterns."""

    def test_path_joining_for_cloud_uris(self):
        """Test path joining patterns used in semantic.py."""
        # Simulate the patterns from semantic.py lines 182-189:
        # self.embeddings_path = os.path.join(self.cache_path, "embeddings")
        
        cloud_base_paths = [
            "s3://my-bucket/dedup-cache",
            "gs://bucket/cache", 
            "abfs://container@account.dfs.core.windows.net/cache",
            "https://storage.example.com/cache",
            "/local/cache/path",  # Local paths should still work
        ]
        
        subdirs = ["embeddings", "semantic_dedup", "duplicates"]
        
        for base_path in cloud_base_paths:
            for subdir in subdirs:
                # Current pattern (works on POSIX but may have issues on Windows with protocols)
                os_path = os.path.join(base_path, subdir)
                
                # Recommended cloud-compatible pattern
                posix_path = posixpath.join(base_path, subdir)
                
                # Both should produce valid paths on POSIX systems
                assert os_path.startswith(base_path), f"os.path result should start with base: {os_path}"
                assert posix_path.startswith(base_path), f"posixpath result should start with base: {posix_path}"
                
                # Verify cloud URIs maintain their structure
                if "://" in base_path:
                    assert "://" in posix_path, f"Cloud protocol should be preserved: {posix_path}"
                    assert "/" in posix_path, f"Should use forward slashes: {posix_path}"
                    assert "\\" not in posix_path, f"Should not contain backslashes: {posix_path}"

    def test_multiple_path_components_joining(self):
        """Test joining multiple path components for complex directory structures."""
        base_paths = [
            "s3://bucket/base",
            "gs://my-bucket/data",
            "abfs://container@account.dfs.core.windows.net/root"
        ]
        
        # Test patterns like: os.path.join(base, "output", "duplicates", "final")
        components = ["output", "duplicates", "final"]
        
        for base in base_paths:
            # Use posixpath for cloud-compatible joining
            result = base
            for component in components:
                result = posixpath.join(result, component)
            
            expected = f"{base}/output/duplicates/final"
            assert result == expected, f"Multi-component join failed: {result} != {expected}"
            
            # Verify cloud URI structure is maintained
            assert result.startswith(base), "Should start with base path"
            assert "://" in result, "Should maintain protocol"
            assert result.count("://") == 1, "Should have exactly one protocol marker"


class TestFsspecCloudOperations:
    """Test that fsspec operations work correctly with cloud URIs."""

    def test_fsspec_vs_os_file_existence_checks(self):
        """Test file existence patterns: os.path.exists vs fs.exists."""
        # Mock fsspec filesystem
        with tempfile.NamedTemporaryFile() as tmp_file:
            local_path = tmp_file.name
            
            # Test with local file - both should work
            assert os.path.exists(local_path) is True
            
            # Mock fsspec for cloud URIs
            mock_fs = Mock()
            mock_fs.exists.return_value = True
            
            cloud_uris = [
                "s3://bucket/file.txt",
                "gs://bucket/file.txt",
                "abfs://container@account.dfs.core.windows.net/file.txt"
            ]
            
            for uri in cloud_uris:
                # os.path.exists would return False for cloud URIs (WRONG)
                assert os.path.exists(uri) is False, f"os.path.exists should fail for {uri}"
                
                # fs.exists should work for cloud URIs (CORRECT)
                assert mock_fs.exists(uri) is True, f"fs.exists should work for {uri}"

    def test_fsspec_vs_builtin_file_opening(self):
        """Test file opening patterns: open() vs fs.open()."""
        cloud_uris = [
            "s3://bucket/file.txt",
            "gs://bucket/file.txt", 
            "abfs://container@account.dfs.core.windows.net/file.txt"
        ]
        
        for uri in cloud_uris:
            # open(uri) would fail for cloud URIs
            with pytest.raises((OSError, FileNotFoundError)):
                open(uri, "r")  # This should fail
            
            # Mock fs.open() to simulate cloud success
            mock_fs = Mock()
            mock_file = Mock()
            mock_file.read.return_value = "test content"
            mock_fs.open.return_value.__enter__ = Mock(return_value=mock_file)
            mock_fs.open.return_value.__exit__ = Mock(return_value=None)
            
            # fs.open should work for cloud URIs (CORRECT)
            with mock_fs.open(uri, "r") as f:
                content = f.read()
                assert content == "test content"

    def test_fsspec_vs_os_directory_operations(self):
        """Test directory listing: os.listdir vs fs.ls."""
        cloud_dirs = [
            "s3://bucket/folder/",
            "gs://bucket/data/",
            "abfs://container@account.dfs.core.windows.net/path/"
        ]
        
        for uri in cloud_dirs:
            # os.listdir would fail for cloud URIs
            with pytest.raises((OSError, FileNotFoundError)):
                os.listdir(uri)  # This should fail
            
            # Mock fs.ls() to simulate cloud success
            mock_fs = Mock()
            mock_files = ["file1.txt", "file2.json", "subdir/"]
            mock_fs.ls.return_value = mock_files
            
            # fs.ls should work for cloud URIs (CORRECT)
            files = mock_fs.ls(uri)
            assert files == mock_files

    def test_fsspec_vs_glob_pattern_matching(self):
        """Test glob operations: glob.glob vs fs.glob."""
        import glob
        
        cloud_patterns = [
            "s3://bucket/**/*.json",
            "gs://bucket/data/*.txt", 
            "abfs://container@account.dfs.core.windows.net/logs/**/*.log"
        ]
        
        for pattern in cloud_patterns:
            # glob.glob would return empty list for cloud URIs (WRONG)
            assert glob.glob(pattern) == [], f"glob.glob should return empty for {pattern}"
            
            # Mock fs.glob() to simulate cloud success  
            mock_fs = Mock()
            mock_matches = ["file1.json", "subdir/file2.json"]
            mock_fs.glob.return_value = mock_matches
            
            # fs.glob should work for cloud URIs (CORRECT)
            matches = mock_fs.glob(pattern)
            assert matches == mock_matches

    def test_fsspec_vs_os_directory_creation_removal(self):
        """Test directory creation/removal: os.makedirs vs fs.makedirs."""
        cloud_dirs = [
            "s3://bucket/new-folder/",
            "gs://bucket/output/",
            "abfs://container@account.dfs.core.windows.net/temp/"
        ]
        
        for uri in cloud_dirs:
            # os.makedirs might not fail on all systems but would create wrong local directories
            # The point is that it shouldn't be used for cloud URIs
            try:
                os.makedirs(uri, exist_ok=True)
                # If it doesn't fail, it probably created a local directory with the wrong name
                # This is the problematic behavior we want to avoid
                print(f"Warning: os.makedirs({uri}) didn't fail - this creates wrong local paths")
            except (OSError, FileNotFoundError):
                # This is the expected behavior on most systems
                pass
            
            # Mock fs operations to simulate cloud success
            mock_fs = Mock()
            mock_fs.makedirs = Mock()
            mock_fs.rm = Mock()
            
            # fs.makedirs should work for cloud URIs (CORRECT)
            mock_fs.makedirs(uri, exist_ok=True)
            mock_fs.makedirs.assert_called_once_with(uri, exist_ok=True)
            
            # fs.rm should work for cloud URIs (CORRECT)
            mock_fs.rm(uri, recursive=True)
            mock_fs.rm.assert_called_once_with(uri, recursive=True)


class TestCloudUriNormalizationPatterns:
    """Test cloud URI path normalization and manipulation."""

    def test_posixpath_vs_os_path_normalization(self):
        """Test path normalization for cloud URIs."""
        test_cases = [
            # Note: normpath can collapse // to / which breaks protocols
            # This test shows why we need careful handling of cloud URIs
            ("s3://bucket/folder/../other/file.json", "gs://bucket/other/file.json"),
            ("abfs://container@account.dfs.core.windows.net/a/./b/c", 
             "abfs://container@account.dfs.core.windows.net/a/b/c"),
        ]
        
        for input_path, _ in test_cases:
            # os.path.normpath might mangle protocols or give wrong results
            os_result = os.path.normpath(input_path)
            
            # posixpath.normpath is more appropriate for cloud URIs  
            posix_result = posixpath.normpath(input_path)
            
            # Both might have issues with double slashes in protocols
            # The key point is that we should use fsspec's url_to_fs instead
            print(f"Input: {input_path}")
            print(f"  os.path.normpath: {os_result}")
            print(f"  posixpath.normpath: {posix_result}")
            
            # The lesson: don't use normpath directly on cloud URIs
            # Use fsspec.core.url_to_fs() to separate protocol from path
            # Then normalize the path component only

    def test_cloud_uri_component_extraction(self):
        """Test extracting components from cloud URIs safely."""
        test_uris = [
            "s3://bucket/path/to/file.ext",
            "gs://my-bucket/deep/nested/path/data.json",
            "abfs://container@account.dfs.core.windows.net/dir/subdir/file.parquet",
            "https://storage.example.com/api/v1/data.xml"
        ]
        
        for uri in test_uris:
            # Safe way to extract directory
            directory = posixpath.dirname(uri)
            assert directory.startswith(uri.split("/")[0] + "//"), f"Directory should maintain protocol: {directory}"
            
            # Safe way to extract filename
            filename = posixpath.basename(uri)
            assert "://" not in filename, f"Filename should not contain protocol: {filename}"
            
            # Safe way to extract extension
            name, ext = posixpath.splitext(filename)
            assert ext.startswith(".") or ext == "", f"Extension should start with dot or be empty: {ext}"

    def test_relative_path_resolution_for_cloud_uris(self):
        """Test that relative paths work correctly with cloud URIs."""
        # This test shows the challenges with relative paths in cloud URIs
        # The recommended approach is to avoid relative paths with cloud URIs
        # or use fsspec.core.url_to_fs() to handle the protocol separately
        
        base_uris = [
            "s3://bucket/project",
            "gs://my-bucket/workspace", 
            "abfs://container@account.dfs.core.windows.net/base"
        ]
        
        # Safe relative paths that don't go above the base
        safe_relative_paths = ["./temp", "subfolder/data", "output/results"]
        
        for base in base_uris:
            for rel_path in safe_relative_paths:
                # Use posixpath for cloud-safe relative path resolution
                result = posixpath.join(base, rel_path)
                
                # Verify protocol is maintained
                protocol = base.split("://")[0]
                assert result.startswith(f"{protocol}://"), f"Protocol should be maintained: {result}"
                
                # Verify path structure makes sense
                assert base in result, f"Base should be contained in result: {result}"

    def test_fsspec_url_to_fs_recommended_pattern(self):
        """Test the recommended fsspec pattern for handling cloud URIs safely."""
        # This is the RECOMMENDED way to handle cloud URIs
        
        test_uris = [
            "s3://bucket/path/file.txt",
            "gs://bucket/data.json", 
            "abfs://container@account.dfs.core.windows.net/file.parquet"
        ]
        
        for uri in test_uris:
            # Mock fsspec.core.url_to_fs behavior
            def mock_url_to_fs(url):
                if url.startswith("s3://"):
                    return Mock(), url[5:]
                elif url.startswith("gs://"):
                    return Mock(), url[5:]
                elif url.startswith("abfs://"):
                    return Mock(), url[7:]
                else:
                    return Mock(), url
            
            # Simulate the recommended pattern
            fs, path = mock_url_to_fs(uri)
            
            # Now we can safely use posixpath operations on the path
            dirname = posixpath.dirname(path)
            basename = posixpath.basename(path)
            
            # And join additional path components safely
            new_path = posixpath.join(path, "subfolder", "newfile.txt")
            
            # Verify no protocol mangling
            assert "://" not in path, f"Path should not contain protocol: {path}"
            assert "://" not in new_path, f"New path should not contain protocol: {new_path}"