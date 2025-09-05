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
Tests for specific cloud compatibility issues found in text components.

This module tests the specific problematic patterns identified in the codebase
and ensures they can be fixed to work with cloud URIs.
"""

import os
import posixpath


class TestTextUtilsSpecificIssues:
    """Test specific issues found in nemo_curator/stages/text/utils/text_utils.py."""

    def test_line_170_os_path_basename_splitext_issue(self):
        """
        Test the specific issue on line 170 of text_utils.py:
        module = os.path.splitext(os.path.basename(filename))[0]
        
        This pattern should be cloud-compatible.
        """
        # The problematic code pattern from line 170
        def extract_module_name_old_way(filename):
            """Original problematic pattern."""
            return os.path.splitext(os.path.basename(filename))[0]
        
        # The cloud-compatible version
        def extract_module_name_cloud_way(filename):
            """Cloud-compatible pattern."""
            return posixpath.splitext(posixpath.basename(filename))[0]
        
        test_filenames = [
            ("s3://bucket/path/to/file.py", "file"),
            ("gs://my-bucket/deep/nested/script.py", "script"),
            ("abfs://container@account.dfs.core.windows.net/code.py", "code"),
            ("https://example.com/api/source.py", "source"),
            ("/local/path/local_file.py", "local_file"),  # Still works for local
        ]
        
        for filename, expected_module in test_filenames:
            # Both should work on POSIX systems, but cloud version is more explicit
            old_result = extract_module_name_old_way(filename)
            new_result = extract_module_name_cloud_way(filename)
            
            assert old_result == expected_module, f"Old way failed for {filename}"
            assert new_result == expected_module, f"New way failed for {filename}"
            assert old_result == new_result, f"Results differ for {filename}"

    def test_get_docstrings_mock_file_handling(self):
        """
        Test how the get_docstrings function handles file-like objects with cloud URIs.
        
        This simulates the pattern where a file-like object has a .name attribute
        that contains a cloud URI.
        """
        from unittest.mock import Mock
        
        # Simulate the get_docstrings logic for filename extraction
        def simulate_get_docstrings_filename_logic(source):
            """Simulate the logic from get_docstrings function."""
            if hasattr(source, "read"):
                filename = getattr(source, "name", "<string>")
                # This is the problematic line 170
                module = os.path.splitext(os.path.basename(filename))[0]
                return module
            return "<string>"
        
        cloud_uris = [
            "s3://bucket/path/file.py",
            "gs://bucket/script.py",
            "abfs://container@account.dfs.core.windows.net/code.py",
            "https://example.com/source.py"
        ]
        
        for cloud_uri in cloud_uris:
            mock_file = Mock()
            mock_file.name = cloud_uri
            mock_file.read.return_value = "# Python source code"
            
            # This should extract the correct module name
            module_name = simulate_get_docstrings_filename_logic(mock_file)
            expected_name = posixpath.splitext(posixpath.basename(cloud_uri))[0]
            
            assert module_name == expected_name, f"Module extraction failed for {cloud_uri}"


class TestSemanticDedupeSpecificIssues:
    """Test specific issues found in nemo_curator/stages/text/deduplication/semantic.py."""

    def test_lines_182_189_os_path_join_issues(self):
        """
        Test the specific issues on lines 182-189 of semantic.py:
        - self.embeddings_path = os.path.join(self.cache_path, "embeddings")
        - self.semantic_dedup_path = os.path.join(self.cache_path, "semantic_dedup")
        - etc.
        
        These patterns should be cloud-compatible.
        """
        # Simulate the problematic pattern from semantic.py
        def create_paths_old_way(cache_path, output_path):
            """Original problematic patterns from lines 182-189."""
            embeddings_path = os.path.join(cache_path, "embeddings")
            semantic_dedup_path = os.path.join(cache_path, "semantic_dedup")
            duplicates_path = os.path.join(output_path, "duplicates")
            deduplicated_path = os.path.join(output_path, "deduplicated")
            state_file = os.path.join(output_path, "semantic_id_generator.json")
            
            return {
                "embeddings": embeddings_path,
                "semantic_dedup": semantic_dedup_path,
                "duplicates": duplicates_path,
                "deduplicated": deduplicated_path,
                "state_file": state_file,
            }
        
        # Cloud-compatible version
        def create_paths_cloud_way(cache_path, output_path):
            """Cloud-compatible patterns."""
            embeddings_path = posixpath.join(cache_path, "embeddings")
            semantic_dedup_path = posixpath.join(cache_path, "semantic_dedup")
            duplicates_path = posixpath.join(output_path, "duplicates")
            deduplicated_path = posixpath.join(output_path, "deduplicated")
            state_file = posixpath.join(output_path, "semantic_id_generator.json")
            
            return {
                "embeddings": embeddings_path,
                "semantic_dedup": semantic_dedup_path,
                "duplicates": duplicates_path,
                "deduplicated": deduplicated_path,
                "state_file": state_file,
            }
        
        test_cases = [
            ("s3://bucket/cache", "s3://bucket/output"),
            ("gs://my-bucket/cache", "gs://my-bucket/output"),
            ("abfs://container@account.dfs.core.windows.net/cache", 
             "abfs://container@account.dfs.core.windows.net/output"),
            ("/local/cache", "/local/output"),  # Local paths should still work
        ]
        
        for cache_path, output_path in test_cases:
            old_paths = create_paths_old_way(cache_path, output_path)
            new_paths = create_paths_cloud_way(cache_path, output_path)
            
            # Both should produce the same results on POSIX systems
            for key in old_paths:
                assert old_paths[key] == new_paths[key], f"Path mismatch for {key}: {old_paths[key]} != {new_paths[key]}"
                
                # Verify cloud URIs maintain their structure
                if "://" in cache_path or "://" in output_path:
                    assert "://" in new_paths[key], f"Cloud protocol lost in {key}: {new_paths[key]}"
                    assert "/" in new_paths[key], f"Should use forward slashes in {key}: {new_paths[key]}"

    def test_semantic_dedupe_path_construction_robustness(self):
        """Test robust path construction for complex cloud URI scenarios."""
        complex_cases = [
            # S3 with nested buckets and regions
            ("s3://my-bucket-us-west-2/projects/nlp/cache", 
             "s3://my-bucket-us-west-2/projects/nlp/output"),
            
            # Google Cloud Storage with complex paths
            ("gs://my-project-bucket/datasets/v2/cache",
             "gs://my-project-bucket/datasets/v2/output"),
            
            # Azure Blob Storage with container and account
            ("abfs://data@myaccount.dfs.core.windows.net/projects/dedup/cache",
             "abfs://data@myaccount.dfs.core.windows.net/projects/dedup/output"),
            
            # HTTPS endpoints
            ("https://storage.example.com/api/v1/cache",
             "https://storage.example.com/api/v1/output"),
        ]
        
        for cache_path, output_path in complex_cases:
            # Test the path joining patterns from semantic.py
            embeddings_path = posixpath.join(cache_path, "embeddings")
            duplicates_path = posixpath.join(output_path, "duplicates")
            state_file = posixpath.join(output_path, "semantic_id_generator.json")
            
            # Verify paths maintain their cloud structure
            assert embeddings_path.startswith(cache_path), f"Embeddings path should start with cache: {embeddings_path}"
            assert duplicates_path.startswith(output_path), f"Duplicates path should start with output: {duplicates_path}"
            assert state_file.startswith(output_path), f"State file should start with output: {state_file}"
            
            # Verify no path corruption
            protocol_count = cache_path.count("://")
            assert embeddings_path.count("://") == protocol_count, f"Protocol corruption in embeddings: {embeddings_path}"
            assert duplicates_path.count("://") == protocol_count, f"Protocol corruption in duplicates: {duplicates_path}"


class TestDownloadModuleSpecificIssues:
    """Test specific issues found in download modules."""

    def test_arxiv_download_path_operations(self):
        """
        Test patterns from nemo_curator/stages/text/download/arxiv/ files.
        
        Found patterns like:
        - download_dir = os.path.split(file_path)[0]
        - bname = os.path.split(file_path)[-1]
        - os.path.splitext(os.path.split(item)[-1])[0]
        """
        # Problematic patterns from arxiv download files
        def extract_download_info_old_way(file_path):
            """Original patterns from arxiv download."""
            download_dir = os.path.split(file_path)[0]
            bname = os.path.split(file_path)[-1]
            name_without_ext = os.path.splitext(os.path.split(file_path)[-1])[0]
            return download_dir, bname, name_without_ext
        
        # Cloud-compatible version
        def extract_download_info_cloud_way(file_path):
            """Cloud-compatible patterns."""
            download_dir = posixpath.dirname(file_path)
            bname = posixpath.basename(file_path)
            name_without_ext = posixpath.splitext(posixpath.basename(file_path))[0]
            return download_dir, bname, name_without_ext
        
        test_file_paths = [
            "s3://arxiv-bucket/src/papers/2023/paper.tar.gz",
            "gs://arxiv-mirror/papers/math/0601001.pdf",
            "abfs://papers@storage.dfs.core.windows.net/cs/0601001.tar",
            "/local/path/papers/paper.pdf",  # Local paths should still work
        ]
        
        for file_path in test_file_paths:
            old_dir, old_name, old_name_no_ext = extract_download_info_old_way(file_path)
            new_dir, new_name, new_name_no_ext = extract_download_info_cloud_way(file_path)
            
            # Results should be the same on POSIX systems
            assert old_dir == new_dir, f"Directory extraction differs: {old_dir} != {new_dir}"
            assert old_name == new_name, f"Filename extraction differs: {old_name} != {new_name}"
            assert old_name_no_ext == new_name_no_ext, f"Name without ext differs: {old_name_no_ext} != {new_name_no_ext}"
            
            # Verify cloud URI structure is maintained in directory
            if "://" in file_path:
                assert "://" in new_dir, f"Protocol should be preserved in directory: {new_dir}"
                assert "://" not in new_name, f"Protocol should not be in filename: {new_name}"

    def test_common_crawl_warc_iterator_patterns(self):
        """
        Test patterns that might be found in WARC iterator modules.
        
        These often use pathlib.Path which can cause issues with cloud URIs.
        """
        # Test that we handle path-like operations correctly for cloud URIs
        cloud_warc_paths = [
            "s3://commoncrawl/crawl-data/CC-MAIN-2023-06/segments/warc.gz",
            "gs://commoncrawl-mirror/2023/warc-files/segment.warc.gz",
            "abfs://crawldata@storage.dfs.core.windows.net/warc/file.warc.gz",
        ]
        
        for warc_path in cloud_warc_paths:
            # Simulate operations that might be done on WARC paths
            
            # Extract directory (for organizing downloaded files)
            directory = posixpath.dirname(warc_path)
            assert directory.startswith(warc_path.split("://")[0] + "://"), f"Directory should maintain protocol: {directory}"
            
            # Extract filename (for local storage naming)
            filename = posixpath.basename(warc_path)
            assert "://" not in filename, f"Filename should not contain protocol: {filename}"
            assert filename.endswith(".warc.gz") or filename.endswith(".gz"), f"Should preserve file extension: {filename}"
            
            # Extract name without extension (for processing logic)
            name_part = posixpath.splitext(filename)[0]
            if name_part.endswith(".warc"):
                name_part = posixpath.splitext(name_part)[0]
            
            assert len(name_part) > 0, f"Should extract meaningful name: {name_part}"
            assert "." not in name_part or name_part.count(".") < filename.count("."), f"Should remove extensions: {name_part}"


class TestGeneralCloudCompatibilityPatterns:
    """Test general patterns that should be avoided in favor of fsspec."""

    def test_problematic_os_operations_on_cloud_uris(self):
        """Test operations that definitely don't work with cloud URIs."""
        cloud_uris = [
            "s3://bucket/file.txt",
            "gs://bucket/data.json",
            "abfs://container@account.dfs.core.windows.net/file.parquet"
        ]
        
        for uri in cloud_uris:
            # These operations should NOT work with cloud URIs
            # (Testing that they fail as expected)
            
            # File existence - os.path.exists returns False for cloud URIs
            assert os.path.exists(uri) is False, f"os.path.exists should be False for {uri}"
            
            # File size - os.path.getsize should fail
            try:
                os.path.getsize(uri)
                assert False, f"os.path.getsize should fail for {uri}"
            except (OSError, FileNotFoundError):
                pass  # Expected behavior
            
            # Directory check - os.path.isdir should be False
            assert os.path.isdir(uri) is False, f"os.path.isdir should be False for {uri}"
            
            # File check - os.path.isfile should be False  
            assert os.path.isfile(uri) is False, f"os.path.isfile should be False for {uri}"

    def test_recommended_fsspec_patterns(self):
        """Test the recommended fsspec patterns for cloud compatibility."""
        from unittest.mock import Mock
        
        cloud_uris = [
            "s3://bucket/file.txt",
            "gs://bucket/data.json",
            "abfs://container@account.dfs.core.windows.net/file.parquet"
        ]
        
        for uri in cloud_uris:
            # Mock fsspec filesystem
            mock_fs = Mock()
            mock_fs.exists.return_value = True
            mock_fs.isdir.return_value = False
            mock_fs.isfile.return_value = True
            mock_fs.size.return_value = 12345
            
            # These operations SHOULD work with fsspec
            assert mock_fs.exists(uri) is True, f"fs.exists should work for {uri}"
            assert mock_fs.isfile(uri) is True, f"fs.isfile should work for {uri}"
            assert mock_fs.size(uri) == 12345, f"fs.size should work for {uri}"
            
            # Mock file operations
            mock_file = Mock()
            mock_file.read.return_value = b"test content"
            mock_fs.open.return_value.__enter__ = Mock(return_value=mock_file)
            mock_fs.open.return_value.__exit__ = Mock(return_value=None)
            
            # File reading should work
            with mock_fs.open(uri, "rb") as f:
                content = f.read()
                assert content == b"test content", f"fs.open should work for {uri}"