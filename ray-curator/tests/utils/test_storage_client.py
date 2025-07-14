"""Test suite for storage_client module."""

import concurrent.futures
import pathlib
import tempfile
from unittest.mock import Mock

import pytest

from ray_curator.utils.storage_client import (
    DOWNLOAD_CHUNK_SIZE_BYTES,
    UPLOAD_CHUNK_SIZE_BYTES,
    BackgroundUploader,
    BaseClientConfig,
    StorageClient,
    StoragePrefix,
    is_storage_path,
)


class TestBaseClientConfig:
    """Test suite for BaseClientConfig class."""

    def test_default_values(self) -> None:
        """Test BaseClientConfig with default values."""
        config = BaseClientConfig()

        assert config.max_concurrent_threads == 100
        assert config.operation_timeout_s == 180
        assert config.can_overwrite is False
        assert config.can_delete is False

    def test_custom_values(self) -> None:
        """Test BaseClientConfig with custom values."""
        config = BaseClientConfig(
            max_concurrent_threads=50,
            operation_timeout_s=300,
            can_overwrite=True,
            can_delete=True
        )

        assert config.max_concurrent_threads == 50
        assert config.operation_timeout_s == 300
        assert config.can_overwrite is True
        assert config.can_delete is True

    def test_partial_custom_values(self) -> None:
        """Test BaseClientConfig with some custom values."""
        config = BaseClientConfig(
            max_concurrent_threads=200,
            can_overwrite=True
        )

        assert config.max_concurrent_threads == 200
        assert config.operation_timeout_s == 180  # default
        assert config.can_overwrite is True
        assert config.can_delete is False  # default


class ConcreteStoragePrefix(StoragePrefix):
    """Concrete implementation of StoragePrefix for testing."""

    @property
    def path(self) -> str:
        """Return the full path for this prefix."""
        return f"test://{self._input}"


class TestStoragePrefix:
    """Test suite for StoragePrefix class."""

    def test_prefix_property_with_slash(self) -> None:
        """Test prefix property when input contains a slash."""
        storage_prefix = ConcreteStoragePrefix("bucket/folder/subfolder")
        assert storage_prefix.prefix == "folder/subfolder"

    def test_prefix_property_without_slash(self) -> None:
        """Test prefix property when input doesn't contain a slash."""
        storage_prefix = ConcreteStoragePrefix("bucket")
        assert storage_prefix.prefix == ""

    def test_prefix_property_with_multiple_slashes(self) -> None:
        """Test prefix property when input contains multiple slashes."""
        storage_prefix = ConcreteStoragePrefix("bucket/folder/subfolder/file.txt")
        assert storage_prefix.prefix == "folder/subfolder/file.txt"

    def test_str_method(self) -> None:
        """Test __str__ method returns the full path."""
        storage_prefix = ConcreteStoragePrefix("bucket/folder")
        assert str(storage_prefix) == "test://bucket/folder"

    def test_path_property(self) -> None:
        """Test path property returns expected format."""
        storage_prefix = ConcreteStoragePrefix("bucket/folder")
        assert storage_prefix.path == "test://bucket/folder"

    def test_storageprefix_instantiation(self) -> None:
        """Test that StoragePrefix can be instantiated with input."""
        storage_prefix = StoragePrefix("test")
        assert storage_prefix._input == "test"


class ConcreteBackgroundUploader(BackgroundUploader):
    """Concrete implementation of BackgroundUploader for testing."""

    def add_task_file(self, _local_path: pathlib.Path, _remote_path: str) -> None:
        """Add a mock file upload task."""
        # Mock implementation for testing
        future = concurrent.futures.Future()
        future.set_result(None)
        self.futures.append(future)


class TestBackgroundUploader:
    """Test suite for BackgroundUploader class."""

    def test_initialization(self) -> None:
        """Test BackgroundUploader initialization."""
        mock_client = Mock()
        chunk_size = 1024

        uploader = ConcreteBackgroundUploader(mock_client, chunk_size)

        assert uploader.client == mock_client
        assert uploader.chunk_size_bytes == chunk_size
        assert isinstance(uploader.executor, concurrent.futures.ThreadPoolExecutor)
        assert uploader.futures == []

    def test_add_task_file(self) -> None:
        """Test adding a task file to the uploader."""
        mock_client = Mock()
        uploader = ConcreteBackgroundUploader(mock_client, 1024)

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = pathlib.Path(tmp_dir) / "test.txt"
            remote_path = "bucket/test.txt"

            uploader.add_task_file(local_path, remote_path)

            assert len(uploader.futures) == 1
            assert uploader.futures[0].done()

    def test_block_until_done(self) -> None:
        """Test blocking until all tasks are done."""
        mock_client = Mock()
        uploader = ConcreteBackgroundUploader(mock_client, 1024)

        # Add some tasks
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file1 = pathlib.Path(tmp_dir) / "test1.txt"
            test_file2 = pathlib.Path(tmp_dir) / "test2.txt"
            uploader.add_task_file(test_file1, "bucket/test1.txt")
            uploader.add_task_file(test_file2, "bucket/test2.txt")

        # This should complete without hanging since our mock tasks complete immediately
        uploader.block_until_done()

        # All futures should be done
        assert all(future.done() for future in uploader.futures)

    def test_abstract_class_cannot_be_instantiated(self) -> None:
        """Test that BackgroundUploader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BackgroundUploader(Mock(), 1024)  # type: ignore[abstract]


class TestStorageClient:
    """Test suite for StorageClient class."""

    def test_abstract_class_cannot_be_instantiated(self) -> None:
        """Test that StorageClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            StorageClient()  # type: ignore[abstract]

    def test_all_methods_are_abstract(self) -> None:
        """Test that all methods in StorageClient are abstract."""
        abstract_methods = StorageClient.__abstractmethods__

        expected_methods = {
            "object_exists",
            "upload_bytes",
            "upload_bytes_uri",
            "download_object_as_bytes",
            "download_objects_as_bytes",
            "list_recursive_directory",
            "list_recursive",
            "upload_file",
            "sync_remote_to_local",
            "make_background_uploader"
        }

        assert abstract_methods == expected_methods


class TestIsStoragePath:
    """Test suite for is_storage_path function."""

    def test_none_path(self) -> None:
        """Test is_storage_path with None path."""
        assert is_storage_path(None, "s3") is False

    def test_s3_path_valid(self) -> None:
        """Test is_storage_path with valid S3 path."""
        assert is_storage_path("s3://bucket/path", "s3") is True

    def test_s3_path_invalid_protocol(self) -> None:
        """Test is_storage_path with S3 path but wrong protocol."""
        assert is_storage_path("s3://bucket/path", "azure") is False

    def test_azure_path_valid(self) -> None:
        """Test is_storage_path with valid Azure path."""
        assert is_storage_path("azure://container/path", "azure") is True

    def test_azure_path_invalid_protocol(self) -> None:
        """Test is_storage_path with Azure path but wrong protocol."""
        assert is_storage_path("azure://container/path", "s3") is False

    def test_local_path(self) -> None:
        """Test is_storage_path with local path."""
        assert is_storage_path("/local/path", "s3") is False
        assert is_storage_path("./relative/path", "s3") is False

    def test_http_path(self) -> None:
        """Test is_storage_path with HTTP path."""
        assert is_storage_path("http://example.com/path", "s3") is False
        assert is_storage_path("https://example.com/path", "azure") is False

    def test_empty_string(self) -> None:
        """Test is_storage_path with empty string."""
        assert is_storage_path("", "s3") is False

    def test_protocol_substring(self) -> None:
        """Test is_storage_path when protocol appears as substring."""
        assert is_storage_path("not-s3://bucket/path", "s3") is False
        assert is_storage_path("prefix-azure://container/path", "azure") is False

    def test_case_sensitivity(self) -> None:
        """Test is_storage_path case sensitivity."""
        assert is_storage_path("S3://bucket/path", "s3") is False
        assert is_storage_path("s3://bucket/path", "S3") is False

    def test_various_protocols(self) -> None:
        """Test is_storage_path with various protocols."""
        test_cases = [
            ("gcs://bucket/path", "gcs", True),
            ("file://path", "file", True),
            ("ftp://server/path", "ftp", True),
            ("custom://path", "custom", True),
        ]

        for path, protocol, expected in test_cases:
            assert is_storage_path(path, protocol) == expected


class TestConstants:
    """Test suite for module constants."""

    def test_download_chunk_size_bytes(self) -> None:
        """Test DOWNLOAD_CHUNK_SIZE_BYTES constant."""
        assert DOWNLOAD_CHUNK_SIZE_BYTES == 10 * 1024 * 1024  # 10 MB

    def test_upload_chunk_size_bytes(self) -> None:
        """Test UPLOAD_CHUNK_SIZE_BYTES constant."""
        assert UPLOAD_CHUNK_SIZE_BYTES == 100 * 1024 * 1024  # 100 MB
