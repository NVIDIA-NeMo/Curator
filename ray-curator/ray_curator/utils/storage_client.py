# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base storage client interface for various storage systems.

This module provides a base interface for storage client implementations
(S3, Azure Blob Storage, etc.) with common operations for interacting
with cloud storage systems.
"""

import abc
import concurrent.futures
import pathlib
from typing import Any

import attrs

# Constants for chunk sizes
DOWNLOAD_CHUNK_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
UPLOAD_CHUNK_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


@attrs.define
class BaseClientConfig:
    """Base configuration class for storage clients.

    Attributes:
        max_concurrent_threads (int): Maximum number of concurrent threads (default: 100).
        operation_timeout_s (int): Timeout for operations in seconds (default: 180).
        can_overwrite (bool): Whether the client can overwrite existing objects (default: False).
        can_delete (bool): Whether the client can delete objects (default: False).

    """

    max_concurrent_threads: int = attrs.field(default=100)
    operation_timeout_s: int = attrs.field(default=180)
    can_overwrite: bool = attrs.field(default=False)
    can_delete: bool = attrs.field(default=False)


@attrs.define
class StoragePrefix:
    """Base class for representing a storage path prefix.

    This is extended by specific implementations like S3Prefix and AzurePrefix.
    """

    _input: str = attrs.field()

    @property
    @abc.abstractmethod
    def path(self) -> str:
        """Return the full path for this prefix."""

    @property
    def prefix(self) -> str:
        """Return the prefix for this storage path.

        Returns:
            The prefix for this storage path.

        """
        parts = self._input.split("/", 1)
        return parts[1] if len(parts) > 1 else ""

    def __str__(self) -> str:
        """Return a string representation."""
        return self.path


class BackgroundUploader(abc.ABC):
    """Abstract base class for background uploaders.

    Attributes:
        client: The storage client instance.
        chunk_size_bytes (int): The size of chunks for uploading.
        executor (ThreadPoolExecutor): The thread pool executor for background tasks.
        futures (List[Future]): List of futures for tracking upload tasks.

    """

    def __init__(self, client: object, chunk_size_bytes: int) -> None:
        """Initialize the BackgroundUploader with the given client and chunk size.

        Args:
            client: The storage client instance.
            chunk_size_bytes: The size of chunks to use for uploading.

        """
        self.client = client
        self.chunk_size_bytes = chunk_size_bytes
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.futures: list[concurrent.futures.Future[None]] = []

    @abc.abstractmethod
    def add_task_file(self, local_path: pathlib.Path, remote_path: str) -> None:
        """Add a file upload task to the background uploader.

        Args:
            local_path: Path to the local file to upload.
            remote_path: Path in the storage system where the file will be uploaded.

        """

    def block_until_done(self) -> None:
        """Wait for all background tasks to complete."""
        concurrent.futures.wait(self.futures)
        self.executor.shutdown(wait=True)


class StorageClient(abc.ABC):
    """Abstract base class for storage clients.

    This class defines the interface for storage client operations such as checking object
    existence, uploading and downloading objects, and listing objects.
    """

    @abc.abstractmethod
    def object_exists(self, dest: StoragePrefix) -> bool:
        """Check if an object exists at the specified path.

        Args:
            dest: The storage prefix of the object to check.

        Returns:
            bool: True if the object exists, False otherwise.

        """

    @abc.abstractmethod
    def upload_bytes(self, dest: StoragePrefix, data: bytes) -> None:
        """Upload bytes data to the specified storage path.

        Args:
            dest: The storage prefix where the object will be stored.
            data: The bytes data to upload.

        Raises:
            ValueError: If the object already exists and overwriting is not allowed.

        """

    @abc.abstractmethod
    def upload_bytes_uri(self, uri: str, data: bytes, chunk_size_bytes: int = UPLOAD_CHUNK_SIZE_BYTES) -> None:
        """Upload bytes data to the specified URI.

        Args:
            uri: The URI where the object will be stored.
            data: The bytes data to upload.
            chunk_size_bytes: The size of chunks to use for uploading.

        """

    @abc.abstractmethod
    def download_object_as_bytes(self, uri: StoragePrefix, chunk_size_bytes: int = DOWNLOAD_CHUNK_SIZE_BYTES) -> bytes:
        """Download an object as bytes from the specified storage path.

        Args:
            uri: The storage prefix of the object to download.
            chunk_size_bytes: The size of chunks to use for downloading.

        Returns:
            bytes: The object's content as bytes.

        """

    @abc.abstractmethod
    def download_objects_as_bytes(self, uris: list[StoragePrefix]) -> list[bytes]:
        """Download multiple objects as bytes from the specified URIs.

        Args:
            uris: A list of URIs of the objects to download.

        Returns:
            A list of bytes containing the object contents.

        """

    @abc.abstractmethod
    def list_recursive_directory(self, uri: StoragePrefix, limit: int = 0) -> list[StoragePrefix]:
        """List all objects recursively, starting from the given prefix.

        Args:
            uri: The storage prefix to list objects from.
            limit: Maximum number of objects to return.

        Returns:
            A list of storage prefixes for all objects found.

        """

    @abc.abstractmethod
    def list_recursive(self, prefix: StoragePrefix, limit: int = 0) -> list[dict[str, Any]]:
        """List all objects recursively, starting from the given prefix.

        Args:
            prefix: Storage prefix to list objects from.
            limit: Maximum number of objects to return.

        Returns:
            A list of dictionaries with object metadata.

        """

    @abc.abstractmethod
    def upload_file(
        self,
        local_path: str,
        remote_path: StoragePrefix,
        chunk_size: int = UPLOAD_CHUNK_SIZE_BYTES,
    ) -> None:
        """Upload a file to the specified path.

        Args:
            local_path: The local path of the file to upload.
            remote_path: The URI where the file will be uploaded.
            chunk_size: The size of chunks to use for uploading.

        Raises:
            ValueError: If the object already exists and overwriting is not allowed.

        """

    @abc.abstractmethod
    def sync_remote_to_local(
        self,
        remote_prefix: StoragePrefix,
        local_dir: pathlib.Path,
        *,
        delete: bool = False,
        chunk_size_bytes: int = DOWNLOAD_CHUNK_SIZE_BYTES,
    ) -> None:
        """Sync contents of a remote prefix with a local directory.

        Args:
            remote_prefix: The remote prefix to sync from.
            local_dir: The local directory path to sync to.
            delete: If True, delete local files that don't exist in the remote prefix.
            chunk_size_bytes: The size of chunks to use for downloading.

        """

    @abc.abstractmethod
    def make_background_uploader(self, chunk_size_bytes: int = UPLOAD_CHUNK_SIZE_BYTES) -> BackgroundUploader:
        """Create and return a BackgroundUploader instance.

        Args:
            chunk_size_bytes: The size of chunks to use for uploading.

        Returns:
            An initialized BackgroundUploader instance.

        """


def is_storage_path(path: str | None, protocol: str) -> bool:
    """Check if a path string is a storage path with the given protocol.

    Args:
        path: The path to check.
        protocol: The protocol to check for (e.g., "s3", "azure").

    Returns:
        bool: True if the path is a storage path with the given protocol, False otherwise.

    """
    if path is None:
        return False
    return path.startswith(f"{protocol}://")
