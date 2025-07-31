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
"""S3 client implementation using boto3.

This module provides a simple S3 client with common operations for interacting
with S3-compatible object storage systems, including chunked downloads and uploads.
"""

from __future__ import annotations

import configparser
import io
import os
import pathlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import attrs
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from loguru import logger
from tqdm import tqdm

from ray_curator.utils.storage_client import (
    DOWNLOAD_CHUNK_SIZE_BYTES,
    UPLOAD_CHUNK_SIZE_BYTES,
    BackgroundUploader,
    BaseClientConfig,
    StorageClient,
    StoragePrefix,
    is_storage_path,
)

S3_PROFILE_PATH = pathlib.Path(os.getenv("CURATOR_S3_PROFILE_PATH", "/dev/shm/s3_creds_file")) 

@attrs.define
class S3ClientConfig(BaseClientConfig):
    """Configuration class for S3 client.

    Attributes:
        aws_access_key_id (str): AWS access key ID. Optional if AWS config file will be parsed directly by boto
        aws_secret_access_key (str): AWS secret access key. Optional if AWS config file will be parsed directly by boto
        endpoint_url (str): S3 endpoint URL.
        region (str): AWS region (default: "").
        aws_session_token (str | None): AWS session token (default: None).

    """

    # profile related
    aws_access_key_id: str | None = attrs.field(default=None)
    aws_secret_access_key: str | None = attrs.field(default=None)
    aws_session_token: str | None = attrs.field(default=None)
    endpoint_url: str | None = attrs.field(default=None)
    region: str | None = attrs.field(default=None)


@attrs.define
class S3Prefix(StoragePrefix):
    """Represents an S3 prefix (bucket and key).

    Attributes:
        _input (str): The input S3 path.

    Properties:
        bucket (str): The S3 bucket name.
        prefix (str): The S3 key prefix.
        path (str): The full S3 path (s3://bucket/prefix).

    """

    def __attrs_post_init__(self) -> None:
        """Post init."""
        # Remove 's3://' prefix if present
        self._input = self._input.removeprefix("s3://")

        # Split into bucket and key
        parts = self._input.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        # Validate bucket name: 3-63 chars, lowercase letters, numbers, dots, hyphens, start/end alphanumeric
        if not re.match(r"^[a-z0-9][a-z0-9\-.]{1,61}[a-z0-9]$", bucket):
            error_msg = f"Invalid S3 bucket name: {bucket}"
            raise ValueError(error_msg)

        # Validate object key characters and length: allow letters, digits, dot, hyphen, underscore, slash, space;
        # max 1024 chars
        if key and not re.match(r"^[A-Za-z0-9.\-_/ ,]{1,1024}$", key):
            error_msg = f"Invalid S3 object key: {key}"
            raise ValueError(error_msg)

    @property
    def bucket(self) -> str:
        """Return the bucket name for this S3 prefix.

        Returns:
            The bucket name.

        """
        return self._input.split("/", 1)[0]

    @property
    def path(self) -> str:
        """Return the full S3 path for this S3 prefix.

        Returns:
            The full S3 path.

        """
        return f"s3://{self.bucket}/{self.prefix}"


class S3Client(StorageClient):
    """S3 client for interacting with S3-compatible object storage systems.

    This class provides methods for common S3 operations such as checking object
    existence, uploading and downloading objects, and listing objects in a bucket.

    Attributes:
        session (boto3.Session): The boto3 session.
        s3 (boto3.client): The boto3 S3 client.
        can_overwrite (bool): Whether the client can overwrite existing objects.
        can_delete (bool): Whether the client can delete objects.

    """

    def __init__(self, config: S3ClientConfig) -> None:
        """Initialize the S3 client with the given configuration.

        Args:
            config (S3ClientConfig): Configuration object containing AWS credentials and settings.

        """
        boto_config = BotoConfig(
            max_pool_connections=config.max_concurrent_threads,
            retries={"max_attempts": 3},
            connect_timeout=config.operation_timeout_s,
            read_timeout=config.operation_timeout_s,
        )
        # If creds are set, specify them
        if config.aws_access_key_id is not None:
            self.session = boto3.Session(
                aws_access_key_id=config.aws_access_key_id,
                aws_secret_access_key=config.aws_secret_access_key,
                aws_session_token=config.aws_session_token,
            )
            self.s3 = self.session.client("s3", endpoint_url=config.endpoint_url, config=boto_config)
        # If omitted, rely on boto3 intrinsic parsing of AWS_CONFIG_FILE
        else:
            logger.warning("This should not happen in current implementation")
            self.session = boto3.Session()
            self.s3 = self.session.client("s3", config=boto_config)

        self.can_overwrite = config.can_overwrite
        self.can_delete = config.can_delete

    def object_exists(self, dest: StoragePrefix) -> bool:
        """Check if an object exists at the specified S3 URI.

        Args:
            dest (S3Prefix): The S3 prefix of the object to check.

        Returns:
            bool: True if the object exists, False otherwise.

        """
        assert isinstance(dest, S3Prefix)
        try:
            self.s3.head_object(Bucket=dest.bucket, Key=dest.prefix)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            # If it's not a 404 error, re-raise the exception
            raise
        else:
            return True

    def upload_bytes(self, dest: StoragePrefix, data: bytes) -> None:
        """Upload bytes data to the specified S3 prefix.

        Args:
            dest (S3Prefix): The S3 prefix where the object will be stored.
            data (bytes): The bytes data to upload.

        Raises:
            ValueError: If the object already exists and overwriting is not allowed.

        """
        assert isinstance(dest, S3Prefix)
        if not self.can_overwrite and self.object_exists(dest):
            error_msg = f"Object {dest.prefix} already exists in bucket {dest.bucket} and overwriting is not allowed."
            raise ValueError(error_msg)

        fileobj = io.BytesIO(data)
        self.s3.upload_fileobj(
            fileobj,
            dest.bucket,
            dest.prefix,
            Config=TransferConfig(multipart_threshold=UPLOAD_CHUNK_SIZE_BYTES, max_concurrency=10),
        )

    def upload_bytes_uri(self, uri: str, data: bytes, chunk_size_bytes: int = UPLOAD_CHUNK_SIZE_BYTES) -> None:  # noqa: ARG002
        """Upload bytes data to the specified S3 URI.

        Args:
            uri: The S3 URI where the object will be stored (e.g., 's3://bucket-name/key').
            data: The bytes data to upload.
            chunk_size_bytes: unused

        """
        self.upload_bytes(S3Prefix(uri), data)

    def download_object_as_bytes(self, uri: StoragePrefix, chunk_size_bytes: int = DOWNLOAD_CHUNK_SIZE_BYTES) -> bytes:
        """Download an object as bytes from the specified S3 prefix.

        Args:
            uri (S3Prefix): The S3 prefix of the object to download.
            chunk_size_bytes (int): The size of chunks to use for downloading.

        Returns:
            bytes: The object's content as bytes.

        """
        assert isinstance(uri, S3Prefix)
        fileobj = io.BytesIO()
        self.s3.download_fileobj(
            uri.bucket,
            uri.prefix,
            fileobj,
            Config=TransferConfig(multipart_threshold=chunk_size_bytes, max_concurrency=10),
        )
        return fileobj.getvalue()

    def download_objects_as_bytes(self, uris: list[StoragePrefix]) -> list[bytes]:
        """Download multiple objects as bytes from the specified URIs.

        Args:
            uris: A list of S3 URIs of the objects to download.

        Returns:
            A list of bytes containing the object contents.

        """
        return [self.download_object_as_bytes(uri) for uri in uris]

    def list_recursive_directory(self, uri: StoragePrefix, limit: int = 0) -> list[StoragePrefix]:
        """List all objects in a bucket recursively, starting from the given prefix.

        Args:
            uri (S3Prefix): The S3 prefix to list objects from.
            limit (int): Limit of list to be returned

        Returns:
            List[S3Prefix]: A list of S3 prefixes for all objects found.

        """
        assert isinstance(uri, S3Prefix)
        objects = self.list_recursive(uri, limit)
        results: list[StoragePrefix] = []
        for obj in objects:
            path = f"s3://{uri.bucket}/{obj['Key']}"
            results.append(S3Prefix(path))
        return results

    def list_recursive(self, s3_prefix: StoragePrefix, limit: int = 0) -> list[dict[str, Any]]:
        """List all objects in a bucket recursively, starting from the given prefix.

        Args:
            s3_prefix: The S3 prefix to list objects from.
            limit (int): Limit of list to be returned

        Returns:
            A list of dictionaries with object metadata.

        """
        paginator = self.s3.get_paginator("list_objects_v2")
        objects = []
        assert isinstance(s3_prefix, S3Prefix)
        for page in paginator.paginate(Bucket=s3_prefix.bucket, Prefix=s3_prefix.prefix):
            if "Contents" in page:
                objects.extend(page["Contents"])
            if limit > 0 and len(objects) >= limit:
                logger.info(f"Truncated list of objects in S3 prefix to {len(objects)} as limit={limit}")
                break
        return objects

    def upload_file(
        self,
        local_path: str,
        remote_path: StoragePrefix,
        chunk_size: int = UPLOAD_CHUNK_SIZE_BYTES,
    ) -> None:
        """Upload a file to the specified S3 path.

        Args:
            local_path: The local path of the file to upload.
            remote_path: The S3 URI where the file will be uploaded (e.g., 's3://bucket-name/key').
            chunk_size: The size of chunks to use for uploading (default: UPLOAD_CHUNK_SIZE_BYTES).

        Raises:
            ValueError: If the object already exists and overwriting is not allowed.

        """
        assert isinstance(remote_path, S3Prefix)
        if not self.can_overwrite and self.object_exists(remote_path):
            error_msg = f"Object {remote_path.path} already exists and overwriting is not allowed."
            raise ValueError(error_msg)

        logger.info(f"Uploading {local_path} to {remote_path}")
        self.s3.upload_file(
            local_path,
            remote_path.bucket,
            remote_path.prefix,
            Config=TransferConfig(multipart_threshold=chunk_size, max_concurrency=10),
        )
        logger.info(f"Upload complete: {remote_path}")

    def sync_remote_to_local(
        self,
        s3_prefix: StoragePrefix,
        local_dir: pathlib.Path,
        *,
        delete: bool = False,
        chunk_size_bytes: int = DOWNLOAD_CHUNK_SIZE_BYTES,
    ) -> None:
        """Sync contents of an S3 prefix with a local directory.

        Args:
            s3_prefix (S3Prefix): The S3 prefix to sync from.
            local_dir (pathlib.Path): The local directory path to sync to.
            delete (bool): If True, delete local files that don't exist in the S3 prefix.
            chunk_size_bytes (int): The size of chunks to use for downloading.

        """
        print(f"Syncing {s3_prefix} to {local_dir}")  # noqa: T201
        local_dir_path = Path(local_dir)
        local_dir_path.mkdir(parents=True, exist_ok=True)

        assert isinstance(s3_prefix, S3Prefix)
        # List all objects in the S3 prefix
        s3_objects = self.list_recursive(s3_prefix)

        # Download or update files
        with tqdm(
            total=len(s3_objects),
            desc="Syncing",
            unit="file",
            ncols=70,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        ) as pbar:
            for obj in s3_objects:
                relative_path = obj["Key"][len(s3_prefix.prefix) :].lstrip("/")
                local_file_path = local_dir_path / relative_path
                local_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Check if local file exists and has the same size and last modified time
                s3_mtime = (
                    obj["LastModified"].timestamp()
                    if isinstance(obj["LastModified"], datetime)
                    else datetime.strptime(obj["LastModified"], "%Y-%m-%dT%H:%M:%S.%fZ %z").timestamp()
                )
                if local_file_path.exists():
                    local_mtime = local_file_path.stat().st_mtime
                    if local_mtime == s3_mtime and local_file_path.stat().st_size == obj["Size"]:
                        pbar.update(1)
                        continue

                self.s3.download_file(
                    s3_prefix.bucket,
                    obj["Key"],
                    str(local_file_path),
                    Config=TransferConfig(multipart_threshold=chunk_size_bytes, max_concurrency=10),
                )
                # Set the local file's modification time to match the S3 object
                os.utime(local_file_path, (s3_mtime, s3_mtime))
                pbar.update(1)

        if delete:
            # Remove local files that don't exist in S3
            local_files = list(local_dir_path.rglob("*"))
            with tqdm(
                total=len(local_files),
                desc="Cleaning",
                unit="file",
                ncols=70,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            ) as pbar:
                for local_file in local_files:
                    if local_file.is_file():
                        relative_path = local_file.relative_to(local_dir_path)
                        s3_key = f"{s3_prefix.prefix}{relative_path}".rstrip("/")
                        if not any(obj["Key"] == s3_key for obj in s3_objects):
                            local_file.unlink()
                    pbar.update(1)

        logger.info(f"\nSync completed: {s3_prefix} -> {local_dir}")

    def make_background_uploader(self, chunk_size_bytes: int = UPLOAD_CHUNK_SIZE_BYTES) -> S3BackgroundUploader:
        """Create and return a BackgroundUploader instance.

        Args:
            chunk_size_bytes: The size of chunks to use for uploading (default: UPLOAD_CHUNK_SIZE_BYTES).

        Returns:
            An initialized BackgroundUploader instance.

        """
        return S3BackgroundUploader(self, chunk_size_bytes)


class S3BackgroundUploader(BackgroundUploader):
    """Handles background uploads to S3."""

    def __init__(self, client: S3Client, chunk_size_bytes: int) -> None:
        """Initialize the BackgroundUploader with the given S3 client and chunk size.

        Args:
            client: The S3 client instance.
            chunk_size_bytes: The size of chunks to use for uploading.

        """
        super().__init__(client, chunk_size_bytes)

    def add_task_file(self, local_path: pathlib.Path, remote_path: str) -> None:
        """Add a file upload task to the background uploader.

        Args:
            local_path: Path to the local file to upload.
            remote_path: Path in the storage system where the file will be uploaded.

        """
        future = self.executor.submit(self._upload_file, local_path, remote_path)
        self.futures.append(future)

    def _upload_file(self, local_path: pathlib.Path, remote_path: str) -> None:
        """Upload a file to S3.

        Args:
            local_path: Path to the local file to upload.
            remote_path: S3 path where the file will be uploaded.

        """
        remote_prefix = S3Prefix(remote_path)
        self.client.upload_file(str(local_path), remote_prefix, self.chunk_size_bytes)  # type: ignore[attr-defined]


def _make_s3_client_config(
    profile_path: pathlib.Path,
    profile_name: str = "default",
    *,
    can_overwrite: bool = False,
    can_delete: bool = False,
) -> S3ClientConfig:
    """Create and return an S3 client configuration from a profile file.

    Args:
        profile_path (pathlib.Path): Path to the S3 profile file.
        profile_name (str): The name of the S3 profile to use (default: "default").
        can_overwrite (bool): Whether the client can overwrite existing objects.
        can_delete (bool): Whether the client can delete objects.

    Returns:
        S3ClientConfig: An S3ClientConfig instance.

    Raises:
        FileNotFoundError: If the profile file does not exist.
        ValueError: If the specified profile is not found in the config file.

    """
    config = configparser.ConfigParser()
    if not profile_path.exists():
        error_msg = f"S3 profile file {profile_path} does not exist."
        raise FileNotFoundError(error_msg)
    config.read(profile_path)

    # find the target profile section
    profile_key = None
    item_len: int = 2
    for section in config.sections():
        if section == profile_name:
            profile_key = section
            break
        if section.startswith("profile "):
            items = section.split()
            if len(items) == item_len and items[1] == profile_name:
                profile_key = section
                c = config[profile_key]
                break

    if profile_key is None:
        error_msg = f"Profile {profile_name} not found in config file {profile_path}"
        raise ValueError(error_msg)
    c = config[profile_key]

    return S3ClientConfig(
        aws_access_key_id=c.get("aws_access_key_id"),
        aws_secret_access_key=c.get("aws_secret_access_key"),
        aws_session_token=c.get("aws_session_token", None),
        endpoint_url=c.get("endpoint_url", None),
        region=c.get("region", None),
        can_overwrite=can_overwrite,
        can_delete=can_delete,
    )


def get_s3_client_config(
    profile_name: str = "default",
    *,
    can_overwrite: bool = False,
    can_delete: bool = False,
) -> S3ClientConfig:
    """Create and return an S3ClientConfig instance.

    Args:
        profile_name (str): The name of the S3 profile to use (default: "default").
        can_overwrite (bool): Whether the client is allowed to overwrite existing objects.
        can_delete (bool): Whether the client is allowed to delete objects.

    Returns:
        S3ClientConfig: An initialized S3ClientConfig instance.

    """
    if S3_PROFILE_PATH.exists():
        # first try s3 profile
        return _make_s3_client_config(
            S3_PROFILE_PATH,
            profile_name,
            can_overwrite=can_overwrite,
            can_delete=can_delete,
        )
    error_msg = f"Not found S3 creds from {S3_PROFILE_PATH}"
    raise ValueError(error_msg)


def is_s3path(path: str | None) -> bool:
    """Check if a path string is an S3 path.

    Args:
        path: The path to check.

    Returns:
        bool: True if the path is an S3 path, False otherwise.

    """
    return is_storage_path(path, "s3")


def create_s3_client(
    target_path: str | None = None,
    profile_name: str = "default",
    *,
    can_overwrite: bool = False,
    can_delete: bool = False,
) -> S3Client | None:
    """Create and return an S3Client instance if the target path is an S3 path.

    Args:
        target_path: The target path to check. If it's an S3 path, an S3Client is created.
        profile_name: The name of the S3 profile to use (default: "default").
        can_overwrite: Whether the client is allowed to overwrite existing objects.
        can_delete: Whether the client is allowed to delete objects.

    Returns:
        An initialized S3Client instance if the target path is an S3 path, None otherwise.

    """
    if is_s3path(target_path):
        return S3Client(get_s3_client_config(profile_name, can_overwrite=can_overwrite, can_delete=can_delete))
    return None
