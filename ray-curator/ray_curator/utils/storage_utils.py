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

"""Storage Utilities."""

import json
import pathlib
import re
from typing import Any

from loguru import logger

from ray_curator.utils import s3_client
from ray_curator.utils.retry_utils import do_with_retries
from ray_curator.utils.s3_client import S3Prefix, is_s3path
from ray_curator.utils.storage_client import StorageClient, StoragePrefix


def is_remote_path(path: str | None) -> bool:
    """Check if a path is on a remote storage system (S3)."""
    if path is None:
        return False
    return is_s3path(path)


def get_storage_client(
    target_path: str,
    *,
    profile_name: str = "default",
    can_overwrite: bool = False,
    can_delete: bool = False,
) -> StorageClient | None:
    """Get the appropriate storage client for a path (S3).

    Args:
        target_path: The path to get a client for.
        profile_name: The profile name to use.
        can_overwrite: Whether the client can overwrite existing objects.
        can_delete: Whether the client can delete objects.

    Returns:
        A storage client instance if the path is a remote path, None otherwise.

    """
    if is_s3path(target_path):
        return s3_client.create_s3_client(
            target_path,
            profile_name,
            can_overwrite=can_overwrite,
            can_delete=can_delete,
        )
    return None


def path_to_prefix(path: str) -> StoragePrefix:
    """Convert a path string to the appropriate storage prefix object.

    Args:
        path: The path to convert.

    Returns:
        A StoragePrefix instance for the path.

    Raises:
        ValueError: If the path is not a valid remote storage path.

    """
    if is_s3path(path):
        return S3Prefix(path)
    error_msg = f"Path {path} is not a valid remote storage path"
    raise ValueError(error_msg)


def read_bytes(
    filepath: StoragePrefix | pathlib.Path | str,
    client: StorageClient | None = None,
) -> bytes:
    """Read bytes from a file, whether local or on a remote storage system.

    Args:
        filepath: The path to the file to read.
        client: The storage client to use for remote paths.

    Returns:
        The file contents as bytes.

    """
    # Convert string paths to the appropriate type
    if isinstance(filepath, str):
        filepath = path_to_prefix(filepath) if is_remote_path(filepath) else pathlib.Path(filepath)

    # Handle remote storage paths
    if isinstance(filepath, StoragePrefix):
        if client is None:
            client = get_storage_client(str(filepath))

        def func_to_call() -> bytes:
            assert client is not None
            return client.download_object_as_bytes(filepath)

        return do_with_retries(func_to_call, backoff_factor=4.0, max_wait_time_s=256.0)
    # Handle local paths
    with filepath.open("rb") as fp:
        return fp.read()


def read_text(
    filepath: StoragePrefix | pathlib.Path | str,
    client: StorageClient | None = None,
) -> str:
    """Read text from a file, whether local or on a remote storage system.

    Args:
        filepath: The path to the file to read.
        client: The storage client to use for remote paths.

    Returns:
        The file contents as a string.

    """
    return read_bytes(filepath, client).decode("utf-8")


def read_json_file(
    filepath: StoragePrefix | pathlib.Path | str,
    client: StorageClient | None = None,
) -> dict[Any, Any]:
    """Read a JSON file, whether local or on a remote storage system.

    Args:
        filepath: The path to the file to read.
        client: The storage client to use for remote paths.

    Returns:
        The parsed JSON content.

    """
    # Convert string paths to the appropriate type
    if isinstance(filepath, str):
        filepath = path_to_prefix(filepath) if is_remote_path(filepath) else pathlib.Path(filepath)

    if isinstance(filepath, StoragePrefix):
        buffer = read_bytes(filepath, client)
        return json.loads(buffer.decode("utf-8"))  # type: ignore[no-any-return]
    with filepath.open() as fp:
        return json.load(fp)  # type: ignore[no-any-return]


def path_exists(
    path: str | StoragePrefix | pathlib.Path,
    client: StorageClient | None = None,
) -> bool:
    """Check if a path exists, whether local or on a remote storage system.

    Args:
        path: The path to check.
        client: The storage client to use for remote paths.

    Returns:
        True if the path exists, False otherwise.

    """
    # Convert string paths to the appropriate type
    if isinstance(path, str):
        path = path_to_prefix(path) if is_remote_path(path) else pathlib.Path(path)

    if isinstance(path, StoragePrefix):
        if client is None:
            client = get_storage_client(str(path))
        assert client is not None
        return client.object_exists(path)
    return pathlib.Path(path).exists()


def verify_path(path: str, level: int = 0) -> None:
    """Verify that a path exists.

    Args:
        path: The path to verify.
        level: For local paths, how many levels up to check.

    Raises:
        FileNotFoundError: If the path does not exist.

    """
    if not is_remote_path(path):
        path_to_check = pathlib.Path(path)
        while level > 0:
            path_to_check = path_to_check.parent
            level -= 1
        if not pathlib.Path(path_to_check).exists():
            error_msg = f"Local path {path_to_check} does not exist."
            raise FileNotFoundError(error_msg)


def create_path(path: str) -> None:
    """Create a path if it does not exist.

    Args:
        path: The path to create.

    """
    if not is_remote_path(path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def is_path_nested(path1: str, path2: str) -> bool:
    """Check if one path is nested inside another.

    Args:
        path1: The first path.
        path2: The second path.

    Returns:
        True if one path is nested inside the other, False otherwise.

    """
    _path1 = path1.rstrip("/") + "/"
    _path2 = path2.rstrip("/") + "/"
    return _path1.startswith(_path2) or _path2.startswith(_path1)


def get_full_path(path: str | StoragePrefix | pathlib.Path, *args: str) -> StoragePrefix | pathlib.Path:
    """Construct a full path from a base path and additional components.

    Args:
        path: The base path.
        *args: Additional path components.

    Returns:
        The full path as a StoragePrefix or Path object.

    """
    # Convert string paths to the appropriate type
    if isinstance(path, str):
        if is_remote_path(path):
            if is_s3path(path):
                return _get_s3_prefix(path, *args)
        else:
            return _get_local_path(pathlib.Path(path), *args)
    elif isinstance(path, StoragePrefix):
        if isinstance(path, S3Prefix):
            return _get_s3_prefix(path, *args)
    assert isinstance(path, pathlib.Path)
    return _get_local_path(path, *args)


def _get_s3_prefix(s3path: str | S3Prefix, *args: str) -> S3Prefix:
    """Construct a full S3 path from a base path and additional components.

    Args:
        s3path: The base S3 path.
        *args: Additional path components.

    Returns:
        The full S3 path as an S3Prefix object.

    """
    prefix = [str(s3path).rstrip("/")]
    if args:
        prefix += [x.strip("/") for x in args]
    return S3Prefix("/".join(prefix))


def _get_local_path(localpath: pathlib.Path, *args: str) -> pathlib.Path:
    """Construct a full local path from a base path and additional components.

    Args:
        localpath: The base local path.
        *args: Additional path components.

    Returns:
        The full local path as a Path object.

    """
    return pathlib.Path(localpath, *args)


def _get_objects_from_storage_prefix_relative(
    prefix: StoragePrefix,
    client: StorageClient,
    limit: int = 0,
) -> list[str]:
    """Get a list of objects from a storage prefix, with paths relative to the prefix.

    Args:
        prefix: The storage prefix to list objects from.
        client: The storage client to use.
        limit: Max numbers to iterate

    Returns:
        A list of object paths relative to the prefix.

    """

    def func_to_call() -> list[StoragePrefix]:
        return client.list_recursive_directory(prefix, limit)

    objects = do_with_retries(func_to_call)
    prefix_path = str(prefix).rstrip("/") + "/"
    objects_relative = []
    for x in objects:
        if not str(x).startswith(prefix_path):
            continue
        if str(x).endswith("/"):
            logger.warning(f"Skipping empty prefix {x}")
            continue
        objects_relative.append(str(x).replace(prefix_path, ""))
    return sorted(objects_relative)


def _get_files_from_localpath_relative(localpath: pathlib.Path) -> list[str]:
    """Get a list of files from a local path, with paths relative to the local path.

    Args:
        localpath: The local path to list files from.

    Returns:
        A list of file paths relative to the local path.

    """
    return sorted([str(x.relative_to(localpath)) for x in localpath.rglob("*") if x.is_file()])


def get_files_relative(path: str, client: StorageClient | None = None, limit: int = 0) -> list[str]:
    """Get a list of files from a path, with paths relative to the base path.

    Args:
        path: The path to list files from.
        client: The storage client to use for remote paths.
        limit: Max numbers to iterate

    Returns:
        A list of file paths relative to the base path.

    """
    if is_remote_path(path):
        prefix = path_to_prefix(path)
        if client is None:
            client = get_storage_client(path)
        assert client is not None
        return _get_objects_from_storage_prefix_relative(prefix, client, limit)
    return _get_files_from_localpath_relative(pathlib.Path(path))


def get_directories_relative(path: str, client: StorageClient | None = None) -> list[str]:
    """Get a list of top-level directories from a path.

    Args:
        path: The path to list directories from.
        client: The storage client to use for remote paths.

    Returns:
        A list of directory names at the top level of the path.

    """
    objects = get_files_relative(path, client)
    dirs = set()
    for obj in objects:
        dirs.add(obj.split("/")[0])
    return sorted(dirs)


def get_next_file(
    filename: str,
    filetype: str,
    output_path: str,
    client: StorageClient | None = None,
) -> StoragePrefix | pathlib.Path:
    """Get the path for the next file in a sequence.

    Args:
        filename: The base filename.
        filetype: The file extension.
        output_path: The output directory.
        client: The storage client to use for remote paths.

    Returns:
        The path for the next file in the sequence.

    """
    next_index = 0
    existing_files = get_files_relative(output_path, client)
    for item in existing_files:
        match = re.match(rf"{filename}_(\d+)\.{filetype}", item)
        if match:
            next_index = max(next_index, int(match.group(1)) + 1)
    next_file_name = f"{filename}_{next_index}.{filetype}"
    return get_full_path(output_path, next_file_name)


def backup_file(  # noqa: C901
    dest: StoragePrefix | pathlib.Path | str,
    client: StorageClient | None = None,
) -> None:
    """Create a backup of a file.

    Args:
        dest: The path to the file to back up.
        client: The storage client to use for remote paths.

    """
    # Convert string paths to the appropriate type
    if isinstance(dest, str):
        dest = path_to_prefix(dest) if is_remote_path(dest) else pathlib.Path(dest)

    # Handle remote storage paths
    if isinstance(dest, StoragePrefix):
        if client is None:
            client = get_storage_client(str(dest))

        assert client is not None
        if client.object_exists(dest):

            def func_to_call() -> None:
                idx = 1
                data = client.download_object_as_bytes(dest)
                bak_dest: StoragePrefix | None = None
                while True:
                    if isinstance(dest, S3Prefix):
                        bak_dest = S3Prefix(str(dest) + f".bak{idx}")
                    assert bak_dest is not None
                    if not client.object_exists(bak_dest):
                        break
                    idx += 1
                client.upload_bytes(bak_dest, data)

            do_with_retries(func_to_call)
    # Handle local paths
    elif dest.exists():
        idx = 1
        while True:
            bak_dest = dest.with_suffix(f"{dest.suffix}.bak{idx}")
            if not bak_dest.exists():
                break
            idx += 1
        dest.rename(bak_dest)
