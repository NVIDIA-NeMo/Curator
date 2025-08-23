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

from pathlib import Path

import fsspec
from fsspec.core import get_filesystem_class, split_protocol
from loguru import logger

FILETYPE_TO_DEFAULT_EXTENSIONS = {
    "parquet": [".parquet"],
    "jsonl": [".jsonl", ".json"],
}


def get_fs(path: str, storage_options: dict[str, str] | None = None) -> fsspec.AbstractFileSystem:
    if not storage_options:
        storage_options = {}
    protocol, path = split_protocol(path)
    return get_filesystem_class(protocol)(**storage_options)


def is_not_empty(
    path: str, fs: fsspec.AbstractFileSystem | None = None, storage_options: dict[str, str] | None = None
) -> bool:
    if fs is None and storage_options is None:
        err_msg = "fs or storage_options must be provided"
        raise ValueError(err_msg)
    elif fs is not None and storage_options is not None:
        err_msg = "fs and storage_options cannot be provided together"
        raise ValueError(err_msg)
    elif fs is None:
        fs = get_fs(path, storage_options)

    return fs.exists(path) and fs.isdir(path) and fs.listdir(path)


def delete_dir(
    path: str, fs: fsspec.AbstractFileSystem | None = None, storage_options: dict[str, str] | None = None
) -> None:
    if fs is None and storage_options is None:
        err_msg = "fs or storage_options must be provided"
        raise ValueError(err_msg)
    elif fs is not None and storage_options is not None:
        err_msg = "fs and storage_options cannot be provided together"
        raise ValueError(err_msg)
    elif fs is None:
        fs = get_fs(path, storage_options)

    if fs.exists(path) and fs.isdir(path):
        fs.rm(path, recursive=True)


def create_or_overwrite_dir(
    path: str, fs: fsspec.AbstractFileSystem | None = None, storage_options: dict[str, str] | None = None
) -> None:
    """
    Creates a directory if it does not exist and overwrites it if it does.
    Warning: This function will delete all files in the directory if it exists.
    """
    if fs is None and storage_options is None:
        err_msg = "fs or storage_options must be provided"
        raise ValueError(err_msg)
    elif fs is not None and storage_options is not None:
        err_msg = "fs and storage_options cannot be provided together"
        raise ValueError(err_msg)
    elif fs is None:
        fs = get_fs(path, storage_options)

    if is_not_empty(path, fs):
        logger.warning(f"Output directory {path} is not empty. Deleting it.")
        delete_dir(path, fs)

    fs.mkdirs(path, exist_ok=True)


def filter_files_by_extension(
    files_list: list[str],
    keep_extensions: str | list[str],
) -> list[str]:
    filtered_files = []
    if isinstance(keep_extensions, str):
        keep_extensions = [keep_extensions]

    # Ensure that the extensions are prefixed with a dot
    file_extensions = tuple([s if s.startswith(".") else f".{s}" for s in keep_extensions])

    for file in files_list:
        if file.endswith(file_extensions):
            filtered_files.append(file)

    if len(files_list) != len(filtered_files):
        logger.warning("Skipped at least one file due to unmatched file extension(s).")

    return filtered_files


def get_all_files_paths_under(
    input_dir: str,
    recurse_subdirectories: bool = False,
    keep_extensions: str | list[str] | None = None,
    storage_options: dict | None = None,
    fs: fsspec.AbstractFileSystem | None = None,
) -> list[str]:
    # TODO: update with a more robust fsspec method
    if fs is None:
        fs = get_fs(input_dir, storage_options)

    file_ls = fs.find(input_dir, maxdepth=None if recurse_subdirectories else 1)
    if "://" in input_dir:
        protocol = input_dir.split("://")[0]
        file_ls = [f"{protocol}://{f}" for f in file_ls]

    file_ls.sort()
    if keep_extensions is not None:
        file_ls = filter_files_by_extension(file_ls, keep_extensions)
    return file_ls


def infer_dataset_name_from_path(path: str) -> str:
    """Infer a dataset name from a path, handling both local and cloud storage paths.
    Args:
        path: Local path or cloud storage URL (e.g. s3://, abfs://)
    Returns:
        Inferred dataset name from the path
    """
    # Split protocol and path for cloud storage
    protocol, pure_path = split_protocol(path)
    if protocol is None:
        # Local path handling
        first_file = Path(path)
        if first_file.parent.name and first_file.parent.name != ".":
            return first_file.parent.name.lower()
        return first_file.stem.lower()
    else:
        path_parts = pure_path.rstrip("/").split("/")
        if len(path_parts) <= 1:
            return path_parts[0]
        return path_parts[-1].lower()


def check_disallowed_kwargs(
    kwargs: dict,
    disallowed_keys: list[str],
    raise_error: bool = True,
) -> None:
    """Check if any of the disallowed keys are in provided kwargs
    Used for read/write kwargs in stages.
    Args:
        kwargs: The dictionary to check
        disallowed_keys: The keys that are not allowed.
        raise_error: Whether to raise an error if any of the disallowed keys are in the kwargs.
    Raises:
        ValueError: If any of the disallowed keys are in the kwargs and raise_error is True.
        Warning: If any of the disallowed keys are in the kwargs and raise_error is False.
    Returns:
        None
    """
    found_keys = set(kwargs).intersection(disallowed_keys)
    if raise_error and found_keys:
        msg = f"Unsupported keys in kwargs: {', '.join(found_keys)}"
        raise ValueError(msg)
    elif found_keys:
        msg = f"Unsupported keys in kwargs: {', '.join(found_keys)}"
        logger.warning(msg)
