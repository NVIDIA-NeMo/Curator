import os
import warnings
from pathlib import Path

import fsspec
import pyarrow.parquet as pq
from fsspec.core import get_filesystem_class, split_protocol
from fsspec.parquet import open_parquet_file
from loguru import logger


def get_fs(path: str, storage_options: dict[str, str] | None = None) -> fsspec.AbstractFileSystem:
    if not storage_options:
        storage_options = {}
    protocol, path = split_protocol(path)
    return get_filesystem_class(protocol)(**storage_options)


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
        warnings.warn("Skipped at least one file due to unmatched file extension(s).", stacklevel=2)

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


def remove_and_create_dir(
    dir_path: str, fs: fsspec.AbstractFileSystem | None = None, storage_options: dict | None = None
) -> None:
    if fs is None:
        fs = get_fs(dir_path, storage_options)

    if fs.exists(dir_path):
        logger.warning(f"Removing and recreating directory {dir_path}")
        fs.rm(dir_path, recursive=True)
    # if AWS then since mkdir doesn't exist we need to touch
    if fs.protocol == "s3":
        fs.touch(os.path.join(dir_path, ".empty"))
    else:
        fs.mkdir(dir_path)



def get_parquet_num_rows(
    file_path: str,
    storage_options: dict | None = None,
) -> int:
    """Get number of rows for local/cloud Parquet files efficiently by only reading the footer"""
    with open_parquet_file(file_path, storage_options=storage_options) as f:
        return pq.read_metadata(f).num_rows

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

