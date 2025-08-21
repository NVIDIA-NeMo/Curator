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

import os
import tarfile
import urllib
import warnings
import zipfile

import fsspec
import wget
from fsspec.core import get_filesystem_class, split_protocol
from loguru import logger


def download_file(source_url: str, target_directory: str, verbose: bool = True) -> str:
    # make sure target_directory is an absolute path to avoid bugs when we change directories to download data later
    target_directory = os.path.abspath(target_directory)

    if verbose:
        logger.info(f"Trying to download data from {source_url} and save it in this directory: {target_directory}")
    filename = os.path.basename(urllib.parse.urlparse(source_url).path)
    target_filepath = os.path.join(target_directory, filename)

    if os.path.exists(target_filepath):
        if verbose:
            logger.info(f"Found file {target_filepath} => will not be attempting download from {source_url}")
    else:
        logger.info(f"Not found file {target_filepath}")
        original_dir = os.getcwd()  # record current working directory so can cd back to it
        os.chdir(target_directory)  # cd to target dir so that temporary download file will be saved in target dir

        wget.download(source_url, target_directory)

        # change back to original directory as the rest of the code may assume that we are in that directory
        os.chdir(original_dir)
        if verbose:
            logger.info("Download completed")

    return target_filepath


def extract_archive(archive_path: str, extract_path: str, force_extract: bool = False) -> str:
    logger.info(f"Attempting to extract all contents from tar file {archive_path} and save in {extract_path}")
    if not force_extract:
        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r") as archive:
                archive_extracted_dir = os.path.commonprefix(archive.getnames()[1:])
        elif zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive_extracted_dir = archive.namelist()[0]
        else:
            raise RuntimeError("Unknown archive format: " + archive_path + ". We only support tar and zip archives.")

        archive_contents_dir = os.path.join(extract_path, archive_extracted_dir)

    if not force_extract and os.path.exists(archive_contents_dir):
        logger.info(f"Directory {archive_contents_dir} already exists => will not attempt to extract file")
    else:
        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r") as archive:
                archive.extractall(path=extract_path, filter="data")
        elif zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(extract_path, filter="data")
        logger.info("Finished extracting")

    if force_extract:
        return None
    return archive_contents_dir

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
