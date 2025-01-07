# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import json
import os
import pathlib
import warnings
from functools import partial, reduce
from typing import List, Optional, Union

import dask.bag as db
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed

from nemo_curator.utils.distributed_utils import (
    read_data,
    single_partition_write_with_filename,
)

NEMO_CURATOR_HOME = os.environ.get(
    "NEMO_CURATOR_HOME", os.path.join(os.path.expanduser("~"), ".nemo_curator")
)


def mkdir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def expand_outdir_and_mkdir(outdir):
    outdir = os.path.abspath(os.path.expanduser(outdir))
    mkdir(outdir)
    return outdir


def filter_files_by_extension(
    files_list: List[str],
    keep_extensions: Union[str, List[str]],
) -> List[str]:
    """
    Given a list of files, filter it to only include files matching given extension(s).

    Args:
        files_list: List of files.
        keep_extensions: A string (e.g., "json") or a list of strings (e.g., ["json", "parquet"])
            representing which file types to keep from files_list.

    """
    filtered_files = []

    if isinstance(keep_extensions, str):
        keep_extensions = [keep_extensions]

    file_extensions = [s if s.startswith(".") else "." + s for s in keep_extensions]

    for file in files_list:
        if file.endswith(tuple(file_extensions)):
            filtered_files.append(file)

    if len(files_list) != len(filtered_files):
        warnings.warn(f"Skipped at least one file due to unmatched file extension(s).")

    return filtered_files


def get_all_files_paths_under(
    root: str,
    recurse_subdirectories: bool = True,
    followlinks: bool = False,
    keep_extensions: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    """
    This function returns a list of all the files under a specified directory.
    Args:
        root: The path to the directory to read.
        recurse_subdirecties: Whether to recurse into subdirectories.
                              Please note that this can be slow for large
                              number of files.
        followlinks: Whether to follow symbolic links.
        keep_extensions: A string or list of strings representing a file type
                   or multiple file types to include in the output, e.g.,
                   "jsonl" or ["jsonl", "parquet"].
    """
    if recurse_subdirectories:
        file_ls = [
            os.path.join(r, f)
            for r, subdirs, files in os.walk(root, followlinks=followlinks)
            for f in files
        ]
    else:
        file_ls = [entry.path for entry in os.scandir(root)]

    file_ls.sort()

    if keep_extensions is not None:
        file_ls = filter_files_by_extension(file_ls, keep_extensions)

    return file_ls


# Using this for restarting jobs
# can lead to problems when there is an error while
# writing a file we can use the offset counter approach
# in jaccard shuffle as a more robust way to restart jobs
def get_remaining_files(
    input_file_path: str,
    output_file_path: str,
    input_file_type: str,
    output_file_type: Optional[str] = None,
    num_files: int = -1,
):
    """
    This function returns a list of the files that still remain to be read.

    Args:
        input_file_path: The path of the input files.
        output_file_path: The path of the output files.
        input_file_type: The type of the input files.
        output_file_type: The type of the output files.
        num_files: The max number of files to be returned. If -1, all files are returned.
    Returns:
        A list of files that still remain to be read.

    """
    if input_file_type == "pickle":
        return [input_file_path]

    if not os.path.exists(output_file_path):
        expand_outdir_and_mkdir(output_file_path)
    completed_files = [
        os.path.basename(entry.path) for entry in os.scandir(output_file_path)
    ]
    completed_files = set(completed_files)

    input_files = [
        entry.path
        for entry in os.scandir(input_file_path)
        if os.path.basename(entry.path)
        not in _update_filetype(completed_files, output_file_type, input_file_type)
    ]
    # Guard against non extension files if present in the input directory
    input_files = [f for f in input_files if f.endswith(input_file_type)]
    input_files.sort()

    len_written_files = len(completed_files)
    if num_files > 0:
        left_to_sample = max(num_files - len_written_files, 0)
    else:
        left_to_sample = len(input_files)

    input_files = input_files[:left_to_sample]
    return input_files


def _update_filetype(file_set, old_file_type, new_file_type):
    if old_file_type is None or new_file_type is None:
        return file_set

    if not old_file_type.startswith("."):
        old_file_type = "." + old_file_type
    if not new_file_type.startswith("."):
        new_file_type = "." + new_file_type

    if old_file_type == new_file_type:
        return file_set

    updated_file_set = {
        (
            f"{os.path.splitext(file)[0]}{new_file_type}"
            if file.endswith(old_file_type)
            else file
        )
        for file in file_set
    }
    return updated_file_set


def get_batched_files(
    input_file_path: str,
    output_file_path: str,
    input_file_type: str,
    batch_size: int = 64,
):
    """
    This function returns a batch of files that still remain to be processed.

    Args:
        input_file_path: The path of the input files.
        output_file_path: The path of the output files.
        input_file_type: The type of the input files.
        batch_size: The number of files to be processed at once
    Returns:
        A batch of files that are not in the output directory.
    """
    remaining_files = get_remaining_files(
        input_file_path, output_file_path, input_file_type
    )
    for i in range(0, len(remaining_files), batch_size):
        yield remaining_files[i : i + batch_size]


def write_dataframe_by_meta(
    df: pd.DataFrame,
    output_dir: str,
    metadata_field: str,
    remove_metadata: bool = False,
    output_type: str = "jsonl",
    include_values: List[str] = None,
    exclude_values: List[str] = None,
    filename_col: str = "file_name",
):
    counts = df[metadata_field].value_counts().to_dict()

    # Apply include_values or value_exclesion_filter if provided
    if include_values is not None and include_values:
        counts = {k: v for k, v in counts.items() if k in include_values}
    elif exclude_values is not None and exclude_values:
        counts = {k: v for k, v in counts.items() if k not in exclude_values}

    for meta_value in counts:
        meta_output_dir = expand_outdir_and_mkdir(os.path.join(output_dir, meta_value))
        meta_slice = df[df[metadata_field] == meta_value]

        if remove_metadata:
            meta_slice = meta_slice.drop(columns=[metadata_field])
        single_partition_write_with_filename(
            meta_slice,
            meta_output_dir,
            output_type=output_type,
            filename_col=filename_col,
        )

    return counts


def merge_counts(first: dict, second: dict):
    for ngram, count in second.items():
        first[ngram] = first.get(ngram, 0) + count

    return first


def write_record(
    input_dir: str,
    file_name: str,
    line: str,
    field: str,
    output_dir: str,
    include_values: List[str] = None,
    exclude_values: List[str] = None,
):
    try:
        # Parse the JSON-encoded string 'line' into a Python dictionary
        line = json.loads(line)

        # Select category value
        category = line[field]

        if (exclude_values and category in exclude_values) or (
            include_values and category not in include_values
        ):
            return None

        # Obtain the relative path
        rel_path, file_name = os.path.split(
            os.path.relpath(file_name, start=os.path.abspath(input_dir))
        )

        output_dir = os.path.join(output_dir, category, rel_path)
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{file_name}", "a") as f:
            f.write(json.dumps(line) + "\n")

        return category
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def separate_by_metadata(
    input_data: Union[dd.DataFrame, str],
    output_dir: str,
    metadata_field: str,
    remove_metadata: bool = False,
    output_type: str = "jsonl",
    input_type: str = "jsonl",
    include_values: List[str] = None,
    exclude_values: List[str] = None,
    filename_col: str = "file_name",
) -> dict:
    """
    Saves the dataframe to subfolders named after a metadata

    Args:
        input_data: Either a DataFrame or a string representing the path to the input directory.
            If a DataFrame is provided, it must have a filename_col for the shard.
        output_dir: The base directory for which all metadata based subdirs will be created under
        metadata_field: The metadata field to split on
        remove_metadata: Whether to remove the metadata from the dataframe when saving it
        output_type: File type the dataset will be written to. Supported file formats include 'jsonl' (default),
            'pickle', or 'parquet'. (default: jsonl)
        include_values: A list of strings representing specific values to be selected or included.
            If provided, only the items matching these values should be kept.
        exclude_values: A list of strings representing specific values to be excluded or ignored.
            If provided, any items matching these values should be skipped.
        filename_col: The column name in the DataFrame that contains the filename. Default is "file_name".

    Returns:
        A delayed dictionary mapping each metadata to the count of entries with that metadata value.
    """

    if include_values is not None and exclude_values is not None:
        print("Error: 'include_values' and 'exclude_values' are mutually exclusive.")

        return

    # Create output_dir if needed
    if output_dir:
        output_dir = expand_outdir_and_mkdir(output_dir)

    if isinstance(input_data, str):
        print(f"Reading {input_type} files from {input_data}", flush=True)

        if input_type in ["json", "jsonl"] and output_type in ["json", "jsonl"]:
            # Read JSONL files with streaming (line-by-line), and include file path
            bag = db.read_text(
                os.path.join(input_data, "**", f"*.{input_type}"),
                include_path=True,
            )

            # Parse JSON lines and retain the file path
            bag = bag.map(
                lambda x: write_record(
                    input_dir=input_data,
                    file_name=x[1],
                    line=x[0],
                    field=metadata_field,
                    output_dir=output_dir,
                    include_values=include_values,
                    exclude_values=exclude_values,
                )
            )

            frequencies = dict(bag.frequencies().compute())
            frequencies.pop(None, None)  # Remove None when applying filters

            return delayed(reduce)(merge_counts, [frequencies])
        else:
            input_data = read_data(
                get_all_files_paths_under(input_data),
                file_type=input_type,
                backend="pandas",
                add_filename=filename_col,
            )
    delayed_counts = [
        delayed(write_dataframe_by_meta)(
            partition,
            output_dir,
            metadata_field,
            remove_metadata,
            output_type,
            include_values,
            exclude_values,
            filename_col,
        )
        for partition in input_data.to_delayed()
    ]

    return delayed(reduce)(merge_counts, delayed_counts)


def parse_str_of_num_bytes(s: str, return_str: bool = False) -> Union[str, int]:
    try:
        power = "kmg".find(s[-1].lower()) + 1
        size = float(s[:-1]) * 1024**power
    except ValueError:
        raise ValueError("Invalid size: {}".format(s))
    if return_str:
        return s
    else:
        return int(size)


def _save_jsonl(documents, output_path, start_index=0, max_index=10000, prefix=None):
    """
    Worker function to write out the data to jsonl files

    """

    def _encode_text(document):
        return document.strip().encode("utf-8")

    def _name(start_index, npad, prefix, i):
        tag = str(start_index + i).rjust(npad, "0")
        return f"{prefix}{tag}"

    # Create the naming function
    npad = int(np.log10(max_index) + 1)
    name = partial(_name, start_index, npad, prefix)

    output_glob_string = os.path.join(output_path, "*.jsonl")

    output_files = documents.map(_encode_text).to_textfiles(
        output_glob_string,
        name_function=name,
    )

    # Delete empty files generated due to empty partitions in the bag
    for output_file in output_files:
        try:
            if os.path.getsize(output_file) == 0:
                os.remove(output_file)
        except Exception as exception:
            print(
                f"An exception occurred when trying to delete {output_file}.\n{exception}",
                flush=True,
            )


def reshard_jsonl(
    input_dir: str,
    output_dir: str,
    output_file_size: str = "100M",
    start_index: int = 0,
    file_prefix: str = "",
):
    """
    Reshards a directory of jsonl files to have a new (approximate) file size for each shard

    Args:
        input_dir: The input directory containing jsonl files
        output_dir: The output directory where the resharded jsonl files will be written
        output_file_size: Approximate size of output files. Must specify with a string and
            with the unit K, M or G for kilo, mega or gigabytes
        start_index: Starting index for naming the output files. Note: The indices may not
            be continuous if the sharding process would output an empty file in its place
        file_prefix: Prefix to use to prepend to output file number
    """

    # Output file size in bytes
    blocksize = parse_str_of_num_bytes(output_file_size)

    input_files = list(get_all_files_paths_under(input_dir))

    # Read in the dask bag
    b = db.read_text(input_files, blocksize=blocksize)

    # Prepare the output
    output_dir = expand_outdir_and_mkdir(output_dir)

    # Save to balanced files
    _save_jsonl(b, output_dir, start_index=start_index, prefix=file_prefix)


def remove_path_extension(path: str):
    p = pathlib.Path(path)
    return os.path.join(p.parent, p.stem)
