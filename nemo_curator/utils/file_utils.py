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
from functools import partial, reduce

import dask.bag as db
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed

from nemo_curator.utils.distributed_utils import single_partition_write_with_filename


def mkdir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def expand_outdir_and_mkdir(outdir):
    outdir = os.path.abspath(os.path.expanduser(outdir))
    mkdir(outdir)
    return outdir


def get_all_files_paths_under(root, recurse_subdirectories=True, followlinks=False):
    """
    This function returns a list of all the files under a specified directory.
    Args:
        root: The path to the directory to read.
        recurse_subdirecties: Whether to recurse into subdirectories.
                              Please note that this can be slow for large
                              number of files.
        followlinks: Whether to follow symbolic links.
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
    return file_ls


# Using this for restarting jobs
# can lead to problems when there is an error while
# writing a file we can use the offset counter approach
# in jaccard shuffle as a more robust way to restart jobs
def get_remaining_files(input_file_path, output_file_path, input_file_type):
    """
    This function returns a list of the files that still remain to be read.

    Args:
        input_file_path: The path of the input files.
        output_file_path: The path of the output files.
        input_file_type: The type of the input files.
    Returns:
        A list of files that still remain to be read.

    """
    if input_file_type == "pickle":
        return [input_file_path]
    completed_files = [
        os.path.basename(entry.path) for entry in os.scandir(output_file_path)
    ]
    completed_files = set(completed_files)
    input_files = [
        entry.path
        for entry in os.scandir(input_file_path)
        if os.path.basename(entry.path) not in completed_files
    ]
    input_files.sort()
    return input_files


def get_batched_files(
    input_file_path, output_file_path, input_file_type, batch_size=64
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
    df: pd.DataFrame, output_dir, metadata_field, remove_metadata, output_type
):
    counts = df[metadata_field].value_counts().to_dict()

    for meta_value in counts:
        meta_output_dir = expand_outdir_and_mkdir(os.path.join(output_dir, meta_value))
        meta_slice = df[df[metadata_field] == meta_value]
        if remove_metadata:
            meta_slice = meta_slice.drop(columns=[metadata_field])
        single_partition_write_with_filename(
            meta_slice, meta_output_dir, output_type=output_type
        )

    return counts


def merge_counts(first: dict, second: dict):
    for ngram, count in second.items():
        first[ngram] = first.get(ngram, 0) + count

    return first


def separate_by_metadata(
    df: dd.DataFrame,
    output_dir,
    metadata_field,
    remove_metadata=False,
    output_type="jsonl",
) -> dict:
    """
    Saves the dataframe to subfolders named after a metadata

    Args:
        df: The dataframe to write. Must have a filename column for the shard.
        output_dir: The base directory for which all metadata based subdirs will be created under
        metadata_field: The metadata field to split on
        remove_metadata: Whether to remove the metadata from the dataframe when saving it

    Returns:
        A delayed dictionary mapping each metadata to the count of entries with that metadata value.
    """
    delayed_data = df.to_delayed()
    delayed_counts = [
        delayed(write_dataframe_by_meta)(
            partition, output_dir, metadata_field, remove_metadata, output_type
        )
        for partition in delayed_data
    ]
    merged_counts = delayed(reduce)(merge_counts, delayed_counts)

    return merged_counts


def parse_str_of_num_bytes(s, return_str=False):
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
    """Worker function to write out the data to jsonl files"""

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
    input_dir, output_dir, output_file_size="100M", start_index=0, file_prefix=""
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
