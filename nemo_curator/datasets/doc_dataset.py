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

import dask.dataframe as dd

from nemo_curator.utils.distributed_utils import read_data, write_to_disk
from nemo_curator.utils.file_utils import get_all_files_paths_under


class DocumentDataset:
    """
    A collection of documents and document metadata.
    Internally it may be distributed across multiple nodes, and may be on GPUs.
    """

    def __init__(self, dataset_df: dd.DataFrame):
        self.df = dataset_df

    def __len__(self):
        return len(self.df)

    def persist(self):
        return DocumentDataset(self.df.persist())

    @classmethod
    def read_json(
        cls,
        input_files,
        backend="pandas",
        files_per_partition=1,
        add_filename=False,
    ):
        return cls(
            _read_json_or_parquet(
                input_files=input_files,
                file_type="jsonl",
                backend=backend,
                files_per_partition=files_per_partition,
                add_filename=add_filename,
            )
        )

    @classmethod
    def read_parquet(
        cls,
        input_files,
        backend="pandas",
        files_per_partition=1,
        add_filename=False,
    ):
        return cls(
            _read_json_or_parquet(
                input_files=input_files,
                file_type="parquet",
                backend=backend,
                files_per_partition=files_per_partition,
                add_filename=add_filename,
            )
        )

    @classmethod
    def read_pickle(
        cls,
        input_files,
        backend="pandas",
        files_per_partition=1,
        add_filename=False,
    ):
        raw_data = read_data(
            input_files=input_files,
            file_type="pickle",
            backend=backend,
            files_per_partition=files_per_partition,
            add_filename=add_filename,
        )

        return cls(raw_data)

    def to_json(
        self,
        output_file_dir,
        write_to_filename=False,
    ):
        """
        See nemo_curator.utils.distributed_utils.write_to_disk docstring for other parameters.

        """
        write_to_disk(
            df=self.df,
            output_file_dir=output_file_dir,
            write_to_filename=write_to_filename,
            output_type="jsonl",
        )

    def to_parquet(
        self,
        output_file_dir,
        write_to_filename=False,
    ):
        """
        See nemo_curator.utils.distributed_utils.write_to_disk docstring for other parameters.

        """
        write_to_disk(
            df=self.df,
            output_file_dir=output_file_dir,
            write_to_filename=write_to_filename,
            output_type="parquet",
        )

    def to_pickle(
        self,
        output_file_dir,
        write_to_filename=False,
    ):
        raise NotImplementedError("DocumentDataset does not support to_pickle yet")


def _read_json_or_parquet(
    input_files,
    file_type,
    backend,
    files_per_partition,
    add_filename,
):
    """
    `input_files` may be a list or a string type.
    If `input_files` is a list, it may be a list of JSONL or Parquet files,
    e.g., `input_files = ["file1.jsonl", "file2.jsonl", "file3.jsonl"]`,
    or a list of directories containing JSONL or Parquet files,
    e.g., `input_files = ["dir1", "dir2", "dir3"]`,
    where each of `dir1`, `dir2`, and `dir3` contain all JSONL or Parquet files.
    If `input_files` is a string, it may be a single JSONL or Parquet file,
    such as `input_files = "my_file.jsonl"`,
    or it may also be a single directory containing JSONL or Parquet files,
    such as `input_files = "my_directory"`.

    See nemo_curator.utils.distributed_utils.read_data docstring for other parameters.

    Returns a DataFrame to be used in initializing a DocumentDataset.

    """
    file_ext = "." + file_type

    if isinstance(input_files, list):
        # List of jsonl or parquet files
        if all(f.endswith(file_ext) for f in input_files):
            raw_data = read_data(
                input_files,
                file_type=file_type,
                backend=backend,
                files_per_partition=files_per_partition,
                add_filename=add_filename,
            )

        # List of directories
        else:
            dfs = []

            for data_path in input_files:
                files = get_all_files_paths_under(
                    root=data_path, recurse_subdirectories=False
                )
                df = read_data(
                    files,
                    file_type=file_type,
                    backend=backend,
                    files_per_partition=files_per_partition,
                    add_filename=add_filename,
                )
                dfs.append(df)

            raw_data = dd.concat(dfs, ignore_unknown_divisions=True)

    elif isinstance(input_files, str):
        # Single file
        if input_files.endswith(file_ext):
            files = [input_files]

        # Directory of jsonl or parquet files
        else:
            files = get_all_files_paths_under(
                root=input_files, recurse_subdirectories=False
            )

        raw_data = read_data(
            input_files=files,
            file_type=file_type,
            backend=backend,
            files_per_partition=files_per_partition,
            add_filename=add_filename,
        )

    else:
        raise TypeError("File input must be a string or list.")

    return raw_data
