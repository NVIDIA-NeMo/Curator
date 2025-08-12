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
from collections.abc import Callable
from functools import wraps
from typing import Literal, Union

import dask.dataframe as dd
import pandas as pd

from nemo_curator.utils.distributed_utils import read_data, write_to_disk
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.utils.import_utils import gpu_only_import

cudf = gpu_only_import("cudf")


class DocumentDataset:
    """
    A collection of documents and document metadata.
    Internally it may be distributed across multiple nodes, and may be on GPUs.
    """

    def __init__(self, dataset_df: dd.DataFrame):
        self.df = dataset_df

    def __len__(self) -> int:
        return len(self.df)

    # `def persist(self) -> Self` requires Python 3.11 or higher
    def persist(self) -> "DocumentDataset":
        return DocumentDataset(self.df.persist())

    @wraps(dd.DataFrame.repartition)
    def repartition(self, *args, **kwargs) -> "DocumentDataset":
        return self.__class__(self.df.repartition(*args, **kwargs))

    def head(self, n: int = 5) -> Union["cudf.DataFrame", "pd.DataFrame"]:
        return self.df.head(n)

    @classmethod
    def read_json(  # noqa: PLR0913
        cls,
        input_files: str | list[str],
        backend: Literal["pandas", "cudf"] = "pandas",
        files_per_partition: int | None = None,
        blocksize: str | None = "1gb",
        add_filename: bool | str = False,
        input_meta: str | dict | None = None,
        columns: list[str] | None = None,
        **kwargs,
    ) -> "DocumentDataset":
        """
        Read JSONL or JSONL file(s).

        Args:
            input_files: The path of the input file(s).
            backend: The backend to use for reading the data.
            files_per_partition: The number of files to read per partition.
            add_filename: Whether to add a filename column to the DataFrame.
                If True, a new column is added to the DataFrame called `file_name`.
                If str, sets new column name. Default is False.
            input_meta: A dictionary or a string formatted as a dictionary, which outlines
                the field names and their respective data types within the JSONL input file.
            columns: If not None, only these columns will be read from the file.

        """
        return cls(
            _read_json_or_parquet(
                input_files=input_files,
                file_type="jsonl",
                backend=backend,
                add_filename=add_filename,
                files_per_partition=files_per_partition,
                blocksize=blocksize,
                input_meta=input_meta,
                columns=columns,
                **kwargs,
            )
        )

    @classmethod
    def read_parquet(  # noqa: PLR0913
        cls,
        input_files: str | list[str],
        backend: Literal["pandas", "cudf"] = "pandas",
        files_per_partition: int | None = None,
        blocksize: str | None = "1gb",
        add_filename: bool | str = False,
        columns: list[str] | None = None,
        **kwargs,
    ) -> "DocumentDataset":
        """
        Read Parquet file(s).

        Args:
            input_files: The path of the input file(s).
            backend: The backend to use for reading the data.
            files_per_partition: The number of files to read per partition.
            add_filename: Whether to add a filename column to the DataFrame.
                If True, a new column is added to the DataFrame called `file_name`.
                If str, sets new column name. Default is False.
            columns: If not None, only these columns will be read from the file.
                There is a significant performance gain when specifying columns for Parquet files.

        """
        return cls(
            _read_json_or_parquet(
                input_files=input_files,
                file_type="parquet",
                backend=backend,
                add_filename=add_filename,
                files_per_partition=files_per_partition,
                blocksize=blocksize,
                columns=columns,
                **kwargs,
            )
        )

    @classmethod
    def read_pickle(
        cls,
        input_files: str | list[str],
        backend: Literal["pandas", "cudf"] = "pandas",
        columns: list[str] | None = None,
        **kwargs,
    ) -> "DocumentDataset":
        """
        Read Pickle file(s).

        Args:
            input_files: The path of the input file(s).
            backend: The backend to use for reading the data.
            files_per_partition: The number of files to read per partition.
            add_filename: Whether to add a filename column to the DataFrame.
                If True, a new column is added to the DataFrame called `file_name`.
                If str, sets new column name. Default is False.
            columns: If not None, only these columns will be read from the file.

        """
        return cls(
            read_data(
                input_files=input_files,
                file_type="pickle",
                backend=backend,
                columns=columns,
                **kwargs,
            )
        )

    @classmethod
    def read_custom(  # noqa: PLR0913
        cls,
        input_files: str | list[str],
        file_type: str,
        read_func_single_partition: Callable[
            [list[str], str, bool, str | dict, dict],
            Union["cudf.DataFrame", "pd.DataFrame"],
        ],
        files_per_partition: int | None = None,
        backend: Literal["pandas", "cudf"] | None = None,
        add_filename: bool | str = False,
        columns: list[str] | None = None,
        input_meta: str | dict | None = None,
        **kwargs,
    ) -> "DocumentDataset":
        """
        Read custom data from a file or directory based on a custom read function.


        Args:
            input_files: The path of the input file(s).
                If input_file is a string that ends with the file_type, we consider it as a single file.
                If input_file is a string that does not end with the file_type, we consider it as a directory
                and read all files under the directory.
                If input_file is a list of strings, we assume each string is a file path.
            file_type: The type of the file to read.
            read_func_single_partition: A function that reads a single file or a list of files in an single Dask partition.
                The function should take the following arguments:
                - files: A list of file paths.
                - file_type: The type of the file to read (in case you want to handle different file types differently).
                - backend: Read below
                - add_filename: Read below
                - columns: Read below
                - input_meta: Read below
            backend: The backend to use for reading the data, in case you want to handle pd.DataFrame or cudf.DataFrame.
            files_per_partition: The number of files to read per partition.
            add_filename: Whether to add a filename column to the DataFrame.
                If True, a new column is added to the DataFrame called `file_name`.
                If str, sets new column name. Default is False.
            columns: If not None, only these columns will be returned from the output of the read_func_single_partition function.
            input_meta: A dictionary or a string formatted as a dictionary, which outlines
                the field names and their respective data types within the JSONL input file.
        """

        if isinstance(input_files, str):
            if input_files.endswith(file_type):
                files = [input_files]
            else:
                files = get_all_files_paths_under(
                    root=input_files,
                    recurse_subdirectories=False,
                    keep_extensions=[file_type],
                )
        elif isinstance(input_files, list):
            files = input_files
        else:
            msg = "input_files must be a string or list"
            raise TypeError(msg)

        return cls(
            read_data(
                input_files=files,
                file_type=file_type,
                backend=backend,
                files_per_partition=files_per_partition,
                blocksize=None,
                add_filename=add_filename,
                columns=columns,
                input_meta=input_meta,
                read_func_single_partition=read_func_single_partition,
                **kwargs,
            )
        )

    def to_json(
        self,
        output_path: str,
        write_to_filename: bool | str = False,
        keep_filename_column: bool = False,
        partition_on: str | None = None,
        compression: str | None = None,
    ) -> None:
        """
        Writes the dataset to the specified path in JSONL format.

        If `write_to_filename` is True, the DataFrame is expected to have a column
        that specifies the filename for each document. This column can be named
        `file_name` by default, or a custom name if `write_to_filename` is a string.

        Args:
            output_path (str): The directory or file path where the dataset will be written.
            write_to_filename (Union[bool, str]): Determines how filenames are handled.
                - If True, uses the `file_name` column in the DataFrame to determine filenames.
                - If a string, uses that string as the column name for filenames.
                - If False, writes all data to the specified `output_path`.
            keep_filename_column (bool): If True, retains the filename column in the output.
                If False, the filename column is dropped from the output.
            partition_on (Optional[str]): The column name used to partition the data.
                If specified, data is partitioned based on unique values in this column,
                with each partition written to a separate directory.
            compression (Optional[str]): The compression to use for the output file.
                If specified, the output file will be compressed using the specified compression.
                Supported compression types are "gzip" or None.
        For more details, refer to the `write_to_disk` function in
        `nemo_curator.utils.distributed_utils`.
        """
        write_to_disk(
            df=self.df,
            output_path=output_path,
            write_to_filename=write_to_filename,
            keep_filename_column=keep_filename_column,
            partition_on=partition_on,
            output_type="jsonl",
            compression=compression,
        )

    def to_parquet(
        self,
        output_path: str,
        write_to_filename: bool | str = False,
        keep_filename_column: bool = False,
        partition_on: str | None = None,
    ) -> None:
        """
        Writes the dataset to the specified path in Parquet format.

        If `write_to_filename` is True, the DataFrame is expected to have a column
        that specifies the filename for each document. This column can be named
        `file_name` by default, or a custom name if `write_to_filename` is a string.

        Args:
            output_path (str): The directory or file path where the dataset will be written.
            write_to_filename (Union[bool, str]): Determines how filenames are handled.
                - If True, uses the `file_name` column in the DataFrame to determine filenames.
                - If a string, uses that string as the column name for filenames.
                - If False, writes all data to the specified `output_path`.
            keep_filename_column (bool): If True, retains the filename column in the output.
                If False, the filename column is dropped from the output.
            partition_on (Optional[str]): The column name used to partition the data.
                If specified, data is partitioned based on unique values in this column,
                with each partition written to a separate directory.

        For more details, refer to the `write_to_disk` function in
        `nemo_curator.utils.distributed_utils`.
        """
        write_to_disk(
            df=self.df,
            output_path=output_path,
            write_to_filename=write_to_filename,
            keep_filename_column=keep_filename_column,
            partition_on=partition_on,
            output_type="parquet",
        )

    def to_pickle(
        self,
        output_path: str,
        write_to_filename: bool | str = False,
    ) -> None:
        msg = "DocumentDataset does not support to_pickle yet"
        raise NotImplementedError(msg)

    @classmethod
    def from_pandas(
        cls,
        data: pd.DataFrame,
        npartitions: int | None = 1,
        chunksize: int | None = None,
        sort: bool | None = True,
        # TODO: Remove this parameter from all parts of the codebase
        name: str | None = None,  # noqa: ARG003
    ) -> "DocumentDataset":
        """
        Creates a document dataset from a pandas data frame.
        For more information on the arguments see Dask's from_pandas documentation
        https://docs.dask.org/en/stable/generated/dask.dataframe.from_pandas.html

        Args:
            data: A pandas dataframe
        Returns:
            A document dataset with a pandas backend (on the CPU).
        """
        return cls(
            dd.from_pandas(
                data=data,
                npartitions=npartitions,
                chunksize=chunksize,
                sort=sort,
            )
        )

    def to_pandas(self) -> pd.DataFrame:
        """
        Creates a pandas dataframe from a DocumentDataset

        Returns:
            A pandas dataframe (on the CPU)
        """
        return self.df.to_backend("pandas").compute()


def _read_json_or_parquet(  # noqa: PLR0913
    input_files: str | list[str],
    file_type: str,
    backend: Literal["cudf", "pandas"],
    add_filename: bool | str = False,
    files_per_partition: int | None = None,
    blocksize: str | None = None,
    input_meta: str | dict | None = None,
    columns: list[str] | None = None,
    **kwargs,
) -> dd.DataFrame:
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
        # List of files
        if all(os.path.isfile(f) for f in input_files):
            raw_data = read_data(
                input_files,
                file_type=file_type,
                backend=backend,
                files_per_partition=files_per_partition,
                blocksize=blocksize,
                add_filename=add_filename,
                input_meta=input_meta,
                columns=columns,
                **kwargs,
            )

        # List of directories
        else:
            dfs = []

            for data_path in input_files:
                files = get_all_files_paths_under(root=data_path, recurse_subdirectories=False)
                df = read_data(
                    files,
                    file_type=file_type,
                    backend=backend,
                    files_per_partition=files_per_partition,
                    blocksize=blocksize,
                    add_filename=add_filename,
                    input_meta=input_meta,
                    columns=columns,
                    **kwargs,
                )
                dfs.append(df)

            raw_data = dd.concat(dfs, ignore_unknown_divisions=True)

    elif isinstance(input_files, str):
        # Single file
        if input_files.endswith(file_ext):
            files = [input_files]

        # Directory of jsonl or parquet files
        else:
            files = get_all_files_paths_under(root=input_files, recurse_subdirectories=False)

        raw_data = read_data(
            input_files=files,
            file_type=file_type,
            backend=backend,
            files_per_partition=files_per_partition,
            blocksize=blocksize,
            add_filename=add_filename,
            input_meta=input_meta,
            columns=columns,
            **kwargs,
        )

    else:
        msg = "File input must be a string or list."
        raise TypeError(msg)

    return raw_data
