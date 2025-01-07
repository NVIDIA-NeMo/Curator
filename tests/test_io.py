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
import random
import string
import tempfile
from datetime import datetime, timedelta

import pandas as pd
import pytest
from dask import dataframe as dd
from dask.dataframe.utils import assert_eq

from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import (
    read_single_partition,
    single_partition_write_with_filename,
    write_to_disk,
)
from nemo_curator.utils.file_utils import get_all_files_paths_under


def _generate_dummy_dataset(num_rows: int = 50) -> str:
    # Function to generate a shuffled sequence of integers
    def shuffled_integers(length: int = num_rows) -> int:
        # Create a list of numbers from 0 to length - 1
        integers = list(range(length))

        # Shuffle the list
        random.shuffle(integers)

        # Yield one number from the list each time the generator is invoked
        for integer in integers:
            yield integer

    # Function to generate a random string of a given length
    def generate_random_string(length: int = 10) -> str:
        characters = string.ascii_letters + string.digits  # Alphanumeric characters

        return "".join(random.choice(characters) for _ in range(length))

    # Function to generate a random datetime
    def generate_random_datetime() -> str:
        # Define start and end dates
        start_date = datetime(1970, 1, 1)  # Unix epoch
        end_date = datetime.now()  # Current date

        # Calculate the total number of seconds between the start and end dates
        delta = end_date - start_date
        total_seconds = int(delta.total_seconds())

        # Generate a random number of seconds within this range
        random_seconds = random.randint(0, total_seconds)

        # Add the random number of seconds to the start date to get a random datetime
        random_datetime = start_date + timedelta(seconds=random_seconds)

        # Convert to UTC and format the datetime
        random_datetime_utc = random_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        return random_datetime_utc

    # Generate the corpus
    corpus = []
    for integer in shuffled_integers():
        corpus.append(
            json.dumps(
                {
                    "id": integer,
                    "date": generate_random_datetime(),
                    "text": generate_random_string(random.randint(5, 100)),
                }
            )
        )

    # Return the corpus
    return "\n".join(corpus)


@pytest.fixture
def jsonl_dataset():
    return _generate_dummy_dataset(num_rows=10)


class TestIO:
    def test_meta_dict(self, jsonl_dataset):
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
            # Write the corpus to the file
            temp_file.write(jsonl_dataset.encode("utf-8"))

            # Flush the data to ensure it's written to disk
            temp_file.flush()

            # Move the cursor to the beginning of the file before reading
            temp_file.seek(0)

            # Read the dataset
            dataset = DocumentDataset.read_json(
                temp_file.name, input_meta={"id": float}
            )

        output_meta = str({col: str(dtype) for col, dtype in dataset.df.dtypes.items()})

        expected_meta = "{'id': 'float64'}"

        assert (
            output_meta == expected_meta
        ), f"Expected: {expected_meta}, got: {output_meta}"

    def test_meta_str(self, jsonl_dataset):
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
            # Write the corpus to the file
            temp_file.write(jsonl_dataset.encode("utf-8"))

            # Flush the data to ensure it's written to disk
            temp_file.flush()

            # Move the cursor to the beginning of the file before reading
            temp_file.seek(0)

            # Read the dataset
            dataset = DocumentDataset.read_json(
                temp_file.name, input_meta='{"id": "float"}'
            )

        output_meta = str({col: str(dtype) for col, dtype in dataset.df.dtypes.items()})

        expected_meta = "{'id': 'float64'}"

        assert (
            output_meta == expected_meta
        ), f"Expected: {expected_meta}, got: {output_meta}"


class TestWriteWithFilename:
    @pytest.mark.parametrize("keep_filename_column", [True, False])
    @pytest.mark.parametrize("file_ext", ["jsonl", "parquet"])
    @pytest.mark.parametrize("filename_col", ["file_name", "filename"])
    def test_multifile_single_partition(
        self, tmp_path, keep_filename_column, file_ext, filename_col
    ):
        df = pd.DataFrame({"a": [1, 2, 3], filename_col: ["file0", "file1", "file1"]})

        single_partition_write_with_filename(
            df=df,
            output_file_dir=tmp_path,
            keep_filename_column=keep_filename_column,
            output_type=file_ext,
            filename_col=filename_col,
        )
        assert os.path.exists(tmp_path / f"file0.{file_ext}")
        assert os.path.exists(tmp_path / f"file1.{file_ext}")

        if not keep_filename_column:
            df = df.drop(filename_col, axis=1)

        df1 = read_single_partition(
            files=[tmp_path / f"file0.{file_ext}"], backend="pandas", filetype=file_ext
        )
        assert_eq(df1, df.iloc[0:1], check_index=False)

        df2 = read_single_partition(
            files=[tmp_path / f"file1.{file_ext}"], backend="pandas", filetype=file_ext
        )
        assert_eq(df2, df.iloc[1:3], check_index=False)

    @pytest.mark.parametrize("keep_filename_column", [True, False])
    @pytest.mark.parametrize("file_ext", ["jsonl", "parquet"])
    def test_singlefile_single_partition(
        self,
        tmp_path,
        keep_filename_column,
        file_ext,
    ):
        df = pd.DataFrame({"a": [1, 2, 3], "file_name": ["file2", "file2", "file2"]})

        single_partition_write_with_filename(
            df=df,
            output_file_dir=tmp_path,
            keep_filename_column=keep_filename_column,
            output_type=file_ext,
        )
        assert len(os.listdir(tmp_path)) == 1
        assert os.path.exists(tmp_path / f"file2.{file_ext}")

        if not keep_filename_column:
            df = df.drop("file_name", axis=1)
        got = read_single_partition(
            files=[tmp_path / f"file2.{file_ext}"], backend="pandas", filetype=file_ext
        )
        assert_eq(got, df)

    def test_multifile_single_partition_error(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "file_name": ["file0", "file1", "file1"]})

        with pytest.raises(ValueError, match="Unknown output type"):
            single_partition_write_with_filename(
                df=df, output_file_dir=tmp_path, output_type="pickle"
            )

    # Test multiple partitions where we need to append to existing files
    @pytest.mark.parametrize(
        "file_ext, read_f",
        [
            ("jsonl", DocumentDataset.read_json),
            ("parquet", DocumentDataset.read_parquet),
        ],
    )
    @pytest.mark.parametrize("filename_col", ["file_name", "filename"])
    def test_multifile_multi_partition(self, tmp_path, file_ext, read_f, filename_col):
        df1 = pd.DataFrame({"a": [1, 2, 3], filename_col: ["file1", "file2", "file2"]})
        df2 = df1.copy()
        df2[filename_col] = "file3"
        df3 = df1.copy()
        df3[filename_col] = ["file4", "file5", "file6"]
        ddf = dd.concat([df1, df2, df3])
        ddf[filename_col] = ddf[filename_col] + f".{file_ext}"
        write_to_disk(
            df=ddf,
            output_path=tmp_path / file_ext,
            write_to_filename=filename_col,
            output_type=file_ext,
        )

        got_df = read_f(
            str(tmp_path / file_ext),
            blocksize=None,
            files_per_partition=2,
            backend="pandas",
            add_filename=filename_col,
        ).df

        assert_eq(got_df, ddf, check_index=False)


class TestFileExtensions:
    def test_keep_extensions(self, tmp_path):
        json_1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        json_1.to_json(tmp_path / "json_1.jsonl", orient="records", lines=True)
        json_2 = pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})
        json_2.to_json(tmp_path / "json_2.jsonl", orient="records", lines=True)

        json_df = pd.concat([json_1, json_2])

        parquet_file = pd.DataFrame({"a": [13, 14, 15], "b": [16, 17, 18]})
        parquet_file.to_parquet(tmp_path / "parquet_file.parquet")
        csv_file = pd.DataFrame({"a": [19, 20, 21], "b": [22, 23, 24]})
        csv_file.to_csv(tmp_path / "csv_file.csv")

        with pytest.raises(RuntimeError):
            doc = DocumentDataset.read_json(str(tmp_path))

        input_files = get_all_files_paths_under(str(tmp_path), keep_extensions="jsonl")
        input_dataset = DocumentDataset.read_json(input_files)
        assert json_df.equals(input_dataset.df.compute())

        input_files = get_all_files_paths_under(
            str(tmp_path), keep_extensions=["jsonl"]
        )
        input_dataset = DocumentDataset.read_json(input_files)
        assert json_df.equals(input_dataset.df.compute())

        input_files = get_all_files_paths_under(
            str(tmp_path), keep_extensions=["jsonl", "parquet"]
        )
        assert sorted(input_files) == [
            str(tmp_path / "json_1.jsonl"),
            str(tmp_path / "json_2.jsonl"),
            str(tmp_path / "parquet_file.parquet"),
        ]

    def test_write_single_jsonl_file(self, tmp_path):
        json_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        json_df.to_json(tmp_path / "json_file.jsonl", orient="records", lines=True)

        input_path = str(tmp_path / "json_file.jsonl")
        output_path = str(tmp_path / "single_output.jsonl")
        doc = DocumentDataset.read_json(input_path)
        doc.to_json(output_path)

        result = DocumentDataset.read_json(output_path)
        assert json_df.equals(result.df.compute())
