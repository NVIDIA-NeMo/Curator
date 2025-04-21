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
from pathlib import Path

import pandas as pd
import pytest

from nemo_curator.datasets import DocumentDataset
from nemo_curator.datasets.doc_dataset import _read_json_or_parquet


def test_to_from_pandas() -> None:
    original_df = pd.DataFrame({"first_col": [1, 2, 3], "second_col": ["a", "b", "c"]})
    dataset = DocumentDataset.from_pandas(original_df)
    converted_df = dataset.to_pandas()
    pd.testing.assert_frame_equal(original_df, converted_df)


def test_persist() -> None:
    original_df = pd.DataFrame({"first_col": [1, 2, 3], "second_col": ["a", "b", "c"]})
    dataset = DocumentDataset.from_pandas(original_df)
    dataset.persist()


def test_repartition() -> None:
    original_df = pd.DataFrame({"first_col": [1, 2, 3], "second_col": ["a", "b", "c"]})
    dataset = DocumentDataset.from_pandas(original_df)
    dataset = dataset.repartition(npartitions=3)
    assert dataset.df.npartitions == 3  # noqa: PLR2004


def test_head() -> None:
    original_df = pd.DataFrame({"first_col": [1, 2, 3], "second_col": ["a", "b", "c"]})
    dataset = DocumentDataset.from_pandas(original_df)
    expected_df = pd.DataFrame({"first_col": [1, 2], "second_col": ["a", "b"]})
    pd.testing.assert_frame_equal(expected_df, dataset.head(2))


def test_read_pickle(tmpdir: Path) -> None:
    original_df = pd.DataFrame({"first_col": [1, 2, 3], "second_col": ["a", "b", "c"]})
    output_file = str(tmpdir / "output.pkl")
    original_df.to_pickle(output_file)
    dataset = DocumentDataset.read_pickle(output_file)
    pd.testing.assert_frame_equal(original_df, dataset.df.compute())


def test_to_pickle(tmpdir: Path) -> None:
    original_df = pd.DataFrame({"first_col": [1, 2, 3], "second_col": ["a", "b", "c"]})
    dataset = DocumentDataset.from_pandas(original_df)

    output_file = str(tmpdir / "output.pkl")
    with pytest.raises(NotImplementedError):
        dataset.to_pickle(output_file)


def test_read_json_or_parquet(tmpdir: Path) -> None:
    original_df = pd.DataFrame({"first_col": [1, 2, 3], "second_col": ["a", "b", "c"]})

    directory_1 = str(tmpdir / "directory_1")
    directory_2 = str(tmpdir / "directory_2")
    os.makedirs(directory_1, exist_ok=True)
    os.makedirs(directory_2, exist_ok=True)

    file_1 = directory_1 + "/file_1.jsonl"
    file_2 = directory_2 + "/file_2.jsonl"
    original_df.to_json(file_1, orient="records", lines=True)
    original_df.to_json(file_2, orient="records", lines=True)

    # List of directories
    data = _read_json_or_parquet(
        input_files=[directory_1, directory_2],
        file_type="jsonl",
        backend="pandas",
        files_per_partition=1,
    )
    assert len(data) == 6  # noqa: PLR2004

    file_series = pd.Series([file_1, file_2])
    # Non string or list input
    with pytest.raises(TypeError):
        data = _read_json_or_parquet(
            input_files=file_series,
            file_type="jsonl",
            backend="pandas",
            files_per_partition=1,
        )
