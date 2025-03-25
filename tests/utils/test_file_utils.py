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

import json
import os
import pathlib
import shutil
import tempfile
import warnings
from functools import reduce
from unittest.mock import MagicMock, mock_open, patch

import dask.bag as db
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from dask import delayed

from nemo_curator.utils.file_utils import (
    NEMO_CURATOR_HOME,
    _save_jsonl,
    _update_filetype,
    expand_outdir_and_mkdir,
    filter_files_by_extension,
    get_all_files_paths_under,
    get_batched_files,
    get_remaining_files,
    merge_counts,
    mkdir,
    parse_str_of_num_bytes,
    remove_path_extension,
    reshard_jsonl,
    separate_by_metadata,
    write_dataframe_by_meta,
    write_record,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestBasicFileUtils:
    """Tests for basic file utility functions."""

    def test_mkdir(self, temp_dir):
        """Test that mkdir creates directories."""
        test_dir = os.path.join(temp_dir, "test_directory")
        mkdir(test_dir)
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)

        # Test with nested directory
        nested_dir = os.path.join(test_dir, "nested", "directory")
        mkdir(nested_dir)
        assert os.path.exists(nested_dir)
        assert os.path.isdir(nested_dir)

    def test_expand_outdir_and_mkdir(self, temp_dir):
        """Test that expand_outdir_and_mkdir expands paths and creates directories."""
        # Test with a normal path
        test_dir = os.path.join(temp_dir, "test_directory")
        result = expand_outdir_and_mkdir(test_dir)
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)
        assert os.path.abspath(test_dir) == result

        # Test with a path containing a tilde
        with patch(
            "os.path.expanduser", return_value=os.path.join(temp_dir, "expanded_user")
        ):
            result = expand_outdir_and_mkdir("~/test")
            assert os.path.join(temp_dir, "expanded_user") in result

    def test_remove_path_extension(self):
        """Test remove_path_extension function."""
        # Test with a simple file path
        path = "/path/to/file.txt"
        result = remove_path_extension(path)
        assert result == "/path/to/file"

        # Test with a path containing multiple dots
        path = "/path/to/file.name.txt"
        result = remove_path_extension(path)
        assert result == "/path/to/file.name"

        # Test with a path with no extension
        path = "/path/to/file"
        result = remove_path_extension(path)
        assert result == "/path/to/file"

        # Test with a path ending with a directory separator
        path = "/path/to/directory/"
        result = remove_path_extension(path)
        assert result == "/path/to/directory"


class TestFileFiltering:
    """Tests for file filtering utility functions."""

    def test_filter_files_by_extension_with_string(self):
        """Test filter_files_by_extension with a string extension."""
        files = [
            "file1.txt",
            "file2.json",
            "file3.parquet",
            "file4.txt",
            "file5.jsonl",
        ]
        result = filter_files_by_extension(files, "txt")
        assert result == ["file1.txt", "file4.txt"]

    def test_filter_files_by_extension_with_list(self):
        """Test filter_files_by_extension with a list of extensions."""
        files = [
            "file1.txt",
            "file2.json",
            "file3.parquet",
            "file4.txt",
            "file5.jsonl",
        ]
        result = filter_files_by_extension(files, ["txt", "json"])
        assert result == ["file1.txt", "file2.json", "file4.txt"]

    def test_filter_files_by_extension_with_period(self):
        """Test filter_files_by_extension with extensions that include a period."""
        files = [
            "file1.txt",
            "file2.json",
            "file3.parquet",
            "file4.txt",
            "file5.jsonl",
        ]
        result = filter_files_by_extension(files, [".txt", ".json"])
        assert result == ["file1.txt", "file2.json", "file4.txt"]

    def test_filter_files_by_extension_warning(self):
        """Test that filter_files_by_extension issues a warning when files are skipped."""
        files = [
            "file1.txt",
            "file2.json",
            "file3.parquet",
            "file4.txt",
            "file5.jsonl",
        ]
        with warnings.catch_warnings(record=True) as w:
            result = filter_files_by_extension(files, "txt")
            assert len(w) == 1
            assert "Skipped at least one file" in str(w[0].message)

    def test_get_all_files_paths_under(self, temp_dir):
        """Test get_all_files_paths_under function."""
        # Create test directory structure
        subdir1 = os.path.join(temp_dir, "subdir1")
        subdir2 = os.path.join(temp_dir, "subdir2")
        os.makedirs(subdir1)
        os.makedirs(subdir2)

        # Create test files
        files = [
            os.path.join(temp_dir, "file1.txt"),
            os.path.join(temp_dir, "file2.json"),
            os.path.join(subdir1, "file3.txt"),
            os.path.join(subdir1, "file4.json"),
            os.path.join(subdir2, "file5.txt"),
            os.path.join(subdir2, "file6.parquet"),
        ]
        for file in files:
            with open(file, "w") as f:
                f.write("test content")

        # Test with recursion enabled (default)
        result = get_all_files_paths_under(temp_dir)
        assert len(result) == 6
        for file in files:
            assert file in result

        # Test with recursion disabled
        result = get_all_files_paths_under(temp_dir, recurse_subdirectories=False)
        assert len(result) == 2
        assert os.path.join(temp_dir, "file1.txt") in result
        assert os.path.join(temp_dir, "file2.json") in result

        # Test with file extension filtering
        result = get_all_files_paths_under(temp_dir, keep_extensions="txt")
        assert len(result) == 3
        assert os.path.join(temp_dir, "file1.txt") in result
        assert os.path.join(subdir1, "file3.txt") in result
        assert os.path.join(subdir2, "file5.txt") in result

        # Test with multiple file extension filtering
        result = get_all_files_paths_under(temp_dir, keep_extensions=["txt", "json"])
        assert len(result) == 5
        assert os.path.join(temp_dir, "file1.txt") in result
        assert os.path.join(temp_dir, "file2.json") in result
        assert os.path.join(subdir1, "file3.txt") in result
        assert os.path.join(subdir1, "file4.json") in result
        assert os.path.join(subdir2, "file5.txt") in result

    def test_update_filetype(self):
        """Test _update_filetype function."""
        # Test with None file types
        file_set = {"file1.txt", "file2.json", "file3"}
        result = _update_filetype(file_set, None, None)
        assert result == file_set

        # Test with same file types
        result = _update_filetype(file_set, "txt", "txt")
        assert result == file_set

        # Test with different file types
        result = _update_filetype(file_set, "txt", "json")
        assert "file1.json" in result
        assert "file2.json" in result
        assert "file3" in result

        # Test with file types containing a period
        result = _update_filetype(file_set, ".txt", ".json")
        assert "file1.json" in result
        assert "file2.json" in result
        assert "file3" in result


class TestFileProcessingUtils:
    """Tests for file processing utility functions."""

    def test_get_remaining_files(self, temp_dir):
        """Test get_remaining_files function."""
        # Create input and output directories
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        # Create input files
        input_files = [os.path.join(input_dir, f"file{i}.txt") for i in range(5)]
        for file in input_files:
            with open(file, "w") as f:
                f.write("test content")

        # Create output files (already processed)
        output_files = [os.path.join(output_dir, f"file{i}.txt") for i in range(2)]
        for file in output_files:
            with open(file, "w") as f:
                f.write("processed content")

        # Test with no output file type specified
        with patch("nemo_curator.utils.file_utils.os.path.exists", return_value=True):
            with patch("nemo_curator.utils.file_utils.os.scandir") as mock_scandir:
                mock_input_scan = MagicMock()
                mock_input_scan.__iter__.return_value = [
                    MagicMock(path=f) for f in input_files
                ]

                mock_output_scan = MagicMock()
                mock_output_scan.__iter__.return_value = [
                    MagicMock(path=f) for f in output_files
                ]

                mock_scandir.side_effect = lambda path: (
                    mock_input_scan if path == input_dir else mock_output_scan
                )

                result = get_remaining_files(input_dir, output_dir, "txt")
                assert len(result) == 3
                assert all(f"file{i}.txt" in r for i, r in zip(range(2, 5), result))

        # Test with pickle input_file_type
        result = get_remaining_files("path/to/data.pickle", output_dir, "pickle")
        assert result == ["path/to/data.pickle"]

        # Test with different output file type
        with patch("nemo_curator.utils.file_utils.os.path.exists", return_value=True):
            with patch("nemo_curator.utils.file_utils.os.scandir") as mock_scandir:
                mock_input_scan = MagicMock()
                mock_input_scan.__iter__.return_value = [
                    MagicMock(path=f) for f in input_files
                ]

                mock_output_scan = MagicMock()
                mock_output_scan.__iter__.return_value = [
                    MagicMock(path=f.replace(".txt", ".json")) for f in output_files
                ]

                mock_scandir.side_effect = lambda path: (
                    mock_input_scan if path == input_dir else mock_output_scan
                )

                result = get_remaining_files(input_dir, output_dir, "txt", "json")
                assert len(result) == 3
                assert all(f"file{i}.txt" in r for i, r in zip(range(2, 5), result))

        # Test with num_files limit
        with patch("nemo_curator.utils.file_utils.os.path.exists", return_value=True):
            with patch("nemo_curator.utils.file_utils.os.scandir") as mock_scandir:
                mock_input_scan = MagicMock()
                mock_input_scan.__iter__.return_value = [
                    MagicMock(path=f) for f in input_files
                ]

                mock_output_scan = MagicMock()
                mock_output_scan.__iter__.return_value = [
                    MagicMock(path=f) for f in output_files
                ]

                mock_scandir.side_effect = lambda path: (
                    mock_input_scan if path == input_dir else mock_output_scan
                )

                result = get_remaining_files(input_dir, output_dir, "txt", num_files=3)
                assert (
                    len(result) == 1
                )  # 3 total files - 2 already processed = 1 remaining

    def test_get_remaining_files_nonexistent_output_dir(self, temp_dir):
        """Test get_remaining_files when output directory doesn't exist."""
        # Create input directory with test files
        input_dir = os.path.join(temp_dir, "input")
        os.makedirs(input_dir)

        # Define a non-existent output directory
        output_dir = os.path.join(temp_dir, "nonexistent_output")

        # Create input files
        input_files = [os.path.join(input_dir, f"file{i}.txt") for i in range(3)]
        for file in input_files:
            with open(file, "w") as f:
                f.write("test content")

        # Mock os.path.exists to return False for output_dir and then pass through for other paths
        original_exists = os.path.exists

        def mock_exists(path):
            if path == output_dir:
                return False
            return original_exists(path)

        with patch(
            "nemo_curator.utils.file_utils.os.path.exists", side_effect=mock_exists
        ):
            with patch(
                "nemo_curator.utils.file_utils.expand_outdir_and_mkdir"
            ) as mock_mkdir:
                with patch("nemo_curator.utils.file_utils.os.scandir") as mock_scandir:
                    # Mock scandir for input directory to return real files
                    mock_input_scan = MagicMock()
                    mock_input_scan.__iter__.return_value = [
                        MagicMock(path=f) for f in input_files
                    ]

                    # Mock scandir for output directory to return empty list (no files)
                    mock_output_scan = MagicMock()
                    mock_output_scan.__iter__.return_value = []

                    mock_scandir.side_effect = lambda path: (
                        mock_input_scan if path == input_dir else mock_output_scan
                    )

                    # Call get_remaining_files with non-existent output directory
                    result = get_remaining_files(input_dir, output_dir, "txt")

                    # Verify that expand_outdir_and_mkdir was called to create the directory
                    mock_mkdir.assert_called_once_with(output_dir)

                    # Verify that all input files are returned (since output dir is empty)
                    assert len(result) == 3
                    assert all(f"file{i}.txt" in r for i, r in zip(range(3), result))

    def test_get_batched_files(self, temp_dir):
        """Test get_batched_files function."""
        # Create input and output directories
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        # Prepare a list of test files
        files = [f"file{i}.txt" for i in range(10)]

        # Mock get_remaining_files to return our test files
        with patch(
            "nemo_curator.utils.file_utils.get_remaining_files", return_value=files
        ):
            # Test with batch_size=4
            batches = list(
                get_batched_files(input_dir, output_dir, "txt", batch_size=4)
            )
            assert len(batches) == 3
            assert batches[0] == files[0:4]
            assert batches[1] == files[4:8]
            assert batches[2] == files[8:10]

            # Test with batch_size larger than number of files
            batches = list(
                get_batched_files(input_dir, output_dir, "txt", batch_size=20)
            )
            assert len(batches) == 1
            assert batches[0] == files

    def test_merge_counts(self):
        """Test merge_counts function."""
        # Test with non-overlapping dictionaries
        first = {"a": 1, "b": 2}
        second = {"c": 3, "d": 4}
        result = merge_counts(first, second)
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}

        # Test with overlapping dictionaries
        first = {"a": 1, "b": 2, "c": 3}
        second = {"b": 4, "c": 5, "d": 6}
        result = merge_counts(first, second)
        assert result == {"a": 1, "b": 6, "c": 8, "d": 6}

        # Test with empty dictionaries
        first = {}
        second = {"a": 1, "b": 2}
        result = merge_counts(first, second)
        assert result == {"a": 1, "b": 2}

        first = {"a": 1, "b": 2}
        second = {}
        result = merge_counts(first, second)
        assert result == {"a": 1, "b": 2}


class TestDataFrameUtils:
    """Tests for DataFrame utility functions."""

    def test_write_dataframe_by_meta(self, temp_dir):
        """Test write_dataframe_by_meta function."""
        # Create a test DataFrame
        df = pd.DataFrame(
            {
                "text": ["hello", "world", "test", "data"],
                "category": ["A", "B", "A", "C"],
                "file_name": ["file1.txt", "file2.txt", "file3.txt", "file4.txt"],
            }
        )

        # Mock single_partition_write_with_filename to avoid actual file writing
        with patch(
            "nemo_curator.utils.file_utils.single_partition_write_with_filename"
        ) as mock_write:
            # Test basic functionality
            result = write_dataframe_by_meta(df, temp_dir, "category")
            assert result == {"A": 2, "B": 1, "C": 1}
            assert mock_write.call_count == 3

            # Test with include_values
            mock_write.reset_mock()
            result = write_dataframe_by_meta(
                df, temp_dir, "category", include_values=["A", "B"]
            )
            assert result == {"A": 2, "B": 1}
            assert mock_write.call_count == 2

            # Test with exclude_values
            mock_write.reset_mock()
            result = write_dataframe_by_meta(
                df, temp_dir, "category", exclude_values=["C"]
            )
            assert result == {"A": 2, "B": 1}
            assert mock_write.call_count == 2

            # Test with remove_metadata=True
            mock_write.reset_mock()
            result = write_dataframe_by_meta(
                df, temp_dir, "category", remove_metadata=True
            )
            assert result == {"A": 2, "B": 1, "C": 1}
            assert mock_write.call_count == 3
            # Verify that the metadata column was dropped
            for call_args in mock_write.call_args_list:
                assert "category" not in call_args[0][0].columns

    def test_write_record(self, temp_dir):
        """Test write_record function."""
        # Create test directories
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir)

        # Test with valid JSON line
        line = '{"text": "hello", "category": "A"}'
        file_name = os.path.join(input_dir, "file1.txt")

        # Mock os.makedirs and open to avoid actual file operations
        with patch("nemo_curator.utils.file_utils.os.makedirs") as mock_makedirs:
            with patch("nemo_curator.utils.file_utils.open", mock_open()) as mock_file:
                result = write_record(
                    input_dir, file_name, line, "category", output_dir
                )
                assert result == "A"
                mock_makedirs.assert_called_once_with(
                    os.path.join(output_dir, "A", ""), exist_ok=True
                )
                mock_file.assert_called_once_with(
                    os.path.join(output_dir, "A", "file1.txt"), "a"
                )
                mock_file().write.assert_called_once_with(
                    '{"text": "hello", "category": "A"}\n'
                )

        # Test with include_values filter that matches
        with patch("nemo_curator.utils.file_utils.os.makedirs") as mock_makedirs:
            with patch("nemo_curator.utils.file_utils.open", mock_open()) as mock_file:
                result = write_record(
                    input_dir,
                    file_name,
                    line,
                    "category",
                    output_dir,
                    include_values=["A"],
                )
                assert result == "A"
                mock_makedirs.assert_called_once()
                mock_file.assert_called_once()

        # Test with include_values filter that doesn't match
        with patch("nemo_curator.utils.file_utils.os.makedirs") as mock_makedirs:
            with patch("nemo_curator.utils.file_utils.open", mock_open()) as mock_file:
                result = write_record(
                    input_dir,
                    file_name,
                    line,
                    "category",
                    output_dir,
                    include_values=["B"],
                )
                assert result is None
                mock_makedirs.assert_not_called()
                mock_file.assert_not_called()

        # Test with exclude_values filter that matches
        with patch("nemo_curator.utils.file_utils.os.makedirs") as mock_makedirs:
            with patch("nemo_curator.utils.file_utils.open", mock_open()) as mock_file:
                result = write_record(
                    input_dir,
                    file_name,
                    line,
                    "category",
                    output_dir,
                    exclude_values=["A"],
                )
                assert result is None
                mock_makedirs.assert_not_called()
                mock_file.assert_not_called()

        # Test with invalid JSON
        result = write_record(
            input_dir, file_name, "invalid json", "category", output_dir
        )
        assert result is None

        # Test with missing field
        line = '{"text": "hello", "other_field": "value"}'
        result = write_record(input_dir, file_name, line, "category", output_dir)
        assert result is None

    def test_separate_by_metadata(self, temp_dir):
        """Test separate_by_metadata function."""
        # This is a complex function with many branches, so we'll test the high-level logic
        input_dir = os.path.join(temp_dir, "input")
        os.makedirs(input_dir, exist_ok=True)

        # Case 1: Input is a DataFrame
        df = dd.from_pandas(
            pd.DataFrame(
                {
                    "text": ["hello", "world"],
                    "category": ["A", "B"],
                    "file_name": ["file1.txt", "file2.txt"],
                }
            ),
            npartitions=1,
        )

        # Create a mock partition that will be returned by to_delayed
        mock_partition = MagicMock()

        # Setup mocks for to_delayed method
        with patch.object(
            df, "to_delayed", return_value=[mock_partition]
        ) as mock_to_delayed:
            # Create a final result mock that will be returned from the function
            final_result_mock = MagicMock()

            # Setup write_dataframe_by_meta mock
            mock_write = MagicMock()
            write_delayed_mock = MagicMock()
            # This mock will represent the delayed function that will be called with arguments
            delayed_write_fn = MagicMock()
            delayed_write_fn.return_value = write_delayed_mock

            # Setup mock for reduce function and merge_counts
            mock_merge_counts = MagicMock()
            mock_reduce = MagicMock()
            delayed_reduce_fn = MagicMock()
            delayed_reduce_fn.return_value = final_result_mock

            with patch("nemo_curator.utils.file_utils.delayed") as mock_delayed:
                with patch(
                    "nemo_curator.utils.file_utils.write_dataframe_by_meta", mock_write
                ):
                    with patch(
                        "nemo_curator.utils.file_utils.merge_counts", mock_merge_counts
                    ):
                        with patch("nemo_curator.utils.file_utils.reduce", mock_reduce):
                            # Configure the mocks to return the expected values
                            # Mock delayed to return a callable that we can check later
                            mock_delayed.side_effect = lambda func: (
                                delayed_write_fn
                                if func is mock_write
                                else (
                                    delayed_reduce_fn
                                    if func is mock_reduce
                                    else MagicMock()
                                )
                            )

                            # Call the function
                            result = separate_by_metadata(df, temp_dir, "category")

                            # Verify to_delayed was called
                            mock_to_delayed.assert_called_once()

                            # Verify delayed was called with write_dataframe_by_meta
                            mock_delayed.assert_any_call(mock_write)

                            # Verify the delayed function was called with the right arguments
                            delayed_write_fn.assert_called_with(
                                mock_partition,
                                temp_dir,
                                "category",
                                False,
                                "jsonl",
                                None,
                                None,
                                "file_name",
                            )

                            # Verify reduce was called correctly
                            mock_delayed.assert_any_call(mock_reduce)
                            delayed_reduce_fn.assert_called_once()

                            # Verify the final result
                            assert result == final_result_mock

        # Case 2: Input is a directory with jsonl files when both input and output are jsonl
        test_file = os.path.join(input_dir, "test.jsonl")
        with open(test_file, "w") as f:
            f.write('{"text": "hello", "category": "A"}\n')
            f.write('{"text": "world", "category": "B"}\n')

        # Mock db.read_text to return a controlled bag
        mock_bag = MagicMock()
        mock_frequencies = MagicMock()
        mock_frequencies_compute = MagicMock(return_value={"A": 1, "B": 1, None: 0})
        mock_frequencies.compute.return_value = mock_frequencies_compute
        mock_bag.frequencies.return_value = mock_frequencies
        mock_bag.map.return_value = mock_bag

        with patch("nemo_curator.utils.file_utils.db.read_text", return_value=mock_bag):
            # Create a final result for this test case
            final_result = {"A": 1, "B": 1}

            # Update how we patch delayed and reduce
            with patch("nemo_curator.utils.file_utils.delayed") as mock_delayed:
                with patch(
                    "nemo_curator.utils.file_utils.reduce", autospec=True
                ) as mock_reduce:
                    # Create a properly callable delayed_reduce mock
                    delayed_reduce_mock = MagicMock()
                    delayed_reduce_mock.return_value = final_result

                    # Configure mock_delayed to return delayed_reduce_mock when called with mock_reduce
                    def delayed_side_effect(func, *args, **kwargs):
                        if func is mock_reduce:
                            return delayed_reduce_mock
                        return MagicMock()

                    mock_delayed.side_effect = delayed_side_effect

                    # Call the function with a directory path
                    result = separate_by_metadata(
                        input_dir,
                        temp_dir,
                        "category",
                        input_type="jsonl",
                        output_type="jsonl",
                    )

                    # Verify that delayed was called with reduce
                    mock_delayed.assert_called_with(mock_reduce)

                    # Verify that delayed_reduce_mock was called with the expected arguments
                    frequencies = {"A": 1, "B": 1}  # After None is removed
                    delayed_reduce_mock.assert_called_once()

                    # Verify the result
                    assert result == final_result

        # Test with both include_values and exclude_values provided
        with patch("builtins.print") as mock_print:
            result = separate_by_metadata(
                input_dir,
                temp_dir,
                "category",
                include_values=["A"],
                exclude_values=["B"],
            )
            assert result is None
            mock_print.assert_called_once_with(
                "Error: 'include_values' and 'exclude_values' are mutually exclusive."
            )

        # Case 3: Input is a string but not JSON files
        with patch("nemo_curator.utils.file_utils.read_data") as mock_read_data:
            with patch(
                "nemo_curator.utils.file_utils.get_all_files_paths_under"
            ) as mock_get_files:
                with patch("nemo_curator.utils.file_utils.delayed") as mock_delayed:
                    with patch(
                        "nemo_curator.utils.file_utils.expand_outdir_and_mkdir"
                    ) as mock_expand:
                        # Set up the mocks
                        mock_get_files.return_value = ["file1.parquet", "file2.parquet"]
                        mock_df = MagicMock()
                        mock_df.to_delayed.return_value = [MagicMock()]
                        mock_read_data.return_value = mock_df
                        mock_reduce_result = MagicMock()
                        mock_delayed.return_value = mock_reduce_result

                        # Call the function
                        result = separate_by_metadata(
                            input_dir, temp_dir, "category", input_type="parquet"
                        )

                        # Assertions
                        assert mock_get_files.called
                        assert mock_read_data.called
                        assert mock_delayed.called


class TestByteConversions:
    """Tests for byte conversion utility functions."""

    def test_parse_str_of_num_bytes(self):
        """Test parse_str_of_num_bytes function."""
        # Test valid inputs
        assert parse_str_of_num_bytes("1k") == 1024
        assert parse_str_of_num_bytes("2K") == 2048
        assert parse_str_of_num_bytes("1m") == 1048576
        assert parse_str_of_num_bytes("1M") == 1048576
        assert parse_str_of_num_bytes("1g") == 1073741824
        assert parse_str_of_num_bytes("1G") == 1073741824

        # Test with return_str=True
        assert parse_str_of_num_bytes("1k", return_str=True) == "1k"

        # Test invalid input
        with pytest.raises(ValueError):
            parse_str_of_num_bytes("invalid")

        with pytest.raises(ValueError):
            parse_str_of_num_bytes("1x")


class TestJSONLProcessing:
    """Tests for JSONL processing utility functions."""

    def test_save_jsonl(self, temp_dir):
        """Test _save_jsonl function."""
        # Create a mock dask bag
        mock_bag = MagicMock()
        mock_encoded_bag = MagicMock()
        mock_bag.map.return_value = mock_encoded_bag
        mock_encoded_bag.to_textfiles.return_value = [
            os.path.join(temp_dir, "file1.jsonl"),
            os.path.join(temp_dir, "file2.jsonl"),
            os.path.join(temp_dir, "empty.jsonl"),
        ]

        # Mock os.path.getsize to simulate an empty file
        def mock_getsize(path):
            if "empty" in path:
                return 0
            return 100

        # Test the function
        with patch(
            "nemo_curator.utils.file_utils.os.path.getsize", side_effect=mock_getsize
        ):
            with patch("nemo_curator.utils.file_utils.os.remove") as mock_remove:
                _save_jsonl(mock_bag, temp_dir, start_index=5, prefix="test_")

                # Verify that map was called to encode the text
                mock_bag.map.assert_called_once()

                # Verify that to_textfiles was called with the correct parameters
                mock_encoded_bag.to_textfiles.assert_called_once()
                args, kwargs = mock_encoded_bag.to_textfiles.call_args
                assert args[0] == os.path.join(temp_dir, "*.jsonl")
                assert "name_function" in kwargs

                # Verify that empty files were removed
                mock_remove.assert_called_once_with(
                    os.path.join(temp_dir, "empty.jsonl")
                )

    def test_save_jsonl_with_exception(self, temp_dir):
        """Test _save_jsonl function when an exception occurs during file removal."""
        # Create a mock dask bag
        mock_bag = MagicMock()
        mock_encoded_bag = MagicMock()
        mock_bag.map.return_value = mock_encoded_bag
        mock_encoded_bag.to_textfiles.return_value = [
            os.path.join(temp_dir, "file1.jsonl"),
            os.path.join(temp_dir, "file2.jsonl"),
            os.path.join(temp_dir, "empty.jsonl"),
        ]

        # Mock os.path.getsize to simulate an empty file
        def mock_getsize(path):
            if "empty" in path:
                return 0
            return 100

        # Mock os.remove to raise an exception
        def mock_remove_with_exception(path):
            if "empty" in path:
                raise PermissionError("Permission denied")

        # Mock print to capture the exception message
        with (
            patch(
                "nemo_curator.utils.file_utils.os.path.getsize",
                side_effect=mock_getsize,
            ),
            patch(
                "nemo_curator.utils.file_utils.os.remove",
                side_effect=mock_remove_with_exception,
            ),
            patch("nemo_curator.utils.file_utils.print") as mock_print,
        ):
            # Test that the function handles the exception gracefully
            _save_jsonl(mock_bag, temp_dir, start_index=5, prefix="test_")

            # Verify that map and to_textfiles were still called
            mock_bag.map.assert_called_once()
            mock_encoded_bag.to_textfiles.assert_called_once()

            # Verify that print was called with an error message containing the exception
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            assert "An exception occurred" in call_args
            assert "Permission denied" in call_args

    def test_reshard_jsonl(self, temp_dir):
        """Test reshard_jsonl function."""
        # Mock the required functions
        with patch(
            "nemo_curator.utils.file_utils.get_all_files_paths_under"
        ) as mock_get_files:
            with patch("nemo_curator.utils.file_utils.db.read_text") as mock_read_text:
                with patch(
                    "nemo_curator.utils.file_utils.expand_outdir_and_mkdir"
                ) as mock_expand:
                    with patch(
                        "nemo_curator.utils.file_utils._save_jsonl"
                    ) as mock_save:
                        with patch(
                            "nemo_curator.utils.file_utils.parse_str_of_num_bytes"
                        ) as mock_parse:
                            # Set up the mocks
                            mock_get_files.return_value = ["file1.jsonl", "file2.jsonl"]
                            mock_read_text.return_value = "mock_bag"
                            mock_parse.return_value = 104857600  # 100MB
                            # Configure mock_expand to return its input, like the real function does
                            mock_expand.side_effect = lambda x: x

                            # Call the function
                            reshard_jsonl(
                                temp_dir,
                                os.path.join(temp_dir, "output"),
                                "100M",
                                5,
                                "test_",
                            )

                            # Assertions
                            mock_get_files.assert_called_once_with(
                                temp_dir, keep_extensions="jsonl"
                            )
                            mock_read_text.assert_called_once_with(
                                ["file1.jsonl", "file2.jsonl"], blocksize=104857600
                            )
                            mock_expand.assert_called_once_with(
                                os.path.join(temp_dir, "output")
                            )
                            mock_save.assert_called_once_with(
                                "mock_bag",
                                os.path.join(temp_dir, "output"),
                                start_index=5,
                                prefix="test_",
                            )
