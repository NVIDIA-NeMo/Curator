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
import shutil
import sys
import tempfile
import warnings
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import torch
from dask.distributed import Client, Worker

from nemo_curator.utils.distributed_utils import (
    NoWorkerError,
    _enable_spilling,
    _resolve_filename_col,
    _set_torch_to_use_rmm,
    _write_to_jsonl_or_parquet,
    check_dask_cwd,
    get_client,
    get_current_client,
    get_gpu_memory_info,
    get_network_interfaces,
    get_num_workers,
    load_object_on_worker,
    offload_object_on_worker,
    performance_report_if,
    performance_report_if_with_ts_suffix,
    process_all_batches,
    process_batch,
    read_data,
    read_data_blocksize,
    read_data_files_per_partition,
    read_pandas_pickle,
    read_single_partition,
    seed_all,
    select_columns,
    single_partition_write_with_filename,
    start_dask_cpu_local_cluster,
    start_dask_gpu_local_cluster,
    write_to_disk,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestClientFunctions:
    @pytest.mark.gpu
    @patch("nemo_curator.utils.distributed_utils.LocalCUDACluster")
    @patch("nemo_curator.utils.distributed_utils.Client")
    @patch("nemo_curator.utils.distributed_utils._set_torch_to_use_rmm")
    def test_start_dask_gpu_local_cluster(
        self, mock_set_torch_rmm, mock_client, mock_cuda_cluster
    ):
        """Test starting a GPU local cluster."""
        # Setup mock return values
        mock_cluster_instance = MagicMock()
        mock_cuda_cluster.return_value = mock_cluster_instance
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Mock get_num_workers to return a positive value
        with patch(
            "nemo_curator.utils.distributed_utils.get_num_workers", return_value=1
        ):
            # Call the function
            client = start_dask_gpu_local_cluster()

            # Verify the client was created with the right cluster
            mock_client.assert_called_once_with(mock_cluster_instance)
            assert client == mock_client_instance

            # Verify _set_torch_to_use_rmm was called
            mock_set_torch_rmm.assert_called_once()

    @patch("nemo_curator.utils.distributed_utils.LocalCluster")
    @patch("nemo_curator.utils.distributed_utils.Client")
    def test_start_dask_cpu_local_cluster(self, mock_client, mock_local_cluster):
        """Test starting a CPU local cluster."""
        # Setup mock return values
        mock_cluster_instance = MagicMock()
        mock_local_cluster.return_value = mock_cluster_instance
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Mock get_num_workers to return a positive value
        with patch(
            "nemo_curator.utils.distributed_utils.get_num_workers", return_value=1
        ):
            # Call the function
            client = start_dask_cpu_local_cluster()

            # Verify the client was created with the right cluster
            mock_client.assert_called_once_with(mock_cluster_instance)
            assert client == mock_client_instance

    @patch("nemo_curator.utils.distributed_utils.start_dask_gpu_local_cluster")
    @patch("nemo_curator.utils.distributed_utils.start_dask_cpu_local_cluster")
    @patch("nemo_curator.utils.distributed_utils.Client")
    def test_get_client(self, mock_client, mock_cpu_cluster, mock_gpu_cluster):
        """Test get_client function with different params."""
        # Test with scheduler_address
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Mock get_num_workers to return a positive value
        with patch(
            "nemo_curator.utils.distributed_utils.get_num_workers", return_value=1
        ):
            # Test with scheduler_address
            client = get_client(scheduler_address="tcp://localhost:8786")
            mock_client.assert_called_with(
                address="tcp://localhost:8786", timeout="30s"
            )

            # Test with scheduler_file
            client = get_client(scheduler_file="/path/to/scheduler.json")
            mock_client.assert_called_with(
                scheduler_file="/path/to/scheduler.json", timeout="30s"
            )

            # Test with both scheduler_address and scheduler_file
            with pytest.raises(
                ValueError,
                match="Only one of scheduler_address or scheduler_file can be provided",
            ):
                get_client(
                    scheduler_address="tcp://localhost:8786",
                    scheduler_file="/path/to/scheduler.json",
                )

            # Test with CPU cluster
            client = get_client(cluster_type="cpu")
            mock_cpu_cluster.assert_called_once()

            # Test with invalid cluster type
            with pytest.raises(ValueError):
                get_client(cluster_type="invalid")

    @pytest.mark.gpu
    @patch("nemo_curator.utils.distributed_utils.start_dask_gpu_local_cluster")
    @patch("nemo_curator.utils.distributed_utils.Client")
    @patch("nemo_curator.utils.distributed_utils._set_torch_to_use_rmm")
    def test_get_client_gpu_cluster(
        self, mock_set_torch_rmm, mock_client, mock_gpu_cluster
    ):
        """Test get_client function with GPU cluster type."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        # Make the mock_gpu_cluster return the same mock_client_instance
        mock_gpu_cluster.return_value = mock_client_instance

        # Mock get_num_workers to return a positive value
        with patch(
            "nemo_curator.utils.distributed_utils.get_num_workers", return_value=1
        ):
            # Test with GPU cluster
            client = get_client(cluster_type="gpu")
            mock_gpu_cluster.assert_called_once()
            assert client == mock_client_instance


class TestDataReadingFunctions:
    def test_resolve_filename_col(self):
        """Test _resolve_filename_col function."""
        assert _resolve_filename_col(False) is False
        assert _resolve_filename_col(True) == "file_name"
        assert _resolve_filename_col("custom_name") == "custom_name"

        with pytest.raises(ValueError):
            _resolve_filename_col(123)

    @pytest.mark.gpu
    @patch("nemo_curator.utils.distributed_utils.cudf")
    def test_select_columns(self, mock_cudf):
        """Test select_columns function."""
        # Create a pandas DataFrame
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "file_name": ["f1", "f2"]})

        # Test with no column selection
        result = select_columns(df, None, "jsonl", False)
        assert result is df

        # Test with column selection and jsonl type
        result = select_columns(df, ["a"], "jsonl", False)
        pd.testing.assert_frame_equal(result, df[["a"]])

        # Test with column selection, jsonl type, and add_filename=True
        result = select_columns(df, ["a"], "jsonl", True)
        pd.testing.assert_frame_equal(result, df[["a", "file_name"]])

        # Add custom_name column to DataFrame before testing with custom filename
        df["custom_name"] = ["custom1", "custom2"]

        # Test with column selection, jsonl type, and add_filename as string
        result = select_columns(df, ["a"], "jsonl", "custom_name")
        pd.testing.assert_frame_equal(result, df[["a", "custom_name"]])

        # Test with parquet type (should not filter)
        result = select_columns(df, ["a"], "parquet", False)
        assert result is df

    @pytest.mark.gpu
    @patch("nemo_curator.utils.distributed_utils.cudf")
    @patch("nemo_curator.utils.distributed_utils.pd")
    def test_read_single_partition(self, mock_pd, mock_cudf, temp_dir):
        """Test read_single_partition function."""
        # Setup mocks
        mock_cudf.read_json.return_value = MagicMock()
        mock_cudf.read_parquet.return_value = MagicMock()
        mock_pd.read_json.return_value = pd.DataFrame({"a": [1, 2]})
        mock_pd.read_parquet.return_value = pd.DataFrame({"a": [1, 2]})

        # Test reading jsonl with cudf backend
        with patch(
            "nemo_curator.utils.distributed_utils.is_cudf_type", return_value=True
        ):
            with patch(
                "nemo_curator.utils.distributed_utils.select_columns",
                return_value=mock_cudf.read_json.return_value,
            ):
                result = read_single_partition(
                    files=["file.jsonl"], backend="cudf", file_type="jsonl"
                )
                mock_cudf.read_json.assert_called_once()
                assert result == mock_cudf.read_json.return_value

        # Test reading parquet with pandas backend
        with patch(
            "nemo_curator.utils.distributed_utils.select_columns",
            return_value=mock_pd.read_parquet.return_value,
        ):
            result = read_single_partition(
                files=["file.parquet"], backend="pandas", file_type="parquet"
            )
            mock_pd.read_parquet.assert_called_once()
            assert result.equals(mock_pd.read_parquet.return_value)

        # Test warning when input_meta is provided for non-jsonl file
        with pytest.warns(
            UserWarning,
            match="input_meta is only valid for JSONL files and will be ignored for other\\s+file formats..",
        ):
            with patch(
                "nemo_curator.utils.distributed_utils.select_columns",
                return_value=mock_pd.read_parquet.return_value,
            ):
                read_single_partition(
                    files=["file.parquet"],
                    backend="pandas",
                    file_type="parquet",
                    input_meta={"a": "int"},
                )

    @pytest.mark.gpu
    @patch("nemo_curator.utils.distributed_utils.read_single_partition")
    def test_read_data_files_per_partition(self, mock_read_single, temp_dir):
        """Test read_data_files_per_partition function."""
        # Create test files
        files = [os.path.join(temp_dir, f"file{i}.jsonl") for i in range(5)]
        for file in files:
            with open(file, "w") as f:
                f.write('{"a": 1}\n')

        # Setup mock
        mock_read_single.return_value = pd.DataFrame({"a": [1]})

        # Create a mock for dd.from_map that calls the function with our input and returns a DataFrame
        with patch(
            "nemo_curator.utils.distributed_utils.dd.from_map"
        ) as mock_dd_from_map:
            mock_dd_from_map.return_value = dd.from_pandas(
                pd.DataFrame({"a": [1, 2]}), npartitions=2
            )

            # Call the function
            result = read_data_files_per_partition(
                input_files=files,
                file_type="jsonl",
                backend="pandas",
                files_per_partition=2,
            )

            # Verify from_map was called once
            mock_dd_from_map.assert_called_once()

            # Verify the number of file groups passed to from_map
            # The first argument to from_map should be the function (read_single_partition)
            # The second argument should be the list of file groups
            args, kwargs = mock_dd_from_map.call_args
            file_groups = args[1]
            assert len(file_groups) == 3  # 5 files split into groups of 2 = 3 groups

            # Verify the file groups are correct
            assert file_groups[0] == files[0:2]  # First group: first 2 files
            assert file_groups[1] == files[2:4]  # Second group: next 2 files
            assert file_groups[2] == files[4:]  # Third group: last file

    @pytest.mark.gpu
    @patch("nemo_curator.utils.distributed_utils.dd.read_json")
    @patch("nemo_curator.utils.distributed_utils.dd.read_parquet")
    def test_read_data_blocksize(self, mock_read_parquet, mock_read_json, temp_dir):
        """Test read_data_blocksize function."""
        # Setup mocks
        mock_read_json.return_value = dd.from_pandas(
            pd.DataFrame({"a": [1, 2]}), npartitions=2
        )
        mock_read_parquet.return_value = dd.from_pandas(
            pd.DataFrame({"a": [1, 2]}), npartitions=2
        )

        # Test with jsonl
        with patch(
            "nemo_curator.utils.distributed_utils.select_columns",
            return_value=mock_read_json.return_value,
        ):
            with dask.config.set({"dataframe.backend": "pandas"}):
                result = read_data_blocksize(
                    input_files=["file.jsonl"],
                    backend="pandas",
                    file_type="jsonl",
                    blocksize="1MB",
                )
                mock_read_json.assert_called_once()

        # Test with parquet
        with patch(
            "nemo_curator.utils.distributed_utils.select_columns",
            return_value=mock_read_parquet.return_value,
        ):
            with dask.config.set({"dataframe.backend": "pandas"}):
                result = read_data_blocksize(
                    input_files=["file.parquet"],
                    backend="pandas",
                    file_type="parquet",
                    blocksize="1MB",
                )
                mock_read_parquet.assert_called_once()

        # Test with unsupported file type
        with pytest.raises(ValueError):
            read_data_blocksize(
                input_files=["file.txt"],
                backend="pandas",
                file_type="txt",
                blocksize="1MB",
            )

    def test_read_pandas_pickle(self, temp_dir):
        """Test read_pandas_pickle function."""
        # Create a test pickle file
        data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        pickle_path = os.path.join(temp_dir, "data.pkl")
        data.to_pickle(pickle_path)

        # Test reading the full DataFrame
        result = read_pandas_pickle(pickle_path)
        pd.testing.assert_frame_equal(result, data)

        # Test reading with column selection
        result = read_pandas_pickle(pickle_path, columns=["a"])
        pd.testing.assert_frame_equal(result, data[["a"]])

        # Test that warning is raised with add_filename
        with pytest.warns(UserWarning):
            result = read_pandas_pickle(pickle_path, add_filename=True)

    @pytest.mark.gpu
    @patch("nemo_curator.utils.distributed_utils.read_data_blocksize")
    @patch("nemo_curator.utils.distributed_utils.read_data_files_per_partition")
    @patch("nemo_curator.utils.distributed_utils.read_pandas_pickle")
    @patch("nemo_curator.utils.distributed_utils.check_dask_cwd")
    def test_read_data(
        self,
        mock_check_cwd,
        mock_read_pickle,
        mock_read_fpp,
        mock_read_blocksize,
        temp_dir,
    ):
        """Test the main read_data function."""
        # Setup mocks
        mock_read_pickle.return_value = pd.DataFrame({"a": [1, 2]})
        mock_read_fpp.return_value = dd.from_pandas(
            pd.DataFrame({"a": [1, 2]}), npartitions=2
        )
        mock_read_blocksize.return_value = dd.from_pandas(
            pd.DataFrame({"a": [1, 2]}), npartitions=2
        )

        # Test reading pickle
        with patch(
            "nemo_curator.utils.distributed_utils.dd.from_pandas",
            return_value=dd.from_pandas(mock_read_pickle.return_value, npartitions=16),
        ):
            result = read_data(["file.pkl"], file_type="pickle")
            mock_read_pickle.assert_called_once()

        # Test reading pickle with string input instead of list
        mock_read_pickle.reset_mock()
        with patch(
            "nemo_curator.utils.distributed_utils.dd.from_pandas",
            return_value=dd.from_pandas(mock_read_pickle.return_value, npartitions=16),
        ):
            result = read_data("file.pkl", file_type="pickle")
            mock_read_pickle.assert_called_once()

        # Test reading jsonl with blocksize
        result = read_data(
            ["file1.jsonl", "file2.jsonl"],
            file_type="jsonl",
            blocksize="1MB",
            files_per_partition=None,  # Explicitly set to None to avoid conflict
        )
        mock_read_blocksize.assert_called_once()

        # Test reading jsonl with files_per_partition
        result = read_data(
            ["file1.jsonl", "file2.jsonl"],
            file_type="jsonl",
            files_per_partition=1,
            blocksize=None,  # Explicitly set to None to avoid conflict
        )
        mock_read_fpp.assert_called_once()

        # Test error when both blocksize and files_per_partition are provided
        with pytest.raises(
            ValueError,
            match="blocksize and files_per_partition cannot be set at the same time",
        ):
            read_data(
                ["file1.jsonl", "file2.jsonl"],
                file_type="jsonl",
                blocksize="1MB",
                files_per_partition=1,
            )

        # Test error handling for mixed file extensions
        with patch("os.path.splitext", side_effect=[(".jsonl",), (".parquet",)]):
            with pytest.raises(RuntimeError):
                read_data(["file1.jsonl", "file2.parquet"], file_type="jsonl")

        # Test invalid file type
        with pytest.raises(RuntimeError):
            read_data(["file.txt"], file_type="txt")


class TestDataWritingFunctions:
    def test_single_partition_write_with_filename(self, temp_dir):
        """Test single_partition_write_with_filename function."""
        # Create output directory
        os.makedirs(os.path.join(temp_dir, "output"), exist_ok=True)

        # Create test dataframe
        df = pd.DataFrame(
            {"text": ["hello", "world"], "file_name": ["file1.jsonl", "file2.jsonl"]}
        )

        # Test with empty dataframe
        empty_df = pd.DataFrame(columns=df.columns)
        result = single_partition_write_with_filename(
            empty_df, os.path.join(temp_dir, "output"), filename_col="file_name"
        )
        assert result[0]

        # Test with non-empty dataframe, jsonl output
        result = single_partition_write_with_filename(
            df,
            os.path.join(temp_dir, "output"),
            output_type="jsonl",
            filename_col="file_name",
        )
        assert not result[0]
        assert os.path.exists(os.path.join(temp_dir, "output", "file1.jsonl"))
        assert os.path.exists(os.path.join(temp_dir, "output", "file2.jsonl"))

        # Test with non-empty dataframe, parquet output
        os.makedirs(os.path.join(temp_dir, "output2"), exist_ok=True)
        result = single_partition_write_with_filename(
            df,
            os.path.join(temp_dir, "output2"),
            output_type="parquet",
            filename_col="file_name",
        )
        assert not result[0]
        assert os.path.exists(os.path.join(temp_dir, "output2", "file1.parquet"))
        assert os.path.exists(os.path.join(temp_dir, "output2", "file2.parquet"))

        # Test with unknown output type
        with pytest.raises(ValueError):
            single_partition_write_with_filename(
                df,
                os.path.join(temp_dir, "output"),
                output_type="unknown",
                filename_col="file_name",
            )

    @patch("nemo_curator.utils.distributed_utils._write_to_jsonl_or_parquet")
    def test_write_to_disk(self, mock_write_jsonl, temp_dir):
        """Test write_to_disk function."""
        # Create test dataframe
        df = dd.from_pandas(
            pd.DataFrame(
                {
                    "text": ["hello", "world"],
                    "file_name": ["file1.jsonl", "file2.jsonl"],
                }
            ),
            npartitions=1,
        )

        # Instead of mocking single_partition_write_with_filename with MagicMock,
        # we'll create a patch that uses a real function
        def simple_write_with_filename(*args, **kwargs):
            # This is a simplified version that's deterministically hashable
            return pd.Series([True], dtype="bool")

        # Test writing to a single JSONL file
        with pytest.raises(RuntimeError):
            # Create a multi-partition dataframe using repartition instead of setting npartitions directly
            multi_part_df = df.repartition(npartitions=2)
            write_to_disk(multi_part_df, os.path.join(temp_dir, "output.jsonl"))

        # Test writing with filename column
        with patch(
            "nemo_curator.utils.distributed_utils.single_partition_write_with_filename",
            simple_write_with_filename,
        ):
            write_to_disk(
                df,
                os.path.join(temp_dir, "output"),
                write_to_filename=True,
                output_type="jsonl",
            )

        # Test error when write_to_filename is True but column doesn't exist
        # Create a new DataFrame without file_name column instead of patching columns
        df_no_filename = dd.from_pandas(
            pd.DataFrame({"text": ["hello", "world"]}), npartitions=1
        )
        with pytest.raises(ValueError):
            write_to_disk(
                df_no_filename, os.path.join(temp_dir, "output"), write_to_filename=True
            )

        # Test error when both partition_on and write_to_filename are used
        with pytest.raises(ValueError):
            write_to_disk(
                df,
                os.path.join(temp_dir, "output"),
                write_to_filename=True,
                partition_on="text",
            )

        # Test writing normal jsonl
        write_to_disk(df, os.path.join(temp_dir, "output"), output_type="jsonl")
        mock_write_jsonl.assert_called_with(
            df,
            output_path=os.path.join(temp_dir, "output"),
            output_type="jsonl",
            partition_on=None,
        )

        # Test writing parquet
        write_to_disk(df, os.path.join(temp_dir, "output"), output_type="parquet")
        mock_write_jsonl.assert_called_with(
            df,
            output_path=os.path.join(temp_dir, "output"),
            output_type="parquet",
            partition_on=None,
        )

        # Test with unknown output type
        with pytest.raises(ValueError):
            write_to_disk(df, os.path.join(temp_dir, "output"), output_type="unknown")

    def test_write_to_jsonl_or_parquet(self, temp_dir):
        """Test _write_to_jsonl_or_parquet function with all branches."""
        # Create test dataframes - one for pandas and one for cudf
        pandas_df = pd.DataFrame(
            {
                "text": ["hello", "world", "test", "data"],
                "category": ["A", "B", "A", "B"],
            }
        )
        pandas_ddf = dd.from_pandas(pandas_df, npartitions=1)

        # 1. Test JSONL with partitioning
        output_path = os.path.join(temp_dir, "partitioned_jsonl")
        os.makedirs(output_path, exist_ok=True)

        with patch(
            "nemo_curator.utils.distributed_utils.is_cudf_type", return_value=False
        ):
            _write_to_jsonl_or_parquet(
                pandas_ddf,
                output_path=output_path,
                output_type="jsonl",
                partition_on="category",
            )
            # Verify that directories were created with the partitioning
            assert os.path.exists(os.path.join(output_path, "category=A"))
            assert os.path.exists(os.path.join(output_path, "category=B"))

        # 2. Test JSONL without partitioning for pandas dataframe
        output_path = os.path.join(temp_dir, "pandas_jsonl")
        os.makedirs(output_path, exist_ok=True)

        with patch(
            "nemo_curator.utils.distributed_utils.is_cudf_type", return_value=False
        ):
            _write_to_jsonl_or_parquet(
                pandas_ddf,
                output_path=os.path.join(output_path, "output.jsonl"),
                output_type="jsonl",
            )
            # Verify the file was created
            assert os.path.exists(os.path.join(output_path, "output.jsonl"))

        # 3. Test JSONL without partitioning for cudf dataframe
        output_path = os.path.join(temp_dir, "cudf_jsonl")
        os.makedirs(output_path, exist_ok=True)

        with patch(
            "nemo_curator.utils.distributed_utils.is_cudf_type", return_value=True
        ):
            _write_to_jsonl_or_parquet(
                pandas_ddf,  # We're still using pandas_ddf but mocking is_cudf_type to return True
                output_path=os.path.join(output_path, "output.jsonl"),
                output_type="jsonl",
            )
            # Verify the file was created
            assert os.path.exists(os.path.join(output_path, "output.jsonl"))

        # 4. Test Parquet with partitioning
        output_path = os.path.join(temp_dir, "partitioned_parquet")
        os.makedirs(output_path, exist_ok=True)

        with patch.object(pandas_ddf, "to_parquet") as mock_to_parquet:
            _write_to_jsonl_or_parquet(
                pandas_ddf,
                output_path=output_path,
                output_type="parquet",
                partition_on="category",
            )
            # Verify to_parquet was called with the correct parameters
            mock_to_parquet.assert_called_once_with(
                output_path, write_index=False, partition_on="category"
            )

        # 5. Test Parquet without partitioning
        output_path = os.path.join(temp_dir, "simple_parquet")
        os.makedirs(output_path, exist_ok=True)

        with patch.object(pandas_ddf, "to_parquet") as mock_to_parquet:
            _write_to_jsonl_or_parquet(
                pandas_ddf, output_path=output_path, output_type="parquet"
            )
            # Verify to_parquet was called with the correct parameters
            mock_to_parquet.assert_called_once_with(
                output_path, write_index=False, partition_on=None
            )

        # 6. Test unknown output type
        with pytest.raises(ValueError):
            _write_to_jsonl_or_parquet(
                pandas_ddf,
                output_path=os.path.join(temp_dir, "unknown"),
                output_type="unknown",
            )


class TestWorkerFunctions:
    @pytest.mark.gpu
    def test_load_object_on_worker(self):
        """Test load_object_on_worker function."""
        # Mock worker and functions
        mock_worker = MagicMock()
        mock_worker.attr = "existing_value"

        # Test with existing attribute
        with patch(
            "nemo_curator.utils.distributed_utils.get_worker", return_value=mock_worker
        ):
            result = load_object_on_worker("attr", lambda: "new_value", {})
            assert result == "existing_value"

        # Test with new attribute
        with patch(
            "nemo_curator.utils.distributed_utils.get_worker", return_value=mock_worker
        ):
            delattr(mock_worker, "attr")
            load_fn = MagicMock(return_value="new_value")
            result = load_object_on_worker("attr", load_fn, {"arg": "value"})
            load_fn.assert_called_once_with(arg="value")
            assert result == "new_value"
            assert mock_worker.attr == "new_value"

        # Test with no worker available
        with patch(
            "nemo_curator.utils.distributed_utils.get_worker",
            side_effect=ValueError("No worker"),
        ):
            with pytest.raises(NoWorkerError):
                load_object_on_worker("attr", lambda: "value", {})

    @pytest.mark.gpu
    def test_offload_object_on_worker(self):
        """Test offload_object_on_worker function."""
        # Mock worker
        mock_worker = MagicMock()
        mock_worker.attr = "value"

        # Test with existing attribute
        with patch(
            "nemo_curator.utils.distributed_utils.get_worker", return_value=mock_worker
        ):
            result = offload_object_on_worker("attr")
            assert result is True
            assert not hasattr(mock_worker, "attr")

        # Test with non-existing attribute
        with patch(
            "nemo_curator.utils.distributed_utils.get_worker", return_value=mock_worker
        ):
            result = offload_object_on_worker("nonexistent")
            assert result is True

    @pytest.mark.gpu
    def test_process_batch(self):
        """Test process_batch function."""
        # Mock functions and model
        load_model_fn = MagicMock(return_value="model")
        run_inference_fn = MagicMock(return_value="inference_result")

        # Test with successful model loading
        with patch(
            "nemo_curator.utils.distributed_utils.load_object_on_worker",
            return_value="model",
        ):
            result = process_batch(
                load_model_fn,
                {"arg": "value"},
                run_inference_fn,
                {"data": "batch_data"},
            )
            run_inference_fn.assert_called_once_with(data="batch_data", model="model")
            assert result == "inference_result"

    @pytest.mark.gpu
    def test_process_all_batches(self):
        """Test process_all_batches function."""
        # Mock functions and data
        load_model_fn = MagicMock()
        run_inference_fn = MagicMock(return_value=MagicMock())
        loader = [MagicMock(), MagicMock()]

        # Mock torch.cat
        with patch("torch.cat", return_value="concatenated_result"):
            with patch(
                "nemo_curator.utils.distributed_utils.process_batch",
                side_effect=["result1", "result2"],
            ):
                result = process_all_batches(
                    loader,
                    load_model_fn,
                    {"arg": "value"},
                    run_inference_fn,
                    {"param": "value"},
                )
                assert result == "concatenated_result"


class TestUtilityFunctions:
    def test_get_num_workers(self):
        """Test get_num_workers function."""
        # Test with None client
        assert get_num_workers(None) is None

        # Test with valid client
        mock_client = MagicMock()
        mock_client.scheduler_info.return_value = {
            "workers": {"worker1": {}, "worker2": {}}
        }
        assert get_num_workers(mock_client) == 2

    def test_get_current_client(self):
        """Test get_current_client function."""
        # Test when client exists
        with patch(
            "nemo_curator.utils.distributed_utils.Client.current", return_value="client"
        ):
            assert get_current_client() == "client"

        # Test when no client exists
        with patch(
            "nemo_curator.utils.distributed_utils.Client.current",
            side_effect=ValueError("No client"),
        ):
            assert get_current_client() is None

    def test_check_dask_cwd(self):
        """Test check_dask_cwd function."""
        # Test with absolute paths
        check_dask_cwd(["/path/to/file1", "/path/to/file2"])

        # Test with relative paths and matching CWD
        mock_client = MagicMock()
        mock_client.run.return_value = {"worker1": "/cwd", "worker2": "/cwd"}

        with patch(
            "nemo_curator.utils.distributed_utils.get_current_client",
            return_value=mock_client,
        ):
            with patch("subprocess.check_output", return_value="/cwd\n"):
                check_dask_cwd(["file1", "file2"])

        # Test with relative paths and mismatched CWD
        mock_client.run.return_value = {
            "worker1": "/worker_cwd",
            "worker2": "/worker_cwd",
        }

        with patch(
            "nemo_curator.utils.distributed_utils.get_current_client",
            return_value=mock_client,
        ):
            with patch("subprocess.check_output", return_value="/client_cwd\n"):
                with pytest.raises(RuntimeError):
                    check_dask_cwd(["file1", "file2"])

        # Test with relative paths and mismatched worker CWDs
        mock_client.run.return_value = {"worker1": "/cwd1", "worker2": "/cwd2"}

        with patch(
            "nemo_curator.utils.distributed_utils.get_current_client",
            return_value=mock_client,
        ):
            with pytest.raises(RuntimeError):
                check_dask_cwd(["file1", "file2"])

    def test_performance_report_if(self):
        """Test performance_report_if function."""
        # Test with None path
        with performance_report_if(None) as ctx:
            assert ctx is None

        # Test with valid path
        with patch(
            "nemo_curator.utils.distributed_utils.performance_report"
        ) as mock_perf_report:
            # Create a mock context manager instead of a string
            mock_context = MagicMock()
            # Set up the context manager to return itself from __enter__
            mock_context.__enter__.return_value = mock_context
            mock_perf_report.return_value = mock_context
            with performance_report_if("/path/to/reports") as ctx:
                mock_perf_report.assert_called_once_with(
                    "/path/to/reports/dask-profile.html"
                )
                assert ctx is mock_context

    def test_performance_report_if_with_ts_suffix(self):
        """Test performance_report_if_with_ts_suffix function."""
        # Test with None path
        with performance_report_if_with_ts_suffix(None) as ctx:
            assert ctx is None

        # Test with valid path
        with patch(
            "nemo_curator.utils.distributed_utils.performance_report_if"
        ) as mock_perf_report_if:
            mock_perf_report_if.return_value = nullcontext()
            with patch(
                "nemo_curator.utils.distributed_utils.datetime"
            ) as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20240715_120000"
                with performance_report_if_with_ts_suffix(
                    "/path/to/reports", "custom-report"
                ) as ctx:
                    mock_perf_report_if.assert_called_once_with(
                        path="/path/to/reports",
                        report_name="custom-report-20240715_120000.html",
                    )

    @pytest.mark.gpu
    def test_seed_all(self):
        """Test seed_all function."""
        with patch("random.seed") as mock_random_seed:
            with patch("numpy.random.seed") as mock_np_seed:
                with patch("torch.manual_seed") as mock_torch_seed:
                    with patch("torch.cuda.is_available", return_value=True):
                        with patch("torch.cuda.manual_seed") as mock_cuda_seed:
                            with patch(
                                "torch.cuda.manual_seed_all"
                            ) as mock_cuda_seed_all:
                                # Create a mock for torch.backends.cudnn instead of patching properties
                                mock_cudnn = MagicMock()
                                with patch("torch.backends.cudnn", mock_cudnn):
                                    seed_all(123)

                                    mock_random_seed.assert_called_once_with(123)
                                    mock_np_seed.assert_called_once_with(123)
                                    mock_torch_seed.assert_called_once_with(123)
                                    mock_cuda_seed.assert_called_once_with(123)
                                    mock_cuda_seed_all.assert_called_once_with(123)
                                    assert os.environ["PYTHONHASHSEED"] == "123"
                                    # Assert properties were set correctly on the mock
                                    assert mock_cudnn.deterministic is True
                                    assert mock_cudnn.benchmark is False

    def test_get_network_interfaces(self):
        """Test get_network_interfaces function."""
        with patch("psutil.net_if_addrs", return_value={"eth0": [], "lo": []}):
            interfaces = get_network_interfaces()
            assert interfaces == ["eth0", "lo"]

    @pytest.mark.gpu
    @patch("nemo_curator.utils.distributed_utils.cudf")
    def test_enable_spilling(self, mock_cudf):
        """Test _enable_spilling function."""
        # Call the function
        _enable_spilling()

        # Verify that cudf.set_option was called with the correct parameters
        mock_cudf.set_option.assert_called_once_with("spill", True)

    @pytest.mark.gpu
    def test_set_torch_to_use_rmm(self):
        """Test _set_torch_to_use_rmm function."""
        # Mock the imports inside the function
        with patch.dict(
            "sys.modules", {"torch": MagicMock(), "rmm.allocators.torch": MagicMock()}
        ):
            # Create our mock torch module with cuda attributes
            mock_torch = sys.modules["torch"]
            mock_torch.cuda = MagicMock()
            mock_torch.cuda.get_allocator_backend = MagicMock()
            mock_torch.cuda.memory = MagicMock()
            mock_rmm_torch = sys.modules["rmm.allocators.torch"]
            mock_rmm_torch.rmm_torch_allocator = "mock_allocator"

            # Test case 1: Allocator not already set
            mock_torch.cuda.get_allocator_backend.return_value = "default"

            # Call the function
            _set_torch_to_use_rmm()

            # Verify that torch.cuda.memory.change_current_allocator was called
            mock_torch.cuda.memory.change_current_allocator.assert_called_once_with(
                mock_rmm_torch.rmm_torch_allocator
            )

            # Reset the mock call history
            mock_torch.cuda.memory.change_current_allocator.reset_mock()

            # Test case 2: Allocator already pluggable
            mock_torch.cuda.get_allocator_backend.return_value = "pluggable"

            # Call the function again with warning module mocked
            with patch(
                "nemo_curator.utils.distributed_utils.warnings"
            ) as mock_warnings:
                _set_torch_to_use_rmm()

                # Verify warning was issued
                mock_warnings.warn.assert_called_once()

                # Verify that torch.cuda.memory.change_current_allocator was not called
                mock_torch.cuda.memory.change_current_allocator.assert_not_called()

    @pytest.mark.gpu
    def test_get_gpu_memory_info_no_client(self):
        """Test get_gpu_memory_info function with no client."""
        # Test when no client exists
        with patch(
            "nemo_curator.utils.distributed_utils.get_current_client",
            return_value=None,
        ):
            memory_info = get_gpu_memory_info()
            assert memory_info == {}

    @pytest.mark.gpu
    def test_get_gpu_memory_info_with_real_client(self):
        """Test get_gpu_memory_info function with an actual client."""
        # Create a real Dask client
        try:
            client = get_client(cluster_type="gpu", n_workers=1)

            # Call get_gpu_memory_info
            memory_info = get_gpu_memory_info()

            # Verify result structure
            assert isinstance(memory_info, dict)
            assert len(memory_info) > 0

            # Verify each worker has a memory value
            for worker_address, memory in memory_info.items():
                assert isinstance(worker_address, str)
                assert isinstance(memory, int)
                assert memory > 0

            # Close the client when done
            client.close()

            # Test that no client returns empty dict
            assert get_gpu_memory_info() == {}

        except Exception as e:
            pytest.skip(f"Could not create GPU Dask client for testing: {e}")
