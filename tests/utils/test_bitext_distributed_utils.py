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
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nemo_curator.utils.distributed_utils import (
    _merge_tmp_simple_bitext_partitions,
    _single_partition_write_to_simple_bitext,
    write_to_disk,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestBitextWritingFunctions:
    """Tests for bitext writing functions in distributed_utils.py."""

    def test_single_partition_write_to_simple_bitext_nonempty(self, temp_dir):
        """Test _single_partition_write_to_simple_bitext with non-empty DataFrame."""
        # Create test DataFrame
        df = pd.DataFrame(
            {
                "src": ["Hello world", "How are you?", "Test sentence"],
                "tgt": ["Hola mundo", "¿Cómo estás?", "Frase de prueba"],
                "src_lang": ["en", "en", "en"],
                "tgt_lang": ["es", "es", "es"],
            }
        )

        # Call the function
        output_file = os.path.join(temp_dir, "test_output")
        result = _single_partition_write_to_simple_bitext(df, output_file)

        # Check return value (should indicate non-empty partition)
        assert not result.iloc[0]

        # Check that the output files were created
        src_file = f"{output_file}.en.0"
        tgt_file = f"{output_file}.es.0"
        assert os.path.exists(src_file)
        assert os.path.exists(tgt_file)

        # Check file contents
        with open(src_file, "r") as f:
            src_lines = f.read().strip().split("\n")
            assert src_lines == ["Hello world", "How are you?", "Test sentence"]

        with open(tgt_file, "r") as f:
            tgt_lines = f.read().strip().split("\n")
            assert tgt_lines == ["Hola mundo", "¿Cómo estás?", "Frase de prueba"]

    def test_single_partition_write_to_simple_bitext_empty(self, temp_dir):
        """Test _single_partition_write_to_simple_bitext with empty DataFrame."""
        # Create empty test DataFrame with correct columns
        df = pd.DataFrame({"src": [], "tgt": [], "src_lang": [], "tgt_lang": []})

        # Add a dummy value to src_lang/tgt_lang to avoid IndexError in function
        df.loc[0] = ["", "", "en", "es"]
        df = df.iloc[0:0]  # Remove the row but keep the dtypes

        # Call the function with warning capture
        output_file = os.path.join(temp_dir, "test_output")
        with pytest.warns(UserWarning, match="Empty partition found"):
            result = _single_partition_write_to_simple_bitext(df, output_file)

        # Check return value (should indicate empty partition)
        assert result.iloc[0]

    @pytest.mark.gpu
    def test_single_partition_write_to_simple_bitext_with_cudf(self, temp_dir):
        """Test _single_partition_write_to_simple_bitext with cuDF DataFrame."""
        # Skip if cuDF is not available
        cudf = pytest.importorskip("cudf")

        # Create test DataFrame
        df = cudf.DataFrame(
            {
                "src": ["Hello world", "Test sentence"],
                "tgt": ["Hola mundo", "Frase de prueba"],
                "src_lang": ["en", "en"],
                "tgt_lang": ["es", "es"],
            }
        )

        # Call the function
        output_file = os.path.join(temp_dir, "test_output")
        result = _single_partition_write_to_simple_bitext(df, output_file)

        # Check return value (should indicate non-empty partition)
        assert not result.iloc[0]

        # Check that the output files were created
        src_file = f"{output_file}.en.0"
        tgt_file = f"{output_file}.es.0"
        assert os.path.exists(src_file)
        assert os.path.exists(tgt_file)

        # Check file contents
        with open(src_file, "r") as f:
            src_lines = f.read().strip().split("\n")
            assert src_lines == ["Hello world", "Test sentence"]

        with open(tgt_file, "r") as f:
            tgt_lines = f.read().strip().split("\n")
            assert tgt_lines == ["Hola mundo", "Frase de prueba"]

    def test_merge_tmp_simple_bitext_partitions(self, temp_dir):
        """Test _merge_tmp_simple_bitext_partitions function."""
        # Create temporary input and output directories
        tmp_dir = os.path.join(temp_dir, "tmp")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create test files in temporary directory
        en_files = ["test.en.0", "test.en.1", "test.en.2"]
        es_files = ["test.es.0", "test.es.1", "test.es.2"]

        # File contents
        en_contents = ["Hello world\n", "How are you?\n", "Test sentence\n"]

        es_contents = ["Hola mundo\n", "¿Cómo estás?\n", "Frase de prueba\n"]

        # Create the files
        for i, (en_file, es_file) in enumerate(zip(en_files, es_files)):
            with open(os.path.join(tmp_dir, en_file), "w") as f:
                f.write(en_contents[i])
            with open(os.path.join(tmp_dir, es_file), "w") as f:
                f.write(es_contents[i])

        # Call the function
        _merge_tmp_simple_bitext_partitions(tmp_dir, output_dir)

        # Check that merged files were created
        assert os.path.exists(os.path.join(output_dir, "test.en"))
        assert os.path.exists(os.path.join(output_dir, "test.es"))

        # Check merged file contents
        with open(os.path.join(output_dir, "test.en"), "r") as f:
            content = f.read()
            assert content == "".join(en_contents)

        with open(os.path.join(output_dir, "test.es"), "r") as f:
            content = f.read()
            assert content == "".join(es_contents)

    def test_write_to_disk_bitext(self, temp_dir):
        """Test write_to_disk function with bitext output_type."""
        # Create test DataFrame
        df = pd.DataFrame(
            {
                "src": ["Hello world", "How are you?", "Test sentence"],
                "tgt": ["Hola mundo", "¿Cómo estás?", "Frase de prueba"],
                "src_lang": ["en", "en", "en"],
                "tgt_lang": ["es", "es", "es"],
                "file_name": ["test.txt", "test.txt", "test.txt"],
            }
        )

        # Convert to Dask DataFrame
        import dask.dataframe as dd

        ddf = dd.from_pandas(df, npartitions=1)

        # Define a simple deterministic function instead of using MagicMock
        def simple_write_function(*args, **kwargs):
            # Return a pandas Series that can be deterministically hashed
            return pd.Series([False], dtype="bool")

        # Mock the necessary functions to avoid actual computation
        with patch(
            "nemo_curator.utils.distributed_utils._single_partition_write_to_simple_bitext",
            simple_write_function,
        ):
            with patch(
                "nemo_curator.utils.distributed_utils._merge_tmp_simple_bitext_partitions"
            ) as mock_merge:
                with patch("shutil.rmtree") as mock_rmtree:
                    # Path for normal bitext write without filename
                    output_path = os.path.join(temp_dir, "output.bitext")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    # Call the function
                    write_to_disk(ddf, output_path, output_type="bitext")

                    # Check that functions were called correctly
                    # We can't assert on simple_write_function since it's not a mock
                    mock_merge.assert_called_once()
                    mock_rmtree.assert_called_once()

    def test_write_to_disk_bitext_with_filename(self, temp_dir):
        """Test write_to_disk function with bitext output_type and write_to_filename=True."""
        # Create test DataFrame
        df = pd.DataFrame(
            {
                "src": ["Hello world", "How are you?", "Test sentence"],
                "tgt": ["Hola mundo", "¿Cómo estás?", "Frase de prueba"],
                "src_lang": ["en", "en", "en"],
                "tgt_lang": ["es", "es", "es"],
                "file_name": ["test.txt", "test.txt", "test.txt"],
            }
        )

        # Convert to Dask DataFrame
        import dask.dataframe as dd

        ddf = dd.from_pandas(df, npartitions=1)

        # Define a deterministic function for patching instead of using MagicMock
        def simple_write_to_bitext(*args, **kwargs):
            # This is a simplified version that's deterministically hashable
            return pd.Series([False], dtype="bool")

        # Patch the functions with our deterministic function
        with patch(
            "nemo_curator.utils.distributed_utils._single_partition_write_to_simple_bitext",
            simple_write_to_bitext,
        ):
            with patch(
                "nemo_curator.utils.distributed_utils._merge_tmp_simple_bitext_partitions"
            ) as mock_merge:
                with patch("shutil.rmtree") as mock_rmtree:
                    # Path for bitext write with filename
                    output_dir = os.path.join(temp_dir, "output_dir")

                    # Call the function
                    write_to_disk(
                        ddf, output_dir, write_to_filename=True, output_type="bitext"
                    )

                    # Verify the function was called with the correct parameters
                    mock_merge.assert_called_once()
                    mock_rmtree.assert_called_once()

                    # Verify the first arg of mock_merge call is the tmp dir
                    mock_merge_args = mock_merge.call_args[0]
                    assert ".tmp" in mock_merge_args[0]
                    assert output_dir == mock_merge_args[1]
