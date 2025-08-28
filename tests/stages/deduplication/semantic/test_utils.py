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

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest


@pytest.mark.gpu
def test_get_array_from_df() -> None:
    import cudf
    import cupy as cp

    from nemo_curator.stages.deduplication.semantic.utils import get_array_from_df

    """Test that get_array_from_df works correctly."""
    df = cudf.DataFrame(
        {
            "embedding": [[3, 4, 5], [1, 2, 2], [1, 0, 0]],
        }
    )
    expected_array = cp.array(
        [
            [3, 4, 5],
            [1, 2, 2],
            [1, 0, 0],
        ]
    )
    result = get_array_from_df(df, "embedding")
    cp.testing.assert_allclose(result, expected_array, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu  # TODO : Remove this once we figure out how to import semantic on CPU
class TestBreakParquetPartitionIntoGroups:
    @patch("pyarrow.parquet.read_metadata", return_value=Mock(num_rows=10_000))
    @patch("nemo_curator.stages.deduplication.semantic.utils.open_parquet_file")
    def test_calculation_logic(self, mock_open_parquet: Mock, mock_read_metadata: Mock) -> None:
        from nemo_curator.stages.deduplication.semantic.utils import break_parquet_partition_into_groups

        """Test the calculation logic of break_parquet_partition_into_groups without actual files."""
        # Mock the parquet metadata to return a specific number of rows

        test_files = [f"mock_file_{i}.parquet" for i in range(1000)]
        # Test with embedding_dim=1000
        # Expected calculation:
        # - cudf_max_num_rows = 2_000_000_000
        # - cudf_max_num_elements = 2_000_000_000 / 1000 = 2_000_000
        # - avg_num_rows = 10_000 * 1.5 = 15_000
        # - max_files_per_subgroup = int(2_000_000 / 15_000) = 133
        # Since we have 1000 files and max_files_per_subgroup=133

        groups = break_parquet_partition_into_groups(test_files, embedding_dim=1000)

        # Verify the mock was called
        mock_open_parquet.assert_called_once()
        mock_read_metadata.assert_called_once()

        # With our calculation, all 10 files should fit in one group
        assert len(groups) == 8, "1000 files each with 10k rows with embedding_dim=1000 should fit in 8 groups"
        for i, group in enumerate(groups):
            if i != len(groups) - 1:
                assert len(group) == 133, f"Group {i} should contain 133 files"
            else:
                assert len(group) == 69, "Last group should contain fewer files"

    def test_small_files_no_break(self, tmp_path: Path) -> None:
        """Test that break_parquet_partition_into_groups correctly splits files to avoid cuDF 2bn row limit."""
        from nemo_curator.stages.deduplication.semantic.utils import break_parquet_partition_into_groups

        # Create test parquet files
        test_files = []
        for i in range(5):
            file_path = tmp_path / f"test_file_{i}.parquet"
            # Create a small test dataframe and save as parquet
            df = pd.DataFrame(
                {
                    "id": list(range(i * 10, (i + 1) * 10)),
                    "embedding": [[1.0, 2.0, 3.0]] * 10,
                }
            )
            df.to_parquet(file_path)
            test_files.append(str(file_path))

        # Test with default embedding dimension (1024)
        groups = break_parquet_partition_into_groups(test_files, embedding_dim=1024)

        # Verify that we get groups (should be all files in one group for small test data)
        assert len(groups) == 1, "Should create one group"
        # Verify all files are included
        all_files_in_group = list(groups[0])
        assert set(all_files_in_group) == set(test_files), "All input files should be included in groups"

    def test_large_files_break(self, tmp_path: Path) -> None:
        """Test break_parquet_partition_into_groups with large embedding dimension that forces multiple groups."""
        from nemo_curator.stages.deduplication.semantic.utils import break_parquet_partition_into_groups

        # Create test parquet files
        test_files = []
        num_rows, num_files = 1000, 10

        # Create 10 files, each with 1000 rows and 2000-dimensional embeddings
        # Each file contains: 1000 rows * 2000 dimensions = 2,000,000 elements
        for i in range(num_files):
            file_path = tmp_path / f"large_test_file_{i}.parquet"
            df = pd.DataFrame(
                {
                    "id": list(range(i * num_rows, (i + 1) * num_rows)),
                    "embedding": [[1.0] * 2000] * num_rows,  # 2000-dim embeddings
                }
            )
            df.to_parquet(file_path)
            test_files.append(str(file_path))

        # Test with embedding_dim=400,000 to force file splitting
        # This parameter tells the function how many dimensions each embedding has
        # The function uses this to calculate the effective row limit for cuDF

        # Calculation breakdown:
        # 1. cuDF max rows: 2,000,000,000 (2 billion)
        # 2. Effective max elements per group: 2,000,000,000 / 400,000 = 5,000
        # 3. Each file has 1000 rows, so with 1.5x safety factor: 1000 * 1.5 = 1,500 rows per file
        # 4. Max files per group: int(5,000 / 1,500) = int(3.33) = 3
        # 5. With 10 files and max 3 files per group: ceil(10 / 3) = 4 groups

        # Expected groups:
        # - Group 0: files 0, 1, 2 (3 files)
        # - Group 1: files 3, 4, 5 (3 files)
        # - Group 2: files 6, 7, 8 (3 files)
        # - Group 3: file 9 (1 file)

        groups = break_parquet_partition_into_groups(test_files, embedding_dim=400_000)

        # Verify we get exactly 4 groups as calculated above
        assert len(groups) == 4, "Should create 4 groups based on embedding_dim=400,000 calculation"
        for i, group in enumerate(groups):
            if i != len(groups) - 1:
                assert len(group) == 3, f"Group {i} should contain 3 files"
                assert set(group) == set(test_files[i * 3 : (i + 1) * 3]), (
                    f"Group {i} should contain files {i * 3} to {(i + 1) * 3}"
                )
            else:
                assert len(group) == 1, f"Group {i} should contain 1 file"
                assert set(group) == set(test_files[i * 3 : (i + 1) * 3]), (
                    f"Group {i} should contain files {i * 3} to {(i + 1) * 3}"
                )

        # If we run with the default value of embedding_dim=1024, we should get one group
        groups = break_parquet_partition_into_groups(test_files)
        assert len(groups) == 1, "Should create one group"
        assert set(groups[0]) == set(test_files), "All input files should be included in groups"
