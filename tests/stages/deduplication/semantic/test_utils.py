# modality: text

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


@pytest.mark.gpu
def test_chunked_parquet_reader_preserves_order(tmp_path: Path) -> None:
    import cudf

    from nemo_curator.stages.deduplication.semantic.utils import iter_parquet_chunks, read_parquet_file_info

    files = []
    for index in range(3):
        path = tmp_path / f"part-{index}.parquet"
        cudf.DataFrame({"id": range(index * 7, (index + 1) * 7), "value": [index] * 7}).to_parquet(path)
        files.append(str(path))
    file_info = read_parquet_file_info(files, retained_columns=["id"])

    chunks = list(
        iter_parquet_chunks(
            files,
            columns=["id"],
            footers=[info.footer for info in file_info],
            chunk_read_limit=64,
            pass_read_limit=64,
        )
    )

    result = cudf.concat(chunks, ignore_index=True)
    assert result["id"].to_arrow().to_pylist() == list(range(21))
    assert [info.num_rows for info in file_info] == [7, 7, 7]


@pytest.mark.gpu  # TODO : Remove this once we figure out how to import semantic on CPU
class TestBreakParquetPartitionIntoGroups:
    def test_calculation_logic(self) -> None:
        from nemo_curator.stages.deduplication.semantic.utils import (
            ParquetFileInfo,
            break_parquet_partition_into_groups,
        )

        """Test the calculation logic of break_parquet_partition_into_groups without actual files."""
        test_files = [f"mock_file_{i}.parquet" for i in range(1000)]
        file_info = [ParquetFileInfo(path, 10_000, 0) for path in test_files]
        groups = break_parquet_partition_into_groups(test_files, embedding_dim=1000, file_info=file_info)

        assert len(groups) == 5
        assert all(len(group) == 200 for group in groups)

    def test_uses_exact_counts_for_skewed_files(self) -> None:
        from nemo_curator.stages.deduplication.semantic.utils import (
            ParquetFileInfo,
            break_parquet_partition_into_groups,
        )

        files = ["large.parquet", "tiny-1.parquet", "tiny-2.parquet"]
        file_info = [
            ParquetFileInfo(files[0], 1_900, 0),
            ParquetFileInfo(files[1], 100, 0),
            ParquetFileInfo(files[2], 1, 0),
        ]

        groups = break_parquet_partition_into_groups(files, embedding_dim=1_000_000, file_info=file_info)

        assert groups == [files[:2], files[2:]]

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
        # 3. Each file has 1000 rows, so exactly 5 files fit per group

        # Expected groups:
        # - Group 0: files 0-4
        # - Group 1: files 5-9

        groups = break_parquet_partition_into_groups(test_files, embedding_dim=400_000)

        assert groups == [test_files[:5], test_files[5:]]

        # If we run with the default value of embedding_dim=1024, we should get one group
        groups = break_parquet_partition_into_groups(test_files)
        assert len(groups) == 1, "Should create one group"
        assert set(groups[0]) == set(test_files), "All input files should be included in groups"
