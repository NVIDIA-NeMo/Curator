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

import io
from pathlib import Path
from unittest.mock import Mock, patch

import fsspec
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
    @patch("nemo_curator.stages.deduplication.semantic.utils.read_parquet_file_row_counts")
    def test_calculation_logic(self, mock_read_row_counts: Mock) -> None:
        from nemo_curator.stages.deduplication.semantic.utils import break_parquet_partition_into_groups

        """Test the calculation logic of break_parquet_partition_into_groups without actual files."""
        test_files = [f"mock_file_{i}.parquet" for i in range(1000)]
        # Mock the bulk parquet metadata read to return a specific number of rows per file
        mock_read_row_counts.return_value = dict.fromkeys(test_files, 10_000)

        # Test with embedding_dim=1000
        # Expected calculation:
        # - cudf_max_num_rows = 2_000_000_000
        # - cudf_max_num_elements = 2_000_000_000 / 1000 = 2_000_000
        # - each file has exactly 10_000 rows according to its footer
        # - max_files_per_subgroup = int(2_000_000 / 10_000) = 200
        # Since we have 1000 files and max_files_per_subgroup=200

        groups = break_parquet_partition_into_groups(test_files, embedding_dim=1000)

        # Verify metadata was read once in bulk
        mock_read_row_counts.assert_called_once_with(test_files, storage_options=None)

        assert len(groups) == 5, "1000 files each with 10k rows with embedding_dim=1000 should fit in 5 groups"
        for group in groups:
            assert len(group) == 200, "Each group should contain 200 files"

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
        # 3. Each file has exactly 1000 rows according to its footer
        # 4. Max files per group: int(5,000 / 1,000) = 5
        # 5. With 10 files and max 5 files per group: 2 groups

        # Expected groups:
        # - Group 0: files 0 through 4 (5 files)
        # - Group 1: files 5 through 9 (5 files)

        groups = break_parquet_partition_into_groups(test_files, embedding_dim=400_000)

        # Verify we get exactly 2 groups as calculated above
        assert len(groups) == 2, "Should create 2 groups based on embedding_dim=400,000 calculation"
        for i, group in enumerate(groups):
            assert len(group) == 5, f"Group {i} should contain 5 files"
            assert set(group) == set(test_files[i * 5 : (i + 1) * 5]), (
                f"Group {i} should contain files {i * 5} to {(i + 1) * 5}"
            )

        # If we run with the default value of embedding_dim=1024, we should get one group
        groups = break_parquet_partition_into_groups(test_files)
        assert len(groups) == 1, "Should create one group"
        assert set(groups[0]) == set(test_files), "All input files should be included in groups"


@pytest.mark.gpu
class TestReadParquetFileRowCounts:
    def test_multiple_files_preserve_order_and_empty_file(self, tmp_path: Path) -> None:
        """Return exact row counts in input order, including a valid zero-row file."""
        from nemo_curator.stages.deduplication.semantic.utils import read_parquet_file_row_counts

        test_files = []
        expected_rows = [7, 0, 3]
        for i, num_rows in enumerate(expected_rows):
            file_path = tmp_path / f"test_file_{i}.parquet"
            pd.DataFrame({"value": range(num_rows)}).to_parquet(file_path)
            test_files.append(str(file_path))

        assert read_parquet_file_row_counts(test_files) == dict(zip(test_files, expected_rows, strict=True))

    def test_remote_files_honor_storage_options(self) -> None:
        """Pass storage options through fsspec for remote inputs."""
        from nemo_curator.stages.deduplication.semantic.utils import read_parquet_file_row_counts

        parquet_buffer = io.BytesIO()
        pd.DataFrame({"value": range(4)}).to_parquet(parquet_buffer)
        filesystem = fsspec.filesystem("memory", auto_mkdir=True)
        filesystem.pipe("metadata-test/data.parquet", parquet_buffer.getvalue())

        assert read_parquet_file_row_counts(
            ["memory://metadata-test/data.parquet"], storage_options={"auto_mkdir": True}
        ) == {"memory://metadata-test/data.parquet": 4}

    def test_multiple_files_use_one_bulk_footer_call(self, tmp_path: Path) -> None:
        """A small file set is passed to pylibcudf in one bulk metadata call."""
        import pylibcudf as plc

        from nemo_curator.stages.deduplication.semantic.utils import read_parquet_file_row_counts

        test_files = []
        for i in range(3):
            file_path = tmp_path / f"test_file_{i}.parquet"
            pd.DataFrame({"value": [i]}).to_parquet(file_path)
            test_files.append(str(file_path))

        original = plc.io.parquet_metadata.read_parquet_footers
        with patch.object(plc.io.parquet_metadata, "read_parquet_footers", wraps=original) as read_footers:
            assert read_parquet_file_row_counts(test_files) == dict.fromkeys(test_files, 1)
        read_footers.assert_called_once()

    def test_large_file_set_uses_bounded_bulk_calls(self) -> None:
        """Bound each pylibcudf call below typical process file-descriptor limits."""
        import pylibcudf as plc

        from nemo_curator.stages.deduplication.semantic.utils import read_parquet_file_row_counts

        test_files = [f"test_file_{i}.parquet" for i in range(513)]
        with (
            patch.object(plc.io, "SourceInfo", side_effect=lambda sources: sources),
            patch.object(
                plc.io.parquet_metadata,
                "read_parquet_footers",
                side_effect=[[Mock(num_rows=1)] * 512, [Mock(num_rows=1)]],
            ) as read_footers,
        ):
            assert read_parquet_file_row_counts(test_files) == dict.fromkeys(test_files, 1)
        assert read_footers.call_count == 2

    def test_corrupt_footer_raises_bounded_error(self, tmp_path: Path) -> None:
        """Do not guess row counts when any footer in a bounded batch is unreadable."""
        from nemo_curator.stages.deduplication.semantic.utils import read_parquet_file_row_counts

        valid_file = tmp_path / "valid.parquet"
        corrupt_file = tmp_path / "corrupt.parquet"
        pd.DataFrame({"value": [1]}).to_parquet(valid_file)
        corrupt_file.write_bytes(b"not parquet")

        with pytest.raises(RuntimeError, match="Failed to read Parquet footer metadata"):
            read_parquet_file_row_counts([str(valid_file), str(corrupt_file)])

    def test_empty_file_list_skips_metadata_read(self) -> None:
        """An empty input has no footer work and returns an empty ordered mapping."""
        from nemo_curator.stages.deduplication.semantic.utils import read_parquet_file_row_counts

        assert read_parquet_file_row_counts([]) == {}
