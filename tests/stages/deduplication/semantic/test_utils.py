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
        cudf.DataFrame(
            {
                "id": range(index * 7, (index + 1) * 7),
                "value": [index] * 7,
                "embeddings": [[1.0, 2.0]] * 7,
            }
        ).to_parquet(path)
        files.append(str(path))
    file_info = read_parquet_file_info(files, retained_columns=["id"], embedding_column="embeddings")

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
    assert [info.embedding_elements for info in file_info] == [14, 14, 14]


@pytest.mark.gpu  # TODO : Remove this once we figure out how to import semantic on CPU
class TestBreakParquetPartitionIntoGroups:
    def test_calculation_logic(self) -> None:
        from nemo_curator.stages.deduplication.semantic.utils import (
            ParquetFileInfo,
            break_parquet_partition_into_groups,
        )

        """Test the calculation logic of break_parquet_partition_into_groups without actual files."""
        test_files = [f"mock_file_{i}.parquet" for i in range(1000)]
        file_info = [ParquetFileInfo(path, 10_000, 0, embedding_elements=10_000_000) for path in test_files]
        groups = break_parquet_partition_into_groups(test_files, file_info=file_info)

        assert len(groups) == 6
        assert all(len(group) <= 199 for group in groups)

    def test_uses_exact_counts_for_skewed_files(self) -> None:
        from nemo_curator.stages.deduplication.semantic.utils import (
            ParquetFileInfo,
            break_parquet_partition_into_groups,
        )

        files = ["large.parquet", "tiny-1.parquet", "tiny-2.parquet"]
        file_info = [
            ParquetFileInfo(files[0], 1_900, 0, embedding_elements=1_899_000_000),
            ParquetFileInfo(files[1], 100, 0, embedding_elements=100_000_000),
            ParquetFileInfo(files[2], 1, 0, embedding_elements=1_000_000),
        ]

        groups = break_parquet_partition_into_groups(files, file_info=file_info)

        assert groups == [files[:2], files[2:]]
