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
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# TODO: Remove the GPU markers when the semantic package can be imported without GPU-only dependencies.


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
def test_parquet_file_info_preserves_order(tmp_path: Path) -> None:
    import cudf

    from nemo_curator.stages.deduplication.semantic.utils import read_parquet_file_info

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

    # Grouping consumes this list positionally, so footer batching must not reorder the caller's files.
    assert [info.path for info in file_info] == files
    assert [info.num_rows for info in file_info] == [7, 7, 7]
    assert [info.embedding_elements for info in file_info] == [14, 14, 14]


@pytest.mark.gpu
def test_parquet_file_info_uses_fsspec_for_remote_uri_without_storage_options() -> None:
    from nemo_curator.stages.deduplication.semantic.utils import read_parquet_file_info

    sink = pa.BufferOutputStream()
    pq.write_table(
        pa.table(
            {
                "metadata.with.dot": [1, 2],
                "embeddings.with.dot": pa.array([[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float32())),
            }
        ),
        sink,
    )
    remote_path = "s3://public-bucket/input.parquet"

    # The in-memory footer keeps this a unit test while the URI verifies remote-path dispatch.
    with patch(
        "nemo_curator.stages.deduplication.semantic.utils.open_parquet_files",
        return_value=[pa.BufferReader(sink.getvalue())],
    ) as open_files:
        [info] = read_parquet_file_info(
            [remote_path],
            retained_columns=["metadata.with.dot"],
            embedding_column="embeddings.with.dot",
        )

    open_files.assert_called_once_with([remote_path], storage_options=None, row_groups=[])
    assert info.path == remote_path
    assert info.num_rows == 2
    assert info.metadata_bytes > 0
    assert info.embedding_elements == 4


@pytest.mark.gpu
class TestBreakParquetPartitionIntoGroups:
    def test_calculation_logic(self) -> None:
        from nemo_curator.stages.deduplication.semantic.utils import (
            ParquetFileInfo,
            break_parquet_partition_into_groups,
        )

        """Test the calculation logic of break_parquet_partition_into_groups without actual files."""
        test_files = [f"mock_file_{i}.parquet" for i in range(1000)]
        file_info = [ParquetFileInfo(path, 10_000, 0, embedding_elements=10_000_000) for path in test_files]

        # The limit is strict: 199 files contain 1.99B embedding leaves and fit, while 200
        # contain exactly 2B and must start the next group. Therefore 1000 files need 6 groups.
        groups = break_parquet_partition_into_groups(file_info)

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

        # Exact footer counts let the 1.899B and 100M files share a 1.999B group. Adding the
        # final 1M file would reach the unsupported 2B boundary, so it starts a new group.
        groups = break_parquet_partition_into_groups(file_info)

        assert groups == [files[:2], files[2:]]
