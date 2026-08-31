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
from unittest.mock import patch

import fsspec
import pandas as pd
import pytest


@pytest.mark.gpu
def test_get_array_from_df() -> None:
    import cudf
    import cupy as cp

    from nemo_curator.stages.deduplication.semantic.utils import get_array_from_df

    df = cudf.DataFrame({"embedding": [[3, 4, 5], [1, 2, 2], [1, 0, 0]]})
    expected_array = cp.array([[3, 4, 5], [1, 2, 2], [1, 0, 0]])
    cp.testing.assert_allclose(get_array_from_df(df, "embedding"), expected_array, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu  # TODO: Remove this once semantic imports work without GPU dependencies.
class TestBreakParquetPartitionIntoGroups:
    def test_bulk_row_counts_preserve_file_order_and_empty_files(self, tmp_path: Path) -> None:
        from nemo_curator.stages.deduplication.semantic.utils import read_parquet_file_row_counts

        files = []
        for index, rows in enumerate((7, 0, 3)):
            path = tmp_path / f"rows_{index}.parquet"
            pd.DataFrame({"value": range(rows)}).to_parquet(path)
            files.append(str(path))

        assert read_parquet_file_row_counts(files) == dict(zip(files, (7, 0, 3), strict=True))

    def test_bulk_row_counts_honor_remote_storage_options(self) -> None:
        from nemo_curator.stages.deduplication.semantic.utils import read_parquet_file_row_counts

        buffer = io.BytesIO()
        pd.DataFrame({"value": range(4)}).to_parquet(buffer)
        filesystem = fsspec.filesystem("memory", auto_mkdir=True)
        filesystem.pipe("metadata-test/data.parquet", buffer.getvalue())

        assert read_parquet_file_row_counts(
            ["memory://metadata-test/data.parquet"], storage_options={"auto_mkdir": True}
        ) == {"memory://metadata-test/data.parquet": 4}

    def test_bulk_row_counts_use_one_pylibcudf_call(self, tmp_path: Path) -> None:
        import pylibcudf as plc

        from nemo_curator.stages.deduplication.semantic.utils import read_parquet_file_row_counts

        files = []
        for index in range(3):
            path = tmp_path / f"part_{index}.parquet"
            pd.DataFrame({"value": [index]}).to_parquet(path)
            files.append(str(path))

        original = plc.io.parquet_metadata.read_parquet_footers
        with patch.object(plc.io.parquet_metadata, "read_parquet_footers", wraps=original) as read_footers:
            assert read_parquet_file_row_counts(files) == dict.fromkeys(files, 1)
        read_footers.assert_called_once()

    def test_bulk_row_counts_reject_corrupt_footer(self, tmp_path: Path) -> None:
        from nemo_curator.stages.deduplication.semantic.utils import read_parquet_file_row_counts

        valid = tmp_path / "valid.parquet"
        corrupt = tmp_path / "corrupt.parquet"
        pd.DataFrame({"value": [1]}).to_parquet(valid)
        corrupt.write_bytes(b"not parquet")

        with pytest.raises(RuntimeError, match="Failed to read Parquet footer metadata"):
            read_parquet_file_row_counts([str(valid), str(corrupt)])

    def test_empty_file_list_skips_metadata_read(self) -> None:
        from nemo_curator.stages.deduplication.semantic.utils import (
            break_parquet_partition_into_groups,
            read_parquet_file_row_counts,
        )

        assert read_parquet_file_row_counts([]) == {}
        assert break_parquet_partition_into_groups([]) == []

    def test_exact_row_counts_bound_groups_and_are_reusable(self) -> None:
        from nemo_curator.stages.deduplication.semantic import utils as semantic_utils

        files = ["a.parquet", "b.parquet", "c.parquet"]
        row_counts = {"a.parquet": 40, "b.parquet": 70, "c.parquet": 30}
        with patch.object(
            semantic_utils,
            "read_parquet_file_row_counts",
            side_effect=AssertionError("metadata should not be reread"),
        ):
            groups = semantic_utils.break_parquet_partition_into_groups(
                files, embedding_dim=20_000_000, row_counts=row_counts
            )

        assert groups == [["a.parquet"], ["b.parquet", "c.parquet"]]

    def test_small_files_no_break(self, tmp_path: Path) -> None:
        from nemo_curator.stages.deduplication.semantic.utils import break_parquet_partition_into_groups

        files = []
        for index in range(5):
            path = tmp_path / f"small_{index}.parquet"
            pd.DataFrame({"id": range(10), "embedding": [[1.0, 2.0, 3.0]] * 10}).to_parquet(path)
            files.append(str(path))

        assert break_parquet_partition_into_groups(files, embedding_dim=1024) == [files]

    def test_large_files_break_at_exact_limit(self) -> None:
        from nemo_curator.stages.deduplication.semantic.utils import break_parquet_partition_into_groups

        files = [f"part_{index}.parquet" for index in range(10)]
        row_counts = dict.fromkeys(files, 1_000)
        groups = break_parquet_partition_into_groups(files, embedding_dim=400_000, row_counts=row_counts)

        assert groups == [files[:5], files[5:]]

    def test_file_larger_than_limit_is_rejected(self) -> None:
        from nemo_curator.stages.deduplication.semantic.utils import break_parquet_partition_into_groups

        with pytest.raises(ValueError, match="exceeding the per-group limit"):
            break_parquet_partition_into_groups(
                ["large.parquet"], embedding_dim=20_000_000, row_counts={"large.parquet": 101}
            )
