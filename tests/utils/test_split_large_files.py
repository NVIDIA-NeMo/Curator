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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable


import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.utils.split_large_files import parse_args, split_parquet_file_by_size


@pytest.fixture
def parquet_file_factory(tmp_path: pathlib.Path):
    def _(num_row_groups: int = 1) -> pathlib.Path:
        # This generates an in-memory pyarrow.Table of 18.5 MB
        # I.e. `t.nbytes / 1e6 == 18.5`
        num_rows = 500_000
        rng = np.random.default_rng(seed=2)
        t = pa.Table.from_pydict(
            {
                "id": np.arange(num_rows),
                "value1": rng.random(num_rows),
                "value2": rng.integers(0, 1000, num_rows),
                "category": rng.choice(["A", "B", "C", "D"], num_rows),
                "timestamp": pd.to_datetime("2023-01-01") + pd.to_timedelta(np.arange(num_rows), unit="s"),
            }
        )
        file = tmp_path / "test.parquet"
        pq.write_table(t, file, row_group_size=t.num_rows // num_row_groups)
        assert pq.ParquetFile(file).num_row_groups == num_row_groups
        return file

    return _


def test_default_target_size(parquet_file_factory: Callable, tmp_path: pathlib.Path):
    parquet_file = parquet_file_factory()
    args = parse_args(["--infile", str(parquet_file), "--outdir", str(tmp_path)])
    assert args.target_size_mb == 128


@pytest.mark.parametrize("num_row_groups", [1, 2, 5, 20])
def test_split_parquet_file_by_size(parquet_file_factory: Callable, tmp_path: pathlib.Path, num_row_groups: int):
    parquet_file = parquet_file_factory(num_row_groups=num_row_groups)
    size_original_mb = pq.read_table(parquet_file).nbytes / (1024 * 1024)
    target_size_mb = size_original_mb / 3
    outdir = tmp_path / "out"
    outdir.mkdir(exist_ok=True)
    split_parquet_file_by_size._function(input_file=parquet_file, outdir=outdir, target_size_mb=target_size_mb)

    expected = pd.read_parquet(parquet_file)
    result = pd.read_parquet(outdir)

    # Ensure the original and split data is the same
    pd.testing.assert_frame_equal(expected, result)

    # Check that split data files have expected sizes
    sizes_mb = [pq.read_table(f).nbytes / (1024 * 1024) for f in outdir.rglob("*")]
    # Below the target size
    assert all(s_mb < target_size_mb for s_mb in sizes_mb)
    # More than half the target (ignoring the last file, which can sometimes be small)
    assert all(s_mb > target_size_mb / 2 for s_mb in sizes_mb[:-1])
