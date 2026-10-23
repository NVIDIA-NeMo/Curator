# modality: text

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from contextlib import suppress

import pytest

with suppress(ImportError):
    import cudf

    from nemo_curator.stages.deduplication.fuzzy.banding import (
        get_band_columns,
        melt_band_columns,
        minhash_to_band_columns,
    )


@pytest.mark.gpu
def test_selected_band_hashes_use_absolute_band_numbers() -> None:
    minhashes = cudf.Series(
        [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 7, 8, 5, 6],
        ]
    )

    all_bands = minhash_to_band_columns(minhashes, num_bands=3, minhashes_per_band=2)
    selected_bands = minhash_to_band_columns(
        minhashes,
        num_bands=3,
        minhashes_per_band=2,
        band_range=(1, 3),
    )

    assert list(selected_bands.columns) == ["_minhash_band_1", "_minhash_band_2"]
    assert selected_bands.to_pandas().equals(all_bands[list(selected_bands.columns)].to_pandas())
    assert selected_bands["_minhash_band_1"].str.startswith("b1_").all()
    assert selected_bands["_minhash_band_2"].str.startswith("b2_").all()


@pytest.mark.gpu
def test_melt_preserves_absolute_band_identity() -> None:
    minhashes = cudf.Series([[1, 2, 3, 4]])
    band_df = minhash_to_band_columns(minhashes, num_bands=2, minhashes_per_band=2)
    band_df.insert(0, "document_id", [7])

    result = melt_band_columns(
        band_df,
        id_field="document_id",
        band_columns=get_band_columns((0, 2)),
    )

    assert list(result.columns) == ["document_id", "_bucket_id"]
    assert result["document_id"].to_pandas().tolist() == [7, 7]
    assert result["_bucket_id"].str.startswith("b0_").to_pandas().tolist() == [True, False]
    assert result["_bucket_id"].str.startswith("b1_").to_pandas().tolist() == [False, True]
