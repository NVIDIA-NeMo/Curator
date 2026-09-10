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

import cudf

from nemo_curator.stages.deduplication.fuzzy.utils import CURATOR_LSH_BUCKET_FIELD

CURATOR_MINHASH_BAND_FIELD_PREFIX = "_minhash_band_"


def get_band_columns(
    band_range: tuple[int, int],
    band_field_prefix: str = CURATOR_MINHASH_BAND_FIELD_PREFIX,
) -> list[str]:
    """Return the top-level Parquet columns for a half-open band range."""
    return [f"{band_field_prefix}{band_number}" for band_number in range(*band_range)]


def minhash_to_band_columns(
    minhashes: cudf.Series,
    num_bands: int,
    minhashes_per_band: int,
    band_range: tuple[int, int] | None = None,
    band_field_prefix: str = CURATOR_MINHASH_BAND_FIELD_PREFIX,
) -> cudf.DataFrame:
    """Hash MinHash signature slices into one top-level column per LSH band."""
    if num_bands < 1:
        msg = f"num_bands must be at least 1, got {num_bands}"
        raise ValueError(msg)
    if minhashes_per_band < 1:
        msg = f"minhashes_per_band must be at least 1, got {minhashes_per_band}"
        raise ValueError(msg)

    band_range = band_range or (0, num_bands)
    if band_range[0] < 0 or band_range[1] > num_bands or band_range[0] >= band_range[1]:
        msg = f"Invalid band range: {band_range}, must be in range [0, {num_bands}]"
        raise ValueError(msg)

    band_df = cudf.DataFrame(index=minhashes.index)
    for band_number in range(*band_range):
        indices = list(
            range(
                band_number * minhashes_per_band,
                (band_number + 1) * minhashes_per_band,
            )
        )
        row_indices = cudf.Series([indices]).repeat(len(minhashes))
        column = f"{band_field_prefix}{band_number}"
        band_df[column] = f"b{band_number}_" + minhashes.list.take(row_indices).hash_values(method="md5")

    return band_df


def melt_band_columns(
    band_df: cudf.DataFrame,
    id_field: str,
    band_columns: list[str],
) -> cudf.DataFrame:
    """Convert projected wide band columns into the rows consumed by the LSH shuffle."""
    melted_df = band_df.melt(
        id_vars=[id_field],
        value_name=CURATOR_LSH_BUCKET_FIELD,
        value_vars=band_columns,
    )
    return melted_df[[id_field, CURATOR_LSH_BUCKET_FIELD]]
