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

from hashlib import md5

import pandas as pd
import pytest
from dask import dataframe as dd

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules import ExactDuplicates


@pytest.fixture(
    params=[False, pytest.param(True, marks=pytest.mark.gpu)], ids=["no gpu", "gpu"]
)
def exact_dedup_data(request):
    df = pd.DataFrame(
        {"id": [1, 2, 300, 4, -1], "text": ["abc", "aba", "abb", "aba", "abc"]}
    )
    df = dd.from_pandas(df, 2)
    if request.param:
        df = df.to_backend("cudf")
    return DocumentDataset(df)


@pytest.fixture
def exact_no_dedup_data(request):
    # A dataset with no exact duplicates
    df = pd.DataFrame({"id": [1, 2, 300], "text": ["abc", "aba", "abb"]})
    df = dd.from_pandas(df, 2)
    return DocumentDataset(df)


class TestExactDuplicates:
    def test_unsupported_hash(self):
        with pytest.raises(ValueError):
            ExactDuplicates(hash_method="sha256")

    @pytest.mark.parametrize("cache_result", [False, True])
    def test_dup(self, exact_dedup_data, cache_result, tmpdir):
        exact_dups = ExactDuplicates(
            id_field="id",
            text_field="text",
            hash_method="md5",
            cache_dir=tmpdir if cache_result else None,
        )
        duplicates = exact_dups.identify_duplicates(exact_dedup_data)
        deduplicated_ds = exact_dups.remove(exact_dedup_data, duplicates)
        deduplicated_ids_series = deduplicated_ds.df.to_backend("pandas").compute()[
            "id"
        ]
        output_deduplicated_ids = set(deduplicated_ids_series.tolist())
        assert (
            len(output_deduplicated_ids) == 3
            and 300 in output_deduplicated_ids
            and len({-1, 1}.intersection(output_deduplicated_ids)) == 1
            and len({2, 4}.intersection(output_deduplicated_ids)) == 1
        )

        duplicates_df = (
            duplicates.df.to_backend("pandas")
            .compute()
            .sort_values(by="id", ignore_index=True)
        )
        expected_df = pd.DataFrame(
            {
                "id": [1, -1] + [2, 4],
                "_hashes": [md5(b"abc").hexdigest()] * 2
                + [md5(b"aba").hexdigest()] * 2,
            }
        ).sort_values(by="id", ignore_index=True)
        pd.testing.assert_frame_equal(duplicates_df, expected_df, check_like=True)

    def test_no_dedup(self, exact_no_dedup_data):
        exact_dups = ExactDuplicates(
            id_field="id",
            text_field="text",
            hash_method="md5",
            perform_removal=True,
        )
        result_df = exact_dups(exact_no_dedup_data).df.compute().reset_index(drop=True)
        expected_df = exact_no_dedup_data.df.compute().reset_index(drop=True)

        pd.testing.assert_frame_equal(result_df, expected_df)
