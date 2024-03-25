# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import pandas as pd
import pytest
from dask import dataframe as dd
from dask.dataframe.utils import assert_eq

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
        result = exact_dups(exact_dedup_data)
        expected_df = exact_dedup_data.df.compute()
        expected_df = expected_df[expected_df.text.duplicated(keep=False)]
        assert_eq(result.df.id, expected_df.id, check_index=False)
