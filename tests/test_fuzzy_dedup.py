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

import os
from itertools import combinations
from typing import Iterable

import numpy as np
import pytest
from dask.dataframe.utils import assert_eq

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules import LSH, MinHash
from nemo_curator.utils.import_utils import gpu_only_import

cudf = gpu_only_import("cudf")
dask_cudf = gpu_only_import("dask_cudf")


@pytest.fixture
def fuzzy_dedup_data():
    df = cudf.DataFrame(
        {
            "id": [1, 2, 300, 4, -1],
            "text": [
                "A test string",
                "A different test string",
                "A different object",
                "The quick brown fox jumps over the lazy dog",
                "The quick black cat jumps over the lazy dog",
            ],
        }
    )
    df = dask_cudf.from_cudf(df, 2)
    return DocumentDataset(df)


def minhash_overlap(minhash1: np.array, minhash2: np.array):
    assert len(minhash1) == len(minhash2)
    overlap = sum(minhash1 == minhash2)
    return overlap / len(minhash1)


def jaccard_index(str1: str, str2: str, char_ngrams):
    return (
        cudf.Series([str1])
        .str.jaccard_index(cudf.Series([str2]), width=char_ngrams)
        .values_host[0]
    )


def generate_all_pairs(item: Iterable):
    return combinations(item, 2)


@pytest.mark.gpu
class TestMinhashes:
    @pytest.mark.parametrize("use_64bit_hash", [False, True])
    @pytest.mark.parametrize("seed,char_ngrams,num_hashes", [(128, 3, 260)])
    def test_identical_minhash(
        self, fuzzy_dedup_data, use_64bit_hash, seed, char_ngrams, num_hashes
    ):
        minhasher1 = MinHash(
            seed=seed,
            num_hashes=num_hashes,
            char_ngrams=char_ngrams,
            use_64bit_hash=use_64bit_hash,
        )
        minhash_sig1 = minhasher1(fuzzy_dedup_data)
        sig_lengths = minhash_sig1.df["_minhash_signature"].compute().list.len()
        assert (sig_lengths == num_hashes).all()

        minhasher2 = MinHash(
            seed=seed,
            num_hashes=num_hashes,
            char_ngrams=char_ngrams,
            use_64bit_hash=use_64bit_hash,
        )
        minhash_sig2 = minhasher2(fuzzy_dedup_data)
        assert_eq(minhash_sig1.df, minhash_sig2.df)

    @pytest.mark.parametrize(
        "use_64bit_hash,seed,char_ngrams,num_hashes",
        [(False, 42, 5, 20), (True, 32768, 10, 18)],
    )
    def test_minhash_approximation(
        self, fuzzy_dedup_data, use_64bit_hash, seed, char_ngrams, num_hashes
    ):
        THRESHOLD = 0.15

        minhasher = MinHash(
            seed=seed,
            num_hashes=num_hashes,
            char_ngrams=char_ngrams,
            use_64bit_hash=use_64bit_hash,
        )
        minhashes = minhasher(fuzzy_dedup_data)
        minhash_signatures = (
            minhashes.df["_minhash_signature"].compute().to_pandas().values
        )
        strings = fuzzy_dedup_data.df["text"].compute().to_pandas().values
        for (sig1, str1), (sig2, str2) in generate_all_pairs(
            tuple(zip(minhash_signatures, strings))
        ):
            true_jaccard = jaccard_index(str1, str2, char_ngrams)
            minhash_approximation = minhash_overlap(sig1, sig2)
            assert abs(true_jaccard - minhash_approximation) < THRESHOLD

    def test_minhash_cache(self, fuzzy_dedup_data, tmpdir):
        minhasher = MinHash(cache_dir=tmpdir)
        result = minhasher(fuzzy_dedup_data)
        assert len(result) == len(fuzzy_dedup_data)
        assert "_minhashes.parquet" in os.listdir(tmpdir)
        assert len(os.listdir(tmpdir / "_minhashes.parquet")) != 0


@pytest.mark.gpu
class TestLSH:
    @pytest.fixture(autouse=True)
    def minhash_data(self):
        df = cudf.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "dataset_id": [1, 1, 2, 3, 4],
                "minhash_sig": [
                    [1, 2, 1, 2, 1, 2],
                    [1, 2, 3, 4, 5, 6],
                    [3, 2, 1, 4, 5, 6],
                    [9, 8, 7, 6, 5, 4],
                    [3, 1, 2, 4, 5, 4],
                ],
            }
        )
        df = dask_cudf.from_cudf(df, 2)
        self.dataset = DocumentDataset(df)

    @pytest.mark.parametrize("buckets_per_shuffle", [1, 2, 3])
    def test_lsh(self, tmpdir, buckets_per_shuffle):
        lsh = LSH(
            cache_dir=tmpdir,
            minhash_length=6,
            num_buckets=3,
            buckets_per_shuffle=buckets_per_shuffle,
            minhash_field="minhash_sig",
            id_fields="id",
        )
        buckets = lsh(self.dataset)
        buckets_df = buckets.df
        docs_list = buckets_df.groupby("_bucket_id").id.collect()
        expected_df = cudf.Series([[1, 2], [2, 3], [4, 5]], name="id")
        assert_eq(expected_df, docs_list, check_index=False)

    def test_multiple_id_cols(self, tmpdir):
        lsh = LSH(
            cache_dir=tmpdir,
            minhash_length=6,
            num_buckets=3,
            buckets_per_shuffle=1,
            id_fields=["id", "dataset_id"],
            minhash_field="minhash_sig",
        )
        buckets = lsh(self.dataset)
        buckets_df = buckets.df.compute().to_pandas()
        buckets_df["new_id"] = list(zip(buckets_df.dataset_id, buckets_df.id))
        docs_list = buckets_df.groupby("_bucket_id").new_id.apply(list)
        expected_df = cudf.Series(
            [[(1, 1), (1, 2)], [(1, 2), (2, 3)], [(3, 4), (4, 5)]], name="new_id"
        )
        assert_eq(expected_df, docs_list, check_index=False)
