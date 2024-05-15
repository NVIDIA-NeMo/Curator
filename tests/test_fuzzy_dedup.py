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

import dask.dataframe as dd
import numpy as np
import pytest
import yaml
from dask import config
from dask.dataframe.utils import assert_eq
from distributed import Client

from nemo_curator import LSH, FuzzyDuplicates, FuzzyDuplicatesConfig, MinHash
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.fuzzy_dedup_utils.merge_utils import extract_partitioning_index
from nemo_curator.utils.import_utils import gpu_only_import, gpu_only_import_from

cudf = gpu_only_import("cudf")
dask_cudf = gpu_only_import("dask_cudf")
LocalCUDACluster = gpu_only_import_from("dask_cuda", "LocalCUDACluster")


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


@pytest.fixture
def large_fuzzy_dedup_data():
    df = cudf.DataFrame(
        {
            "id": np.arange(500),
            "text": [
                "A test string",
                "A different test string",
                "A different object",
                "The quick brown fox jumps over the lazy dog",
                "The quick black cat jumps over the lazy dog",
            ]
            * 100,
        }
    )
    df = dask_cudf.from_cudf(df, 5).reset_index(drop=True)
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
            minhash_approximation = minhash_overlap(np.array(sig1), np.array(sig2))
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
            num_hashes=6,
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
            num_hashes=6,
            num_buckets=3,
            buckets_per_shuffle=1,
            id_fields=["id", "dataset_id"],
            minhash_field="minhash_sig",
        )
        buckets = lsh(self.dataset)
        buckets_df = buckets.df.compute().to_pandas()
        buckets_df["new_id"] = list(
            map(list, zip(buckets_df.dataset_id, buckets_df.id))
        )
        docs_list = buckets_df.groupby("_bucket_id").new_id.apply(list)
        expected_df = cudf.Series(
            [[(1, 1), (1, 2)], [(1, 2), (2, 3)], [(3, 4), (4, 5)]], name="new_id"
        )
        assert_eq(expected_df, docs_list, check_index=False)


@pytest.mark.gpu
class TestFuzzyDuplicates:
    @pytest.fixture(autouse=True, scope="class")
    def gpu_client(self, request):
        with LocalCUDACluster(n_workers=1) as cluster, Client(cluster) as client:
            request.cls.client = client
            request.cls.cluster = cluster
            yield

    @pytest.mark.parametrize("use_64_bit_hash", [False, True])
    @pytest.mark.parametrize(
        "num_buckets,jaccard_threshold,duplicate_docs",
        # Duplcated docs estimated from true_jaccard values
        [
            (5, 0.5, [[4, -1]]),
            (10, 0.39, [[4, -1], [1, 2]]),
            (3, 0.3, [[4, -1], [1, 2, 300]]),
        ],
    )
    def test_fuzzy_dedup(
        self,
        fuzzy_dedup_data,
        use_64_bit_hash,
        num_buckets,
        jaccard_threshold,
        duplicate_docs,
        tmpdir,
    ):
        print(self.client)
        # Dedup might fail when indices per partition do not start from 0
        fuzzy_dedup_data.df = fuzzy_dedup_data.df.reset_index(drop=True)
        config = FuzzyDuplicatesConfig(
            cache_dir=tmpdir,
            id_field="id",
            text_field="text",
            seed=42,
            char_ngrams=5,
            num_buckets=num_buckets,
            hashes_per_bucket=1,
            use_64_bit_hash=use_64_bit_hash,
            buckets_per_shuffle=5,
            false_positive_check=True,
            num_anchors=2,
            jaccard_threshold=jaccard_threshold,
        )
        fuzzy_duplicates = FuzzyDuplicates(config=config)
        result = fuzzy_duplicates(fuzzy_dedup_data)
        result_df = result.df.compute()
        # Drop non duplicated docs
        result_df = result_df[result_df.group.duplicated(keep=False)]
        result_df = result_df.groupby("group").id.collect()
        # Sort to maintain uniform ordering

        result_df = result_df.list.sort_values()
        result_df = result_df.sort_values()
        expected_df = cudf.Series(duplicate_docs, name="id")
        expected_df = expected_df.list.sort_values()
        expected_df = expected_df.sort_values()
        assert_eq(expected_df, result_df, check_index=False)

    @pytest.mark.xfail
    def test_non_uniform_indices(
        self,
        tmpdir,
    ):
        print(self.client)
        # Dedup might fail when indices per partition do not start from 0
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
        data = DocumentDataset(df)
        duplicate_docs = [[4, -1], [1, 2, 300]]
        config = FuzzyDuplicatesConfig(
            cache_dir=tmpdir,
            id_field="id",
            text_field="text",
            seed=42,
            char_ngrams=5,
            num_buckets=10,
            hashes_per_bucket=1,
            use_64_bit_hash=False,
            buckets_per_shuffle=5,
            false_positive_check=True,
            num_anchors=2,
            jaccard_threshold=0.39,
        )
        fuzzy_duplicates = FuzzyDuplicates(config=config)
        result = fuzzy_duplicates(data)
        result_df = result.df.compute()
        # Drop non duplicated docs
        result_df = result_df[result_df.group.duplicated(keep=False)]
        result_df = result_df.groupby("group").id.collect()
        # Sort to maintain uniform ordering

        result_df = result_df.list.sort_values()
        result_df = result_df.sort_values()
        expected_df = cudf.Series(duplicate_docs, name="id")
        expected_df = expected_df.list.sort_values()
        expected_df = expected_df.sort_values()
        assert_eq(expected_df, result_df, check_index=False)

    @pytest.mark.parametrize("num_anchors", [1, 3, 10])
    def test_num_anchors(self, large_fuzzy_dedup_data, num_anchors, tmpdir):
        config = FuzzyDuplicatesConfig(
            cache_dir=tmpdir,
            id_field="id",
            text_field="text",
            seed=42,
            char_ngrams=5,
            num_buckets=5,
            hashes_per_bucket=1,
            use_64_bit_hash=False,
            buckets_per_shuffle=5,
            false_positive_check=True,
            num_anchors=num_anchors,
            jaccard_threshold=0.39,
        )
        fuzzy_duplicates = FuzzyDuplicates(config=config)
        fuzzy_duplicates(large_fuzzy_dedup_data)
        anchor_docs_df_cols = dask_cudf.read_parquet(
            tmpdir / "anchor_docs_with_bk.parquet"
        ).columns
        assert all(f"anchor_{i}_id" in anchor_docs_df_cols for i in range(num_anchors))


class TestFuzzyDuplicatesConfig:
    def test_bad_inputs(self, tmpdir):
        with pytest.raises(ValueError):
            FuzzyDuplicatesConfig(cache_dir=tmpdir, num_anchors=0)
        with pytest.warns(
            UserWarning, match="Using a higher number of anchor docs might"
        ):
            FuzzyDuplicatesConfig(cache_dir=tmpdir, num_anchors=3)
        with pytest.raises(ValueError):
            FuzzyDuplicatesConfig(cache_dir=tmpdir, jaccard_threshold=1.2)
        with pytest.raises(NotImplementedError):
            FuzzyDuplicatesConfig(cache_dir=tmpdir, false_positive_check=False)
        with pytest.raises(ValueError):
            FuzzyDuplicatesConfig(cache_dir=tmpdir, buckets_per_shuffle=0)

    def test_from_yaml(self, tmpdir):
        yaml_params = {
            "cache_dir": "./",
            "num_anchors": 2,
            "jaccard_threshold": 0.8,
            "false_positive_check": True,
            "buckets_per_shuffle": 1,
        }
        with open(tmpdir / "config.yaml", "w") as f:
            yaml.dump(yaml_params, f)
        config = FuzzyDuplicatesConfig.from_yaml(tmpdir / "config.yaml")
        for param in yaml_params:
            assert getattr(config, param) == yaml_params[param]


@pytest.mark.parametrize(
    "backend",
    [
        "pandas",
        pytest.param(
            "cudf",
            marks=pytest.mark.gpu,
        ),
    ],
)
def test_extract_partitioning_index(backend):

    def add_partition_info(df, partition_info=None):
        if partition_info is None:
            df["file_id"] = -1
        else:
            df["file_id"] = partition_info["number"]
        return df

    with config.set({"dataframe.backend": backend}):

        # Create a random `unshuffled` DataFrame with a
        # "part_id" column to be used as the shuffle index
        npartitions_left = 7
        unshuffled = dd.from_dict(
            {"part_id": np.random.randint(25, size=1000, dtype="int32")},
            npartitions=npartitions_left,
        )

        # Create a `bk_mapping` DataFrame that defines
        # the "correct" mapping beween "part_id" and
        # the destination partition ("file_id")
        npartitions_right = 5
        bk_mapping = (
            dd.from_dict(
                {"part_id": np.arange(25, dtype="int32")},
                npartitions=npartitions_right,
            )
            .shuffle("part_id")
            .map_partitions(add_partition_info)
            .compute()
        )

    # Use `extract_partitioning_index` to calculate
    # the partitioning index and assign it as a new
    # "_partitions" column
    result, _ = extract_partitioning_index(
        unshuffled,
        "part_id",
        bk_mapping,
        npartitions_right,
        npartitions_right,
    )

    # Rename the "_partitions" column, shuffle by "part_id",
    # and then assign a "file_id" column to reflect the final
    # partition of each row
    check = (
        result.rename(columns={"_partitions": "expected_file_id"})
        .shuffle(
            "part_id",
            npartitions=npartitions_right,
        )
        .map_partitions(add_partition_info)
        .compute()
    )

    # Check that the real and expected partitions match
    assert (check["file_id"] == check["expected_file_id"]).all()
