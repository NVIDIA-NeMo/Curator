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

from __future__ import annotations

import math
import os
import time
import warnings
from datetime import datetime
from typing import List, Tuple, Union

import cudf
import cugraph.dask as dcg
import cugraph.dask.comms.comms as Comms
import cupy as cp
import dask_cudf
import numpy as np
from cugraph import MultiGraph
from dask import dataframe as dd
from dask.dataframe.shuffle import shuffle as dd_shuffle
from dask.utils import M
from tqdm import tqdm

from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.modules.config import FuzzyDuplicatesConfig
from nemo_curator.modules.meta import Sequential
from nemo_curator.utils.distributed_utils import (
    get_current_client,
    get_num_workers,
    performance_report_if,
)
from nemo_curator.utils.fuzzy_dedup_utils.id_mapping import int_ids_to_str
from nemo_curator.utils.fuzzy_dedup_utils.io_utils import (
    aggregated_anchor_docs_with_bk_read,
    get_restart_offsets,
    update_restart_offsets,
)
from nemo_curator.utils.fuzzy_dedup_utils.merge_utils import (
    extract_partitioning_index,
    filter_text_rows_by_bucket_batch,
    merge_left_to_shuffled_right,
)
from nemo_curator.utils.fuzzy_dedup_utils.output_map_utils import (
    build_partition,
    get_agg_text_bytes_df,
)
from nemo_curator.utils.fuzzy_dedup_utils.shuffle_utils import (
    text_bytes_aware_shuffle,
    write_partitioned_file,
)


class MinHash:
    """
    Computes minhash signatures of a document corpus
    """

    def __init__(
        self,
        seed: int = 42,
        num_hashes: int = 260,
        char_ngrams: int = 5,
        use_64bit_hash: bool = False,
        logger: Union[logging.LoggerAdapter, str] = "./",
        id_field: str = "id",
        text_field: str = "text",
        profile_dir: str = None,
        cache_dir: str = None,
    ):
        """
        Parameters
        ----------
        seed: Seed for minhash permutations
        num_hashes: Length of minhash signature (No. of minhash permutations)
        char_ngrams: Width of text window (in characters) while computing minhashes.
        use_64bit_hash: Whether to use a 64 bit hash function.
        logger: Existing logger to log to, or a path to a log directory.
        id_field: Column in the Dataset denoting document ID.
        text_field: Column in the Dataset denoting document content.
        profile_dir: str, Default None
          If specified directory to write dask profile
        cache_dir: str, Default None
          If specified, will compute & write id,minhash pairs to directory
        """
        self.num_hashes = num_hashes
        self.char_ngram = char_ngrams
        self.seeds = self.generate_seeds(n_seeds=self.num_hashes, seed=seed)
        self.minhash_method = self.minhash64 if use_64bit_hash else self.minhash32
        self.id_field = id_field
        self.text_field = text_field

        if cache_dir is None and profile_dir is not None:
            warnings.warn(
                "cache_dir for intermediate outputs is required to generate profiles"
            )
        self.cache_dir = cache_dir
        self.profile_dir = profile_dir

        if isinstance(logger, str):
            self._logger = create_logger(
                rank=0,
                log_file=os.path.join(logger, "Minhash.log"),
                name="Minhash",
            )
        else:
            self._logger = logger

    def generate_seeds(self, n_seeds: int = 260, seed: int = 0) -> np.ndarray:
        """
        Generate seeds for all minhash permutations based on the given seed.
        """
        gen = np.random.RandomState(seed)
        return gen.randint(0, 1e6, size=n_seeds)

    def minhash32(
        self, ser: cudf.Series, seeds: np.ndarray, char_ngram: int
    ) -> cudf.Series:
        """
        Compute 32bit minhashes based on the MurmurHash3 algorithm
        """
        if not isinstance(ser, cudf.Series):
            raise TypeError("Expected data of type cudf.Series")
        seeds = cudf.Series(seeds, dtype="uint32")
        return ser.str.minhash(seeds=seeds, width=char_ngram)

    def minhash64(
        self, ser: cudf.Series, seeds: np.ndarray, char_ngram: int
    ) -> cudf.Series:
        """
        Compute 64bit minhashes based on the MurmurHash3 algorithm
        """
        if not isinstance(ser, cudf.Series):
            raise TypeError("Expected data of type cudf.Series")
        seeds = cudf.Series(seeds, dtype="uint64")
        return ser.str.minhash64(seeds=seeds, width=char_ngram)

    def __call__(self, dataset: DocumentDataset) -> Union[str, DocumentDataset]:
        """
        Computes the MinHash Signatures for a given dataset.
        Parameters
        ----------
        dataset: DocumentDataset
        The input datset to compute MinHashes.
        Returns
        -------
        DocumentDataset containing IDs of all documents and the corresponding MinHash Signature
        """
        result = dataset.df[[self.id_field]]
        result["_minhash_signature"] = dataset.df[self.text_field].map_partitions(
            self.minhash_method,
            seeds=self.seeds,
            char_ngram=self.char_ngram,
        )

        if self.cache_dir is None:
            return DocumentDataset(result)

        t0 = time.time()
        self._logger.info("Starting execution for Minhashes")
        write_path = os.path.join(self.cache_dir, "_minhashes.parquet")
        if os.path.exists(write_path):
            warnings.warn(
                f"Output path {write_path} already exists and will be overwritten"
            )
        with performance_report_if(
            self.profile_dir,
            f"minhash-profile-{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        ):
            result.to_parquet(write_path, write_index=False, overwrite=True)
        self._logger.info(
            f"Minhash signature computation for dataset took {time.time() - t0}s complete at {write_path}"  # noqa:E501
        )
        return DocumentDataset(
            dask_cudf.read_parquet(write_path, blocksize="2GB", aggregate_files=True)
        )


class LSH:
    """
    Performs LSH on a MinhashSignatures
    """

    def __init__(
        self,
        cache_dir: str,
        num_hashes: int,
        num_buckets: int,
        buckets_per_shuffle: int = 1,
        logger: Union[logging.LoggerAdapter, str] = "./",
        id_fields: Union[str, list] = "id",
        minhash_field: str = "_minhash_signature",
        profile_dir: str = None,
    ):
        """
        Parameters
        ----------
        cache_dir: str
          Needs to be specified, will compute & write duplicate id, bucket pairs to cache directory.
        num_hashes: Length of minhash signature
        num_buckets: Number of bands/buckets to create from the minhash signature.
          Hashes_per_signature = num_hashes / num_buckets
        buckets_per_shuffle: Number of bands/buckets to shuffle concurrently.
          Larger values process larger batches by processing multiple bands
          but might lead to memory pressures and related errors.
        logger: Existing logger to log to, or a path to a log directory.
        id_field: Columns in the Dataset denoting document ID.
        minhash_field: Column in the Dataset denoting minhash signature.
        profile_dir: str, Default None
          If specified directory to write dask profile
        """
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.id_fields = [id_fields] if isinstance(id_fields, str) else id_fields
        self.minhash_field = minhash_field
        self.buckets_per_shuffle = buckets_per_shuffle
        self.bucket_ranges = self._generate_bucket_ranges(
            self.num_buckets, self.num_hashes
        )

        if cache_dir is None:
            raise ValueError(
                "cache_dir for intermediate outputs is required for this stage"
            )
        self.cache_dir = cache_dir
        self.profile_dir = profile_dir

        if isinstance(logger, str):
            self._logger = create_logger(
                rank=0,
                log_file=os.path.join(logger, "LSH.log"),
                name="LSH",
            )
        else:
            self._logger = logger

    def _generate_bucket_ranges(
        self, num_buckets: int, num_hashes: int
    ) -> List[List[int]]:
        """
        Generates a list of indices for the minhash ranges given num_bands &
        num_hashes.
        eg: num_bands=3, num_hashes=6
        [[0, 1], [2, 3], [4, 5]]
        """
        minhashes_per_bucket = num_hashes // num_buckets

        bucket_ranges = [
            list(
                range(
                    bucket * minhashes_per_bucket, (bucket + 1) * minhashes_per_bucket
                )
            )
            for bucket in range(num_buckets)
        ]
        return bucket_ranges

    def minhash_to_buckets(
        self,
        df: cudf.DataFrame,
        bucket_ranges: List[List[int]],
    ) -> cudf.DataFrame:
        df2 = df[self.id_fields]
        for i, h in enumerate(bucket_ranges):
            indices = cudf.Series([h]).repeat(len(df2))
            df2[f"_bucket_{i}"] = f"b{i}_" + df[self.minhash_field].list.take(
                indices
            ).hash_values(method="md5")
        return df2

    def bucket_id_to_int(
        self,
        bucket_ddf: dask_cudf.DataFrame,
        bucket_col_name: str = "bucket_id",
        start_id: int = 0,
    ) -> Tuple[dask_cudf.DataFrame, int]:
        """
        Maps bucket ids to a contigious integer range from starting from start_id.
        """
        unique_bucket_df = (
            bucket_ddf[[bucket_col_name]]
            .map_partitions(lambda x: x.drop_duplicates(ignore_index=True))
            .persist()
        )
        end_bucket_id = len(unique_bucket_df) - 1 + start_id
        unique_bucket_df["bucket_int_id"] = np.uint64(1)
        unique_bucket_df["bucket_int_id"] = unique_bucket_df["bucket_int_id"].cumsum()
        unique_bucket_df["bucket_int_id"] = (
            unique_bucket_df["bucket_int_id"] - 1 + start_id
        )
        bucket_ddf = bucket_ddf.merge(unique_bucket_df, on=[bucket_col_name])
        bucket_ddf = bucket_ddf.drop(columns=[bucket_col_name])
        bucket_ddf = bucket_ddf.rename(columns={"bucket_int_id": "_bucket_id"})
        bucket_ddf["_bucket_id"] = bucket_ddf["_bucket_id"].astype(np.uint64)
        return (bucket_ddf, end_bucket_id)

    def _minhash_to_bucket_meta(
        self, df: dask_cudf.DataFrame
    ) -> Tuple[cudf.DataFrame, int]:
        meta = df._meta_nonempty[self.id_fields]
        meta[self.minhash_field] = [np.ones(self.num_hashes)] * len(meta)
        return self.minhash_to_buckets(meta, self.bucket_ranges)

    def lsh(
        self,
        write_path: str,
        df: dask_cudf.DataFrame,
    ) -> None:
        """
        Computes buckets and writes them as parquet files to the write_path
        """
        meta = self._minhash_to_bucket_meta(df)
        df = df.map_partitions(
            self.minhash_to_buckets,
            bucket_ranges=self.bucket_ranges,
            meta=meta,
        )
        bucket_start_id = 0
        for i in range(0, self.num_buckets, self.buckets_per_shuffle):
            value_vars = [
                f"_bucket_{i}"
                for i in range(i, min(self.num_buckets, i + self.buckets_per_shuffle))
            ]
            df2 = df.melt(
                id_vars=self.id_fields, value_name="_bucket_id", value_vars=value_vars
            )[self.id_fields + ["_bucket_id"]]

            df2 = df2.shuffle(
                on=["_bucket_id"],
                ignore_index=True,
                npartitions=max(1, 2 ** math.floor(math.log2(df2.npartitions))),
            ).map_partitions(lambda x: x[x["_bucket_id"].duplicated(keep=False)])

            df2 = df2.reset_index(drop=True)
            df2, end_id = self.bucket_id_to_int(
                df2, bucket_col_name="_bucket_id", start_id=bucket_start_id
            )
            # If bucketing return empty dataframe
            if end_id < bucket_start_id:
                continue
            bucket_start_id = end_id + 1

            # Workaround for dtype mismatches with empty partitions
            dtypes = df2.dtypes.to_dict()
            df2 = df2.map_partitions(lambda x: x.astype(dtypes))

            if i == 0:
                if os.path.exists(write_path):
                    warnings.warn(
                        f"Output path {write_path} already exists and will be overwritten"
                    )
                df2.to_parquet(write_path, write_index=False, overwrite=True)
            else:
                df2.to_parquet(write_path, write_index=False, append=True)

            self._logger.info(f"Wrote data for buckets: {value_vars}")

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        df = dataset.df

        write_path = os.path.join(self.cache_dir, "_buckets.parquet")
        t0 = time.time()
        with performance_report_if(
            self.profile_dir,
            f"lsh-profile-{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        ):
            self.lsh(write_path=write_path, df=df)
        self._logger.info(f"Computing and writing buckets took {time.time() - t0} s")

        buckets_df = dask_cudf.read_parquet(write_path, split_row_groups=False)
        return DocumentDataset(buckets_df)


class FuzzyDuplicates:
    def __init__(
        self,
        config: FuzzyDuplicatesConfig,
        logger: Union[logging.LoggerAdapter, str] = "./",
    ):
        """
        Parameters
        ----------
        config: FuzzyDuplicatesConfig,
            Config options for finding FuzzyDuplicates
        logger: Existing logger to log to, or a path to a log directory.

        Returns
        -------
        DocumentDataset containing IDs of all documents and the corresponding duplicate group
        they belong to. Documents in the same group are near duplicates.
        """
        if isinstance(logger, str):
            self._logger = create_logger(
                rank=0,
                log_file=os.path.join(logger, "FuzzyDuplicates.log"),
                name="FuzzyDuplicates",
            )
        else:
            self._logger = logger

        self.config = config
        self.minhash = MinHash(
            seed=self.config.seed,
            num_hashes=self.config.num_hashes,
            char_ngrams=self.config.char_ngrams,
            use_64bit_hash=self.config.use_64_bit_hash,
            logger=self._logger,
            id_field=self.config.id_field,
            text_field=self.config.text_field,
            profile_dir=self.config.profile_dir,
            cache_dir=self.config.cache_dir,
        )
        self.lsh = LSH(
            cache_dir=self.config.cache_dir,
            num_hashes=self.config.num_hashes,
            num_buckets=self.config.num_buckets,
            buckets_per_shuffle=self.config.buckets_per_shuffle,
            logger=self._logger,
            id_fields=[self.config.id_field],
            profile_dir=self.config.profile_dir,
        )
        self.map_buckets = _MapBuckets(
            id_fields=[self.config.id_field],
            text_field=self.config.text_field,
            logger=self._logger,
            num_anchors=self.config.num_anchors,
        )
        self.jaccard_shuffle = _Shuffle(
            id_fields=[self.config.id_field],
            text_field=self.config.text_field,
            logger=self._logger,
            profile_dir=self.config.profile_dir,
        )
        self.jaccard_compute = JaccardSimilarity(
            id_field=self.config.id_field,
            text_field=self.config.text_field,
            ngram_width=self.config.char_ngrams,
            anchor_id_fields=[
                f"anchor_{i}_{self.config.id_field}"
                for i in range(self.config.num_anchors)
            ],
        )
        self.connected_components = ConnectedComponents(
            cache_dir=self.config.cache_dir,
            jaccard_pairs_path=os.path.join(
                self.config.cache_dir, "jaccard_similarity_results.parquet"
            ),
            id_column=self.config.id_field,
            convert_str_ids=False,
            jaccard_threshold=self.config.jaccard_threshold,
        )

    def __call__(self, dataset: DocumentDataset):
        """
        Parameters
        ----------
        dataset: DocumentDataset
            The input datset to compute FuzzyDuplicates. Must contain a text and unique id field.

        Returns
        -------
        DocumentDataset containing IDs of all documents and the corresponding duplicate group
        they belong to. Documents in the same group are near duplicates.
        """
        # Minhash + LSH
        print("Stage1: Starting Minhash + LSH computation")
        minhashLSH = Sequential([self.minhash, self.lsh])
        buckets_df = minhashLSH(dataset)
        print("Stage1: Minhash + LSH complete!")

        # Map buckets to lower cardinality distribution
        print("Stage2 (False Postive Check): Starting Map_Buckets")
        ddf_mapped_buckets_w_anchors = self.map_buckets.map_buckets_with_anchors(
            documents_df=dataset.df, buckets_df=buckets_df.df
        )
        mapped_buckets_w_anchors_path = os.path.join(
            self.config.cache_dir, "anchor_docs_with_bk.parquet"
        )
        ddf_mapped_buckets_w_anchors.to_parquet(
            mapped_buckets_w_anchors_path, write_index=False
        )
        print("Stage2 (False Postive Check): Map_Buckets Complete!")

        # Shuffle documents based on mapped buckets
        print("Stage3 (False Postive Check): Shuffle docs")
        shuffled_docs_path = os.path.join(
            self.config.cache_dir, "shuffled_docs.parquet"
        )
        self.jaccard_shuffle.shuffle_docs_on_buckets(
            documents_df=dataset.df,
            bucket_w_anchors_path=mapped_buckets_w_anchors_path,
            output_shuffled_docs_path=shuffled_docs_path,
            bucket_mapping_df_blocksize=256,
            parts_per_worker=1,
            bucket_parts_per_worker=8,
        )
        print("Stage3 (False Postive Check): Shuffle docs complete!")

        # jaccard comparision within buckets
        print("Stage4 (False Postive Check): Jaccard Similarity in Buckets")
        jaccard_pairs_path = os.path.join(
            self.config.cache_dir, "jaccard_similarity_results.parquet"
        )
        jaccard_pairs_df = self.jaccard_compute.jaccard_compute(
            shuffled_docs_path=shuffled_docs_path
        )
        jaccard_pairs_df.to_parquet(
            jaccard_pairs_path,
            write_index=False,
            write_metadata_file=False,
        )
        print("Stage4 (False Postive Check): Jaccard Similarity in Buckets Complete!")

        # Connected components across buckets
        print("Stage5: Connected Components across buckets")
        cc_path = os.path.join(self.config.cache_dir, "connected_components.parquet")
        self.connected_components.cc_workflow(cc_path)
        print("Stage5: Connected Components across buckets complete!")
        return DocumentDataset(dask_cudf.read_parquet(cc_path, split_row_groups=False))


class _MapBuckets:
    """
    buckets to a logical partition by using a modified bin packing algorithm.
    Combines buckets generated from LSH (typically high cardinality)
    to more coarse lower cardinality bucket groups by mapping multiple buckets
    to a logical partition using document length information and a modified bin
    packing algorithm.
    Only needed if running False Postive check to remove false positives.
    """

    def __init__(
        self,
        id_fields: Union[list, str] = "id",
        text_field: str = "text",
        bucket_field: str = "_bucket_id",
        num_anchors: int = 2,
        logger: Union[logging.LoggerAdapter, str] = "./",
    ):
        """
        id_fields: list or str
            id fields of df
        text_field: str = "text",
        bucket_column: str = "bucket_column",
        num_anchors: int = 2,
        logger: Union[logging.LoggerAdapter, str] = "./",
        """
        self.id_fields = [id_fields] if isinstance(id_fields, str) else id_fields
        self.text_field = text_field
        self.num_anchors = num_anchors
        self.bucket_field = bucket_field
        if isinstance(logger, str):
            self._logger = create_logger(
                rank=0,
                log_file=os.path.join(logger, "Map_Buckets.log"),
                name="Map_Buckets",
            )
        else:
            self._logger = logger

    @staticmethod
    def _get_output_part_ids_with_approx_equal_sum(
        bucket_text_bytes_df: cudf.DataFrame,
        max_text_bytes_per_part: int,
        buckets_column: str,
        bytes_column: str,
        output_partition_column: str,
    ) -> cudf.DataFrame:
        """
        Create a output_series that maps the ser.index into `nparts`
        so that the total sum of bucket_val_counts_df
        for each output id are all most equal and
        less than max_text_bytes_per_part
        This is used downstream for creating equal output_ids
        """
        sizes = bucket_text_bytes_df[bytes_column].values
        bucket_output_ar = build_partition(
            sizes=sizes.get(), max_size=max_text_bytes_per_part
        )
        df = cudf.DataFrame()
        df[buckets_column] = bucket_text_bytes_df[buckets_column]
        df[output_partition_column] = bucket_output_ar
        return df

    def _get_output_map_from_text_bytes_per_bucket(
        self,
        ddf_bk_text_bytes,
        bytes_column,
        output_partition_column="_output_partition_id",
    ):
        # String bytes limit for cuDF
        # https://github.com/rapidsai/cudf/issues/13733
        max_text_bytes_per_part = int(np.iinfo(np.int32).max // 1.2)

        self._logger.info(f"max_text_bytes_per_part = {max_text_bytes_per_part}")
        # Increasing in an attempt to prevent hitting
        # ulimits
        output_map_df_meta = cudf.DataFrame(
            {self.bucket_field: [0], output_partition_column: [1]}
        )
        output_map_df_meta = output_map_df_meta.astype(
            {self.bucket_field: np.uint64, output_partition_column: np.int32}
        )

        output_map_df = ddf_bk_text_bytes.map_partitions(
            _MapBuckets._get_output_part_ids_with_approx_equal_sum,
            max_text_bytes_per_part=max_text_bytes_per_part,
            buckets_column=self.bucket_field,
            bytes_column=bytes_column,
            output_partition_column=output_partition_column,
            meta=output_map_df_meta,
        )
        output_map_df = output_map_df.persist()
        self._logger.info(
            f"Step 1 of output_map_df of len: {len(output_map_df)} computed"
        )
        lower_bounds = (
            output_map_df[output_partition_column]
            .map_partitions(lambda s: (s.max() + 1))
            .compute()
        )
        lower_bounds = np.cumsum(lower_bounds)

        def update_id(df, lower_bound):
            df[output_partition_column] += lower_bound
            return df

        updated_parts = [
            output_map_df.get_partition(i).map_partitions(
                update_id, lower_bounds[i - 1]
            )
            for i in range(1, len(lower_bounds))
        ]
        updated_parts.append(output_map_df.get_partition(0))
        output_map_df = dask_cudf.concat(updated_parts)
        output_map_df = output_map_df.persist()
        self._logger.info(
            f"All steps of output_map_df of len: {len(output_map_df)} computed"
        )
        return output_map_df

    def _get_output_map_based_on_str_bytes(
        self, buckets_df, documents_df, bytes_column="_text_bytes"
    ):
        """
        Add output_partition_id to buckets_ddf
        """
        documents_df = documents_df.copy()
        documents_df[bytes_column] = documents_df[self.text_field].map_partitions(
            lambda s: s.str.byte_count()
        )
        n_partitions = buckets_df.npartitions
        documents_df = documents_df.drop(columns=[self.text_field]).repartition(
            npartitions=n_partitions
        )
        buckets_df = buckets_df.merge(documents_df).repartition(
            npartitions=n_partitions
        )
        del documents_df
        ddf_bk_text_bytes, agg_df_len = get_agg_text_bytes_df(
            df=buckets_df,
            agg_column=self.bucket_field,
            bytes_column=bytes_column,
            n_partitions=n_partitions,
            shuffle=True,
        )
        self._logger.info(f"Agg_df computed of length = {agg_df_len}")
        del buckets_df
        output_map_df = self._get_output_map_from_text_bytes_per_bucket(
            ddf_bk_text_bytes=ddf_bk_text_bytes,
            bytes_column=bytes_column,
        )
        return output_map_df

    def _random_select_anchor(self, buckets_df, n=2):
        """
        Randomly select `n` anchors from each bucket.
        """
        buckets_df = buckets_df.copy()
        buckets_df["_id_hash"] = buckets_df[self.id_fields].hash_values()
        buckets_df = buckets_df.sort_values([self.bucket_field, "_id_hash"])
        buckets_df["_order_in_bucket"] = buckets_df.groupby(
            self.bucket_field
        ).cumcount()
        buckets_df["is_anchor"] = buckets_df["_order_in_bucket"] < n
        for i in range(0, n):
            buckets_df[f"is_anchor_id_{i}"] = buckets_df["_order_in_bucket"] == i
        buckets_df = buckets_df.drop(columns=["_id_hash", "_order_in_bucket"], axis=1)
        buckets_df = buckets_df.reset_index(drop=True)
        buckets_df = buckets_df[buckets_df.is_anchor]
        return buckets_df

    def _add_anchor_docs(self, buckets_df, num_anchors):
        """
        Get anchor documents for each bucket.
        """
        df_anchor_bk = self._random_select_anchor(buckets_df=buckets_df, n=num_anchors)
        df_anchor_docs = None
        for i in range(num_anchors):
            df_anchor_bk_i = df_anchor_bk[df_anchor_bk[f"is_anchor_id_{i}"]][
                [self.bucket_field] + self.id_fields
            ].reset_index(drop=True)
            column_mapping = {id: f"anchor_{i}_{id}" for id in self.id_fields}
            df_anchor_bk_i = df_anchor_bk_i.rename(columns=column_mapping)
            if i == 0:
                df_anchor_docs = df_anchor_bk_i
            else:
                df_anchor_docs = df_anchor_bk_i.merge(
                    df_anchor_docs, on=[self.bucket_field], how="inner"
                )

        df_anchor_docs_with_bk = buckets_df.merge(
            df_anchor_docs, on=[self.bucket_field], how="inner"
        )
        return df_anchor_docs_with_bk

    def map_buckets_with_anchors(
        self,
        documents_df: dask_cudf.DataFrame,
        buckets_df: dask_cudf.DataFrame,
        shuffle_type: Union[str, bool, None] = "tasks",
    ) -> dask_cudf.DataFrame:
        """
        Get anchor docs with bucket info
        Args:
            input_data_paths: list of paths to input data
            input_bucket_path: path to input buckets
            text_ddf_blocksize: blocksize for text ddf
            num_files: number of files to read
            num_workers: number of workers
            shuffle_type: type of shuffle to use
        Returns:
            ddf_anchor_docs_with_bk
        """
        output_map_df = self._get_output_map_based_on_str_bytes(
            buckets_df=buckets_df, documents_df=documents_df
        )
        ddf_anchor_docs_with_bk = buckets_df.map_partitions(
            self._add_anchor_docs, num_anchors=self.num_anchors
        )
        self._logger.info("output_map_df is based on string bytes")
        ddf_anchor_docs_with_bk = ddf_anchor_docs_with_bk.merge(
            output_map_df, on=self.bucket_field
        )
        # Bucket is no longer needed
        ddf_anchor_docs_with_bk = ddf_anchor_docs_with_bk.drop(
            columns=[self.bucket_field]
        )
        # Below removes any duplicates lying around after dropping buckets
        ddf_anchor_docs_with_bk = ddf_anchor_docs_with_bk.map_partitions(
            M.drop_duplicates,
            meta=ddf_anchor_docs_with_bk._meta,
            enforce_metadata=False,
            transform_divisions=False,
            align_dataframes=False,
        )
        ddf_anchor_docs_with_bk = dd_shuffle(
            ddf_anchor_docs_with_bk,
            self.id_fields,
            ignore_index=True,
            shuffle_method=shuffle_type,
        ).map_partitions(
            M.drop_duplicates,
            meta=ddf_anchor_docs_with_bk._meta,
            enforce_metadata=False,
            transform_divisions=False,
            align_dataframes=False,
        )
        del output_map_df
        return ddf_anchor_docs_with_bk


class _Shuffle:
    def __init__(
        self,
        id_fields: Union[str, list] = "id",
        text_field: str = "text",
        logger: Union[logging.LoggerAdapter, str] = "./",
        profile_dir: str = None,
        int_to_str_id: str = None,
    ):
        if isinstance(logger, str):
            self._logger = create_logger(
                rank=0,
                log_file=os.path.join(logger, "LSH.log"),
                name="LSH",
            )
        else:
            self._logger = logger

        self.id_fields = id_fields
        self.text_field = text_field
        self.profile_dir = profile_dir
        self.int_to_str_id = int_to_str_id

    def shuffle_docs_on_buckets(
        self,
        documents_df: dask_cudf.DataFrame,
        bucket_w_anchors_path: str,
        output_shuffled_docs_path: str,
        bucket_mapping_df_blocksize,
        parts_per_worker: int = 1,
        bucket_parts_per_worker: int = 8,
        partition_on: str = "_output_partition_id",
    ):
        ddf_anchor_docs_with_bk, bk_mapping = aggregated_anchor_docs_with_bk_read(
            path=bucket_w_anchors_path,
            blocksize=bucket_mapping_df_blocksize,
        )
        self._logger.info("Getting ddf_anchor_docs_with_bk completed")
        self._logger.debug(
            f"ddf_anchor_docs_with_bk.npartitions = {ddf_anchor_docs_with_bk.npartitions}"
        )
        st = time.time()
        num_workers = get_num_workers(get_current_client())
        parts_per_batch = num_workers * parts_per_worker
        self._logger.debug(f"parts_per_batch  = {parts_per_batch}")
        parts_per_bucket_batch = num_workers * bucket_parts_per_worker
        self._logger.debug(f"parts_per_bucket_batch  = {parts_per_bucket_batch}")

        dask_profile_name = "suffle_docs"
        dask_profile_name = dask_profile_name + f"-parts_per_batch-{parts_per_batch}"
        dask_profile_name = (
            dask_profile_name + f"-parts_per_bucket_batch-{parts_per_bucket_batch}"
        )
        dask_profile_name = (
            dask_profile_name + f"-{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        documents_df = documents_df[self.id_fields + [self.text_field]]

        with performance_report_if(self.profile_dir, dask_profile_name):
            self._batched_merge_and_write(
                left_df=documents_df,
                right_df=ddf_anchor_docs_with_bk,
                output_path=output_shuffled_docs_path,
                merge_on=self.id_fields,
                partition_on=partition_on,
                parts_per_text_batch=parts_per_batch,
                parts_per_bucket_batch=parts_per_bucket_batch,
                bk_mapping=bk_mapping,
                num_workers=num_workers,
            )
        self._logger.info(f"Writing+Shuffling data took = {time.time()-st} s")

    def _batched_merge_and_write(
        self,
        left_df: dask_cudf.DataFrame,
        right_df: dask_cudf.DataFrame,
        output_path: str,
        merge_on: List[str],
        partition_on: str,
        parts_per_text_batch: int,
        parts_per_bucket_batch: int,
        bk_mapping,
        num_workers: int = None,
    ):
        total_text_partitions = left_df.npartitions
        total_bucket_partitions = right_df.npartitions

        # Extract global partitioning index
        left_df, global_partitioning_index = extract_partitioning_index(
            left_df,
            merge_on,
            bk_mapping,
            parts_per_bucket_batch,
            total_bucket_partitions,
        )

        # Set start offsets
        bucket_part_start_offset, text_part_start_offset = get_restart_offsets(
            output_path
        )

        # Set end offsets
        # NOTE: These end offsets are always set to the end
        # of the data. However, we may want to be able to set
        # both the start and end offsets from the command line
        # in the future.
        bucket_part_end_offset = total_bucket_partitions
        text_part_end_offset = total_text_partitions

        # Check that offsets are valid
        assert bucket_part_start_offset % parts_per_bucket_batch == 0
        assert bucket_part_end_offset > bucket_part_start_offset
        assert text_part_end_offset > text_part_start_offset

        # Initialize "retry" variables
        #
        # - retry_count: The number of successive batches that
        #     we have already performed at a reduced batch size.
        # - retry_threshold: The number of successive batches
        #     for which we should keep the batch size low
        #     before attempting the default batch size again.
        #     Every time we return to the default batch size
        #     and immediately fail, retry_threshold will double.
        parts_per_text_batch_retry = None
        retry_count, retry_threshold = 0, 1

        self._logger.info(
            f"Starting at bucket-map partition {bucket_part_start_offset}"
            f" and text-df partition {text_part_start_offset}",
        )

        for bucket_part_offset in tqdm(
            range(
                bucket_part_start_offset, bucket_part_end_offset, parts_per_bucket_batch
            )
        ):

            # Outer loop over batches of "bucket-map" partitions
            end_bucket_offset = min(
                bucket_part_offset + parts_per_bucket_batch, bucket_part_end_offset
            )
            print(
                f"\nStarted processing bucket-map partitions {bucket_part_offset} "
                f"through {end_bucket_offset} of {bucket_part_end_offset}",
                flush=True,
            )
            st_bucket = time.time()

            # Select our bucket-mapping batch
            subset_bucket_df = right_df.partitions[bucket_part_offset:end_bucket_offset]
            subset_bucket_df = subset_bucket_df.persist()

            # Filter out rows of left_df that we know cannot
            # align with any rows of subset_bucket_df
            left_df_use = filter_text_rows_by_bucket_batch(
                left_df,
                global_partitioning_index,
                bucket_part_offset,
                bucket_part_end_offset,
                total_bucket_partitions,
            )

            text_part_offset = text_part_start_offset
            while text_part_offset < text_part_end_offset:

                # Check if we are "retrying" with a smaller "parts_per_text_batch"
                if parts_per_text_batch_retry:
                    parts_per_text_batch_use = parts_per_text_batch_retry
                else:
                    st_text = time.time()
                    parts_per_text_batch_use = parts_per_text_batch
                print(f"Using {parts_per_text_batch_use} text partitions.", flush=True)

                # Select partitions for our text batch
                end_text_offset = min(
                    text_part_offset + parts_per_text_batch_use, text_part_end_offset
                )
                subset_text_df = left_df_use.partitions[
                    text_part_offset:end_text_offset
                ]

                try:
                    # NOTE: If we have more text-df partitions than bucket-map
                    # partitions, we are more likely to see an OverflowError
                    output_df = text_bytes_aware_shuffle(
                        df=merge_left_to_shuffled_right(
                            subset_text_df,
                            subset_bucket_df,
                            merge_on,
                        ),
                        partition_on=partition_on,
                        text_column=self.text_field,
                        num_workers=num_workers,
                    )
                except OverflowError as err:
                    # We encountered an overflow error!
                    # Let's try again with less text data
                    parts_per_text_batch_retry = int(parts_per_text_batch_use / 2)
                    if parts_per_text_batch_retry < 1:
                        raise err
                    print(
                        f"\nWe encountered an OverflowError and will retry "
                        f"the current batch with {parts_per_text_batch_retry} "
                        f"text partitions instead of {parts_per_text_batch_use}.",
                        flush=True,
                    )
                    continue

                if self.int_to_str_id is not None:
                    output_df = output_df.map_partitions(
                        int_ids_to_str, id_column=self.int_to_str_id
                    )
                batch_label = f"{end_bucket_offset}_{end_text_offset}"
                written_files = output_df.map_partitions(
                    write_partitioned_file,
                    output_path,
                    partition_on,
                    batch_label,
                    meta=cudf.Series([True]),
                )
                written_files = written_files.compute()
                update_restart_offsets(output_path, bucket_part_offset, end_text_offset)
                del output_df

                print(
                    "Text-df partition ",
                    f"{end_text_offset}/{text_part_end_offset} "
                    f"completed in {time.time()-st_text}",
                    flush=True,
                )

                # Update loop control-flow variables
                if parts_per_text_batch_use == parts_per_text_batch:
                    # We succeeded at the default batch size.
                    # Reset the retry count
                    retry_count, retry_threshold = 0, 1
                else:
                    # We succeeded at a lower batch size
                    retry_count += 1
                    if retry_count >= retry_threshold:
                        # Go back to the default text-batch size,
                        # but increase the retry_threshold in
                        # case we fail again
                        parts_per_text_batch_retry = None
                        retry_count, retry_threshold = 0, min(retry_threshold * 2, 16)
                text_part_offset += parts_per_text_batch_use

            update_restart_offsets(output_path, end_bucket_offset, end_text_offset)
            print(
                "Bucket partition ",
                f"{end_bucket_offset}/{bucket_part_end_offset} "
                f"completed in {time.time()-st_bucket}",
                flush=True,
            )

            # Need to reset text_part_start_offset to 0 after
            # a single bucket-batch pass (only matters if we are
            # breaking the bucket-mapping df into multiple batches)
            text_part_start_offset = 0


class JaccardSimilarity:
    def __init__(
        self,
        id_field="id",
        anchor_id_fields=["anchor_0_id", "anchor_1_id"],
        text_field="text",
        ngram_width=5,
    ):
        self.id_field = id_field
        self.anchor_id_fields = anchor_id_fields
        self.text_field = text_field
        self.anchor_id = f"anchor_{id_field}"
        self.left_id = f"{self.id_field}_x"
        self.right_id = f"{self.id_field}_y"
        self.ngram_width = ngram_width

    def __call__(DocumentDataset):
        raise NotImplementedError

    def jaccard_compute(self, shuffled_docs_path):
        paths = [
            entry.path
            for entry in os.scandir(shuffled_docs_path)
            if not entry.path.endswith(".txt")
        ]
        meta_df = cudf.DataFrame(
            {
                self.left_id: ["x"],
                self.right_id: ["y"],
                "jaccard": np.float32([0.0]),
            }
        )
        result_df = dd.from_map(
            self._compute_jaccard_on_1_partition, paths, meta=meta_df
        ).reset_index(drop=True)
        return result_df

    def _compute_jaccard_on_1_partition(self, path):
        try:
            df = cudf.read_parquet(path)
            pair_df = self._compute_jaccard_and_create_pair_df(df)
        except OverflowError:
            paths = [entry.path for entry in os.scandir(os.path.join(path))]
            anchor_df_str_size_ls = [
                self._get_anchor_docs_and_string_size(path) for path in paths
            ]
            anchor_df = cudf.concat(
                [anchor_doc for anchor_doc, _ in anchor_df_str_size_ls],
                ignore_index=True,
            ).drop_duplicates()
            df_str_size = [str_size for _, str_size in anchor_df_str_size_ls]
            paths = JaccardSimilarity._create_bins(
                df_str_size, np.iinfo(np.int32).max // 10
            )
            pair_dfs = []
            for path in paths:
                print(path)
                df = cudf.read_parquet(path).reset_index(drop=True)
                df = cudf.concat([df, anchor_df], ignore_index=True)
                pair_df = self._compute_jaccard_and_create_pair_df(df)
                pair_dfs.append(pair_df)
            pair_df = cudf.concat(pair_dfs, ignore_index=True)
        return pair_df

    def _get_anchor_docs_and_string_size(self, path):
        df = cudf.read_parquet(path)
        str_bytes = df[self.text_field].str.byte_count().sum()
        is_anchor_flag = df[self.id_field] == df[self.anchor_id_fields[0]]
        for anchor_id in self.anchor_id_fields[1:]:
            is_anchor_flag = is_anchor_flag | (df[self.id_field] == df[anchor_id])
        anchor_df = df[is_anchor_flag].reset_index(drop=True)
        return anchor_df, {"path": path, "str_bytes": str_bytes}

    @staticmethod
    def _create_bins(path_dicts, max_size):
        path_dicts.sort(key=lambda x: x["str_bytes"], reverse=True)
        bins, bin_sizes = [], []
        for path_d in path_dicts:
            new_path, new_size = path_d["path"], path_d["str_bytes"]
            for i, bin_size in enumerate(bin_sizes):
                if bin_size + new_size <= max_size:
                    bins[i].append(new_path)
                    bin_sizes[i] += new_size
                    new_size = 0
                    break
            if new_size:
                bins.append([new_path])
                bin_sizes.append(new_size)
        return bins

    def _compute_jaccard_and_create_pair_df(self, df):
        df = df.drop_duplicates(
            subset=[self.id_field] + self.anchor_id_fields, ignore_index=True
        )
        anchor_columns = self.anchor_id_fields
        id_field = self.id_field
        result_ls = []
        try:
            for anchor_col in anchor_columns:
                doc_df = df[[id_field, self.text_field, anchor_col]]
                doc_df = doc_df.rename(columns={anchor_col: self.anchor_id})
                doc_df = doc_df[doc_df[id_field] != doc_df[self.anchor_id]]
                anchor_df = self._get_anchor_df(df, anchor_col)
                result_df = self._compute_jaccard_pair(doc_df, anchor_df)
                result_ls.append(result_df)

            return cudf.concat(result_ls)
        except OverflowError as e:
            print(
                "Failed with  OverflowError in compute_jaccard_and_create_pair_df",
                flush=True,
            )
            print(df, flush=True)
            print("--" * 30)
            print("Error")
            print("---" * 30)
            raise e

    def _get_anchor_df(self, df, anchor_col):
        anchor_df = df[df[self.id_field] == df[anchor_col]]
        anchor_df = anchor_df.reset_index(drop=True)
        anchor_df = anchor_df[[anchor_col, self.text_field]]
        anchor_df = anchor_df.rename(columns={anchor_col: self.anchor_id})
        return anchor_df

    def _compute_jaccard_pair(self, docs_df, anchor_df):
        nrows_at_once = JaccardSimilarity._get_max_num_rows_to_process_once(
            df=docs_df, text_field=self.text_field
        )
        result_ls = []
        for i in range(0, docs_df.shape[0], nrows_at_once):
            pair_df = docs_df[i : i + nrows_at_once]
            pair_df = pair_df.merge(anchor_df, on=self.anchor_id)
            pair_df = pair_df.rename(
                columns={self.id_field: self.left_id, self.anchor_id: self.right_id}
            )
            mask = pair_df[self.left_id] != pair_df[self.right_id]
            pair_df = pair_df[mask].reset_index(drop=True)
            if len(pair_df) == 0:
                result_df = self._create_empty_jaccard_result()
            else:
                result_df = self._compute_jaccard_partition(pair_df)
            result_ls.append(result_df)
        if len(result_ls) == 0:
            return self._create_empty_jaccard_result()
        df_pair = cudf.concat(result_ls)
        return df_pair

    def _create_empty_jaccard_result(self):
        df = cudf.DataFrame()
        df[self.left_id] = "x"
        df[self.right_id] = "y"
        df["jaccard"] = np.empty(shape=0, dtype=np.float32)
        return df

    def _compute_jaccard_partition(self, df):
        text_x = f"{self.text_field}_x"
        text_y = f"{self.text_field}_y"
        df["jaccard"] = df[text_x].str.jaccard_index(df[text_y], width=self.ngram_width)
        df.drop(columns=[text_x, text_y], inplace=True)
        return df

    @staticmethod
    def _get_max_num_rows_to_process_once(df, text_field):
        nbytes = df["text"].str.byte_count().sum()
        # Number of exmploded bytes
        exploded_bytes = nbytes * 5 * 2
        max_chars_allowed = 2_147_483_647
        byte_ratio = int(exploded_bytes) // max_chars_allowed
        if byte_ratio > 1:
            nrows_at_once = len(df) // byte_ratio
        else:
            nrows_at_once = len(df)

        nrows_at_once = max(1, nrows_at_once)
        return nrows_at_once


class ConnectedComponents:
    def __init__(
        self,
        cache_dir: str,
        jaccard_pairs_path: str,
        id_column="id",
        convert_str_ids=False,
        jaccard_threshold: int = 0.8,
    ):
        self.cache_dir = cache_dir
        self.jaccard_pairs_path = jaccard_pairs_path
        self.id_column = id_column
        self.left_id = f"{id_column}_x"
        self.right_id = f"{id_column}_y"
        self.convert_str_ids = convert_str_ids
        self.jaccard_threshold = jaccard_threshold

    def cc_workflow(self, output_path):
        deduped_parsed_id_path = self._write_dedup_parsed_id()
        encoded_jaccard_pair_path = self._write_encoded_jaccard_pair(
            deduped_parsed_id_path
        )
        deduped_encoded_jaccard_path = self._write_dedup_encoded_jaccard_pair(
            encoded_jaccard_pair_path
        )
        cc_path = self._run_connected_components(
            deduped_encoded_jaccard_path, deduped_parsed_id_path, output_path
        )
        return cc_path

    def _run_connected_components(
        self,
        deduped_encoded_jaccard_path,
        deduped_parsed_id_path,
        output_path,
    ):
        Comms.initialize(p2p=True)
        df = dask_cudf.read_parquet(
            deduped_encoded_jaccard_path, blocksize="1GB", aggregate_files=True
        )
        df = df[df["jaccard"] == 1].reset_index(drop=True)

        labels_df = dask_cudf.read_parquet(deduped_parsed_id_path)
        num_nodes = len(labels_df)
        self_edge_df = labels_df[["uid"]].rename(columns={"uid": self.left_id})
        self_edge_df[self.right_id] = self_edge_df[self.left_id]

        df = df[[self.left_id, self.right_id]].astype(np.int64)
        df = dask_cudf.concat([df, self_edge_df])

        G = MultiGraph(directed=False)
        G.from_dask_cudf_edgelist(
            df, source=self.left_id, destination=self.right_id, renumber=False
        )
        result = dcg.weakly_connected_components(G)
        del G
        max_partitions = min(32, result.npartitions)
        n_components = len(result[["labels"]].drop_duplicates(split_out=max_partitions))
        num_labels = len(result)
        print("# of groups", n_components)
        print("# of docs removed", num_labels - n_components)
        labels_df = labels_df.merge(
            result, left_on=["uid"], right_on=["vertex"], how="inner"
        )
        id_columns = (
            ["dataset_id", "doc_id"] if self.convert_str_ids else [self.id_column]
        )
        labels_df = labels_df[id_columns + ["labels"]]
        labels_df = labels_df.rename(columns={"labels": "group"})
        labels_df = labels_df.persist()
        # Doing an inner merge above
        # should not change any rows

        assert num_nodes == len(labels_df)
        print(f"assert num_nodes:{num_nodes}==labels_df:{len(labels_df)} passed")
        labels_df.to_parquet(output_path, write_index=False)
        Comms.destroy()

    @staticmethod
    def _sort_ids(df, id_columns):
        x = df[id_columns].values
        x = cp.sort(x, axis=1)
        for i, id_column in enumerate(id_columns):
            df[id_column] = x[:, i]
            df[id_column] = df[id_column].astype("uint64")
        return df

    @staticmethod
    def thresholding(df, threshold, column_to_threshold):
        mask = df[column_to_threshold] > threshold
        df.loc[mask, column_to_threshold] = np.int8(1)
        df.loc[~mask, column_to_threshold] = np.int8(0)
        return df

    def _write_dedup_encoded_jaccard_pair(self, encoded_jaccard_pair_path):
        output_path = f"{self.cache_dir}/final_dedup_encoded_jaccard_pair.parquet"

        ddf = dask_cudf.read_parquet(
            encoded_jaccard_pair_path, blocksize="512MB", aggregate_files=True
        )
        meta = {self.left_id: "uint64", self.right_id: "uint64", "jaccard": "float32"}
        ddf = ddf.map_partitions(
            ConnectedComponents._sort_ids,
            id_columns=[self.left_id, self.right_id],
            meta=meta,
        )
        ddf = ddf.map_partitions(
            ConnectedComponents.thresholding,
            threshold=self.jaccard_threshold,
            column_to_threshold="jaccard",
            meta=meta,
        )
        ddf = ddf.map_partitions(
            M.drop_duplicates,
            meta=ddf._meta,
            enforce_metadata=False,
            transform_divisions=False,
            align_dataframes=False,
        )

        ddf = dd_shuffle(
            ddf,
            [self.left_id, self.right_id],
            ignore_index=True,
            shuffle_method="tasks",
        )
        ddf = ddf.map_partitions(
            M.drop_duplicates,
            meta=ddf._meta,
            enforce_metadata=False,
            transform_divisions=False,
            align_dataframes=False,
        )
        ddf.to_parquet(output_path, write_index=False)
        return output_path

    def _convert_str_id_pair_to_int(self, df):
        for id, tag in zip([self.left_id, self.right_id], ["x", "y"]):
            dx = df[id].str.rsplit("-", n=1, expand=True)
            df[f"dataset_id_{tag}"] = dx[0].astype("uint32").values
            df[f"doc_id_{tag}"] = dx[1].astype("int64").values
            df = df.drop(columns=[id])
        return df

    def _write_dedup_parsed_id(self):
        dedup_parsed_id_path = f"{self.cache_dir}/dedup_parsed_id.parquet"
        ddf = dask_cudf.read_parquet(
            self.jaccard_pairs_path,
            columns=[self.left_id, self.right_id],
            blocksize="1GB",
            aggregate_files=True,
        )

        id_columns = [self.id_column]
        if self.convert_str_ids:
            ddf = ddf.map_partitions(
                self._convert_str_id_pair_to_int,
                meta={
                    "dataset_id_x": "uint32",
                    "doc_id_x": "int64",
                    "dataset_id_y": "uint32",
                    "doc_id_y": "int64",
                },
            )
            id_columns = ["dataset_id", "doc_id"]

        unique_docs = ddf.map_partitions(
            ConnectedComponents._get_unique_ids_per_partition, id_columns=id_columns
        )
        unique_docs = unique_docs.drop_duplicates(split_out=ddf.npartitions // 4)
        unique_docs["uid"] = np.uint64(1)
        unique_docs["uid"] = unique_docs["uid"].cumsum()
        unique_docs["uid"] = unique_docs["uid"] - 1
        unique_docs.to_parquet(dedup_parsed_id_path, write_index=False)
        return dedup_parsed_id_path

    def _write_encoded_jaccard_pair(self, dedup_parsed_id_path):
        output_path = f"{self.cache_dir}/encoded_jaccard_pair/"
        ddf_id = dask_cudf.read_parquet(
            dedup_parsed_id_path, blocksize="2GB", aggregate_files=True
        )
        ddf_id = ddf_id.persist()
        len(ddf_id)
        ddf = dask_cudf.read_parquet(
            self.jaccard_pairs_path,
            blocksize="256MB",
            aggregate_files=True,
        )
        id_columns = [self.id_column]
        if self.convert_str_ids:
            ddf = ddf.map_partitions(
                self._convert_str_id_pair_to_int,
                meta={
                    "jaccard": "float32",
                    "dataset_id_x": "uint32",
                    "doc_id_x": "int64",
                    "dataset_id_y": "uint32",
                    "doc_id_y": "int64",
                },
            )
            id_columns = ["dataset_id", "doc_id"]

        num_workers = get_num_workers(get_current_client())
        self._batched_merge_and_write(
            ddf=ddf,
            ddf_id=ddf_id,
            output_path=output_path,
            id_columns=id_columns,
            batch_size=num_workers,
        )
        return output_path

    def _batched_merge_and_write(
        self, ddf, ddf_id, output_path, id_columns, batch_size=32
    ):
        total_batches = (ddf.npartitions + batch_size - 1) // batch_size
        for batch_id, offset in enumerate(range(0, ddf.npartitions, batch_size)):
            st = time.time()
            subset_ddf = ddf.partitions[offset : offset + batch_size]
            for tag in ["x", "y"]:
                pair_ids = []
                for id_col in id_columns:
                    pair_ids.append(f"{id_col}_{tag}")
                subset_ddf = subset_ddf.merge(
                    ddf_id,
                    left_on=pair_ids,
                    right_on=id_columns,
                    how="inner",
                    broadcast=True,
                )
                subset_ddf = subset_ddf.drop(
                    columns=pair_ids,
                )
                subset_ddf = subset_ddf.rename(
                    columns={"uid": f"{self.id_column}_{tag}"}
                )

            subset_ddf = subset_ddf[[self.left_id, self.right_id, "jaccard"]]
            output_batch_path = os.path.join(output_path, f"{batch_id}.parquet")
            subset_ddf.to_parquet(output_batch_path, write_index=False)

            et = time.time()
            print(
                f"batch_id = {batch_id}/{total_batches}, time = {et - st}", flush=True
            )

    @staticmethod
    def _get_unique_ids_per_partition(df, id_columns):
        unique_df_ls = []
        for tag in ["x", "y"]:
            cols_to_drop = []
            for id_col in id_columns:
                cols_to_drop.append(f"{id_col}_{tag}")

            subset_df = df[cols_to_drop].drop_duplicates()
            subset_df = subset_df.rename(
                columns={f"{id_col}_{tag}": f"{id_col}" for id_col in id_columns}
            )
            unique_df_ls.append(subset_df)
        unique_df = cudf.concat(unique_df_ls, ignore_index=True)
        unique_df = unique_df.drop_duplicates()
        return unique_df
