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

import logging
import math
import os
import time
import warnings
from typing import List, Optional, Tuple, Union

import cudf
import dask_cudf
import numpy as np

from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.utils.distributed_utils import performance_report_if_with_ts_suffix
from nemo_curator.utils.fuzzy_dedup_utils.io_utils import check_empty_buckets


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
        false_positive_check: bool = False,
        logger: Union[logging.LoggerAdapter, str] = "./",
        id_fields: Union[str, list] = "id",
        minhash_field: str = "_minhash_signature",
        profile_dir: Optional[str] = None,
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
          but might lead to memory pressures and related errors.
        false_positive_check: bool
          If True, writes out buckets in a format compatible with downstream false positive check.
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
        self.buckets_as_int = false_positive_check

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
    ) -> bool:
        """
        Computes hash buckets for the DataFrame and writes them as parquet files to the specified path.

        Parameters:
            - write_path (str): The directory path to write parquet files.
            - df (dask_cudf.DataFrame): The input DataFrame with minhashes to be bucketed.
        Returns:
            are_buckets_empty: True if buckets were empty (no duplicates found), False otherwise.
        """
        wrote_buckets = False
        are_buckets_empty = True

        meta = self._minhash_to_bucket_meta(df)
        df = df.map_partitions(
            self.minhash_to_buckets,
            bucket_ranges=self.bucket_ranges,
            meta=meta,
        )
        bucket_start_id = 0
        for i in range(0, self.num_buckets, self.buckets_per_shuffle):
            bucket_columns = [
                f"_bucket_{i}"
                for i in range(i, min(self.num_buckets, i + self.buckets_per_shuffle))
            ]
            df2 = df.melt(
                id_vars=self.id_fields,
                value_name="_bucket_id",
                value_vars=bucket_columns,
            )[self.id_fields + ["_bucket_id"]]

            df2 = df2.shuffle(
                on=["_bucket_id"],
                ignore_index=True,
                npartitions=max(1, 2 ** math.floor(math.log2(df2.npartitions))),
            ).map_partitions(lambda x: x[x["_bucket_id"].duplicated(keep=False)])

            df2 = df2.reset_index(drop=True)
            # Buckets to Int
            if self.buckets_as_int:
                df2, end_id = self.bucket_id_to_int(
                    df2, bucket_col_name="_bucket_id", start_id=bucket_start_id
                )
                # If bucketing return empty dataframe
                if end_id < bucket_start_id:
                    self._logger.info(
                        f"No duplicate documents found for buckets: {bucket_columns}"
                    )
                    continue
                bucket_start_id = end_id + 1
                are_buckets_empty = False

            wrote_buckets, are_buckets_empty = self._write_bucket_parquet(
                df2,
                write_path,
                wrote_buckets,
                are_buckets_empty,
                bucket_columns,
            )

        if are_buckets_empty:
            self._logger.info("No duplicate documents found during LSH")
            if os.path.exists(write_path):
                import shutil

                shutil.rmtree(write_path)

        return are_buckets_empty

    def _write_bucket_parquet(
        self,
        df: dask_cudf.DataFrame,
        write_path: str,
        wrote_buckets: bool,
        are_buckets_empty: bool,
        buckets_to_write: List[str],
    ) -> tuple[bool, bool]:
        """
        Utility function to write the bucketed data to parquet
        handling cases of overwriting and appending as needed.
        """
        if not wrote_buckets:
            if os.path.exists(write_path):
                warnings.warn(
                    f"Output path {write_path} already exists and will be overwritten"
                )
            df.to_parquet(write_path, write_index=False, overwrite=True)
        else:
            df.to_parquet(
                write_path,
                write_index=False,
                overwrite=are_buckets_empty,
                append=not are_buckets_empty,
                ignore_divisions=True,
            )
        # Only check if buckets written so far are empty
        if are_buckets_empty:
            are_buckets_empty = check_empty_buckets(write_path)
        wrote_buckets = True

        if are_buckets_empty:
            self._logger.info(
                f"No duplicate documents found for buckets: {buckets_to_write}"
            )
        else:
            self._logger.info(f"Wrote data for buckets: {buckets_to_write}")
        return wrote_buckets, are_buckets_empty

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        df = dataset.df

        write_path = os.path.join(self.cache_dir, "_buckets.parquet")
        t0 = time.time()
        with performance_report_if_with_ts_suffix(self.profile_dir, "lsh-profile"):
            empty_result = self.lsh(write_path=write_path, df=df)
        self._logger.info(
            f"Time taken for LSH = {time.time() - t0}s and output written at {write_path}"
        )

        if empty_result:
            return None

        buckets_df = dask_cudf.read_parquet(write_path, split_row_groups=False)
        return DocumentDataset(buckets_df)
