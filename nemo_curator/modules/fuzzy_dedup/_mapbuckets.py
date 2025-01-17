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
import os
from typing import Union

import cudf
import dask_cudf
import numpy as np
from dask.utils import M

from nemo_curator.log import create_logger
from nemo_curator.utils.fuzzy_dedup_utils.output_map_utils import (
    build_partition,
    get_agg_text_bytes_df,
)


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
        max_text_bytes_per_part = int(np.iinfo(np.int32).max * 3)

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
        ddf_anchor_docs_with_bk = ddf_anchor_docs_with_bk.shuffle(
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
