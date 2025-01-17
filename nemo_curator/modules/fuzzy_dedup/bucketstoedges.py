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
import time
import warnings
from itertools import pairwise
from typing import Optional, Union

import cudf
import dask_cudf
import numpy as np
import pandas as pd
import pyarrow as pa

from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.utils.distributed_utils import performance_report_if_with_ts_suffix


class BucketsToEdges:
    """
    Maps buckets generated from LSH into an edgelist that
    can be processed further by Connected Components to find duplicate
    documents
    """

    def __init__(
        self,
        cache_dir: str = None,
        id_fields: Union[list, str] = "id",
        str_id_name: str = "id",
        bucket_field: str = "_bucket_id",
        logger: Union[logging.LoggerAdapter, str] = "./",
        profile_dir: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        cache_dir: str or None
          If specified, will compute & write the edgelist to a file
        id_fields: list or str
          id fields of documents in buckets_df
        str_id_name: str
          Ignored if there is a single id field. Multiple id fields
          will be combined into a single id field with the given name.
        bucket_field: str
          Column denoting bucket ID
        num_buckets: Number of bands/buckets to create from the minhash signature.
          Hashes_per_signature = num_hashes / num_buckets
        """
        self.cache_dir = cache_dir
        self.id_fields = [id_fields] if isinstance(id_fields, str) else id_fields
        self.str_id_name = str_id_name if len(self.id_fields) > 1 else self.id_fields[0]
        self.output_ids = [f"{self.str_id_name}_x", f"{self.str_id_name}_y"]
        self.bucket_field = bucket_field
        self.profile_dir = profile_dir
        if isinstance(logger, str):
            self._logger = create_logger(
                rank=0,
                log_file=os.path.join(logger, "Buckets_to_Edges.log"),
                name="Buckets_to_Edges",
            )
        else:
            self._logger = logger

    @staticmethod
    def _combine_multiple_ids(
        input_df: cudf.DataFrame, input_id_fields: list, output_id_field: str
    ) -> cudf.DataFrame:
        if output_id_field in input_df.columns:
            raise ValueError(
                f"Input df already contains column named: {output_id_field}"
            )

        output_df = input_df.copy()[input_df.columns.difference(input_id_fields)]

        output_df[output_id_field] = input_df[input_id_fields[0]].astype(str)
        for input_field in input_id_fields[1:]:
            output_df[output_id_field] = output_df[output_id_field] = (
                input_df[input_id_fields[0]].astype(str)
                + "-"
                + input_df[input_field].astype(str)
            )

        return output_df

    def buckets_to_edges(
        self,
        buckets_df: cudf.DataFrame,
    ) -> cudf.DataFrame:

        grouped_buckets = (
            buckets_df.groupby(self.bucket_field)[self.str_id_name]
            .agg(list)
            .list.sort_values()
        )
        bucket_docs = grouped_buckets.to_arrow().to_pylist()
        edges = []
        # Create pairs of all documents within a bucket since they are near duplicates
        # Effectively create a edge list of all near duplicate documents
        for bucket_doc in bucket_docs:
            edges.extend(pairwise(bucket_doc))
        edges = pd.DataFrame(edges, columns=self.output_ids)
        edges = pa.Table.from_pandas(edges)
        result_df = cudf.DataFrame.from_arrow(edges)
        del edges
        result_df = result_df.drop_duplicates(self.output_ids).reset_index(drop=True)
        result_df["jaccard"] = np.float32(1.0)
        return result_df

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        buckets_df = dataset.df
        self._logger.info(f"Starting conversion of LSH Buckets to Graph Edgelist")
        if len(self.id_fields) > 1:
            buckets_df = buckets_df.map_partitions(
                BucketsToEdges._combine_multiple_ids,
                input_id_fields=self.id_fields,
                output_id_field=self.str_id_name,
            )

        meta = [(output_id, str) for output_id in self.output_ids]
        meta.append(("jaccard", np.float32))
        edges_df = buckets_df.map_partitions(self.buckets_to_edges, meta=meta)

        if self.cache_dir is None:
            return DocumentDataset(edges_df)

        write_path = os.path.join(self.cache_dir, "_edges.parquet")
        if os.path.exists(write_path):
            warnings.warn(
                f"Output path {write_path} already exists and will be overwritten"
            )
        t0 = time.time()
        with performance_report_if_with_ts_suffix(
            self.profile_dir,
            "bucket-to-edges",
        ):
            edges_df.to_parquet(write_path, write_index=False, overwrite=True)
        self._logger.info(
            f"Time taken for Converted Buckets To Edgelist = {time.time() - t0}s and output written at {write_path}"
        )

        return DocumentDataset(
            dask_cudf.read_parquet(write_path, split_row_groups=False)
        )
