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

import os
import time
import warnings
from contextlib import nullcontext
from datetime import datetime
from hashlib import md5
from typing import Union

import pandas as pd
from dask import config
from dask import dataframe as dd

from nemo_curator._compat import DASK_P2P_ERROR
from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.utils.distributed_utils import performance_report_if
from nemo_curator.utils.gpu_utils import is_cudf_type


class ExactDuplicates:
    """Find exact duplicates in a document corpus"""

    SUPPORTED_HASHES = {"md5"}

    def __init__(
        self,
        logger: Union[logging.LoggerAdapter, str] = "./",
        id_field: str = "id",
        text_field: str = "text",
        hash_method: str = "md5",
        profile_dir: str = None,
        cache_dir: str = None,
    ):
        """
        Parameters
        ----------
        logger: Existing logger to log to, or a path to a log directory.
        id_field: Column in the Dataset denoting document ID.
        text_field: Column in the Dataset denoting document content.
        hash_method: The hashing algorithm used for identifying exact duplicates. Currently supports {"md5"}
        profile_dir: str, Default None
          If specified directory to write dask profile
        cache_dir: str, Default None
          If specified, will compute & write duplicate id's to cache directory.
        """

        if hash_method not in self.SUPPORTED_HASHES:
            raise ValueError(
                f"{hash_method} not in supported hash_methods. Choose a hash_method from {self.SUPPORTED_HASHES}"
            )
        self.hash_method = hash_method
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
                log_file=os.path.join(logger, "ExactDuplicates.log"),
                name="ExactDuplicates",
            )
        else:
            self._logger = logger

    def _exact_dup_ids(self, df: dd.DataFrame):
        """
        Get the id's for text/documents that are exact duplicates
        Parameters
        ----------
        df: dask.dataframe.core.DataFrame
          A dataframe with the following requirements:
          * A column where each row is the text from one document
          * A unique ID column for each document
        """
        hash_df = self._compute_hashes(df)
        shuffle_context = (
            config.set({"dataframe.shuffle.method": "tasks"})
            if DASK_P2P_ERROR
            else nullcontext()
        )
        with shuffle_context:
            dup_ids = hash_df.shuffle(
                on=["_hashes"],
                ignore_index=True,
                npartitions=max(1, (hash_df.npartitions // 3)),
            ).map_partitions(lambda x: x[x["_hashes"].duplicated(keep=False)])
        return dup_ids

    def _compute_hashes(
        self,
        df: dd.DataFrame,
    ) -> dd.DataFrame:
        """
        Computes the hash of the text_column provided and returns a dataframe
        containing the id_column and relevant hashes in the _hashes column.
        """
        self._logger.info("Starting lazy hash generation")
        res = df[[self.id_field]]
        res["_hashes"] = df[self.text_field].map_partitions(self.hash_documents)
        self._logger.info(
            f"Lazy hash generation complete for {res.npartitions} partitions"
        )
        return res

    def hash_documents(
        self, df: Union[cudf.Series, pd.Series]
    ) -> Union[cudf.Series, pd.Series]:
        """
        Compute hashes for a Series containing documents
        """
        if is_cudf_type(df):
            return df.hash_values(method=self.hash_method)
        elif isinstance(df, pd.Series):
            # TODO: Generalize ty using self.hash_method
            return df.apply(lambda x: md5(x.encode()).hexdigest())

    def __call__(self, dataset: DocumentDataset) -> Union[DocumentDataset, str]:
        """
        Find document ID's for exact duplicates in a given DocumentDataset
        Parameters
        ----------
        dataset: DocumentDataset
          The input datset to find exact duplicates
        Returns
        -------
        DocumentDataset containing ID's and hashes of all duplicate documents
        """
        result = self._exact_dup_ids(df=dataset.df)

        if self.cache_dir is None:
            return DocumentDataset(result)

        t0 = time.time()
        self._logger.info("Starting execution for ExactDedup")
        write_path = os.path.join(self.cache_dir, "_exact_duplicates.parquet")
        if os.path.exists(write_path):
            warnings.warn(
                f"Output path f{write_path} already exists and will be overwritten"
            )
        with performance_report_if(
            self.profile_dir,
            f"exact-dedup-profile-{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        ):
            result.to_parquet(write_path, write_index=False, overwrite=True)
        self._logger.info(
            f"Exact dedup computation for dataset took {time.time() - t0}s complete at {write_path}"  # noqa:E501
        )
        if is_cudf_type(result):
            import dask_cudf

            result_dataset = dask_cudf.read_parquet(write_path, split_row_groups=False)
        else:
            result_dataset = dd.read_parquet(write_path)
        return DocumentDataset(result_dataset)
