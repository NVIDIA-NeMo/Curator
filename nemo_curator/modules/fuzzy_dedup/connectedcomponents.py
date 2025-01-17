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
from typing import Optional, Union

import cudf
import cugraph.dask as dcg
import cugraph.dask.comms.comms as Comms
import cupy as cp
import dask_cudf
import numpy as np
from cugraph import MultiGraph
from dask.utils import M

from nemo_curator.log import create_logger
from nemo_curator.utils.distributed_utils import performance_report_if_with_ts_suffix


class ConnectedComponents:
    def __init__(
        self,
        cache_dir: str,
        jaccard_pairs_path: str,
        id_column="id",
        jaccard_threshold: float = 0.8,
        logger: Union[logging.LoggerAdapter, str] = "./",
        profile_dir: Optional[str] = None,
    ):
        self.cache_dir = cache_dir
        self.jaccard_pairs_path = jaccard_pairs_path
        self.id_column = id_column
        self.left_id = f"{id_column}_x"
        self.right_id = f"{id_column}_y"
        self.jaccard_threshold = jaccard_threshold
        self.profile_dir = profile_dir
        if isinstance(logger, str):
            self._logger = create_logger(
                rank=0,
                log_file=os.path.join(logger, "ConnectedComponents.log"),
                name="ConnectedComponents",
            )
        else:
            self._logger = logger

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
        t0 = time.time()
        with performance_report_if_with_ts_suffix(
            self.profile_dir, "connected-components-run"
        ):

            Comms.initialize(p2p=False)
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
            n_components = len(
                result[["labels"]].drop_duplicates(split_out=max_partitions)
            )
            num_labels = len(result)
            labels_df = labels_df.merge(
                result, left_on=["uid"], right_on=["vertex"], how="inner"
            )
            id_columns = [self.id_column]
            labels_df = labels_df[id_columns + ["labels"]]
            labels_df = labels_df.rename(columns={"labels": "group"})
            labels_df = labels_df.persist()
            # Doing an inner merge above
            # should not change any rows

            self._logger.info(
                "Result of connected compoinents are "
                f"# of groups : {n_components}, "
                f"# of docs removed : {num_labels - n_components}, "
                f"# nodes = {num_nodes}, "
                f"# rows in labels_df = {len(labels_df)}"
            )
            assert num_nodes == len(labels_df)
            # Ensure all docs in the same group are in the same partition
            labels_df = labels_df.shuffle(on=["group"], ignore_index=True)
            labels_df.to_parquet(output_path, write_index=False, overwrite=True)
            Comms.destroy()
        self._logger.info(
            f"Time taken for Connected Components Run = {time.time() - t0}s and output written at {output_path}"
        )

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
        t0 = time.time()
        with performance_report_if_with_ts_suffix(
            self.profile_dir, "connected-components-dedup-encoded-jaccard-pair"
        ):

            ddf = dask_cudf.read_parquet(
                encoded_jaccard_pair_path, blocksize="512MB", aggregate_files=True
            )
            meta = {
                self.left_id: "uint64",
                self.right_id: "uint64",
                "jaccard": "float32",
            }
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

            ddf = ddf.shuffle(
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
            ddf.to_parquet(output_path, write_index=False, overwrite=True)
        self._logger.info(
            f"Time taken for Dedup Encoding Jaccard Pairs = {time.time() - t0}s and output written at {output_path}"
        )
        return output_path

    def _write_dedup_parsed_id(self):
        dedup_parsed_id_path = f"{self.cache_dir}/dedup_parsed_id.parquet"
        t0 = time.time()
        with performance_report_if_with_ts_suffix(
            self.profile_dir, "connected-components-dedup-parsed-id"
        ):
            ddf = dask_cudf.read_parquet(
                self.jaccard_pairs_path,
                columns=[self.left_id, self.right_id],
                blocksize="512MB",
                aggregate_files=True,
            )
            id_columns = [self.id_column]
            unique_docs = ddf.map_partitions(
                ConnectedComponents._get_unique_ids_per_partition, id_columns=id_columns
            )
            unique_docs = unique_docs.drop_duplicates(
                # Dask does not guard against split_out=0
                split_out=max(ddf.npartitions // 4, 1)
            )
            unique_docs["uid"] = np.uint64(1)
            unique_docs["uid"] = unique_docs["uid"].cumsum()
            unique_docs["uid"] = unique_docs["uid"] - 1
            unique_docs.to_parquet(
                dedup_parsed_id_path, write_index=False, overwrite=True
            )
        self._logger.info(
            f"Time taken for Dedup Parsed Id = {time.time() - t0}s and output written at {dedup_parsed_id_path}"
        )
        return dedup_parsed_id_path

    def _write_encoded_jaccard_pair(self, dedup_parsed_id_path):
        output_path = f"{self.cache_dir}/encoded_jaccard_pair/"
        t0 = time.time()
        with performance_report_if_with_ts_suffix(
            self.profile_dir, "connected-components-encoded-jaccard-pair"
        ):
            ddf_id = dask_cudf.read_parquet(
                dedup_parsed_id_path, blocksize="2GB", aggregate_files=True
            )
            ddf = dask_cudf.read_parquet(
                self.jaccard_pairs_path,
                blocksize="1GB",
                aggregate_files=True,
            )
            self._merge_and_write(
                ddf=ddf,
                ddf_id=ddf_id,
                output_path=output_path,
                id_column=self.id_column,
            )
        self._logger.info(
            f"Time taken for Encoding Jaccard Pairs = {time.time() - t0}s and output written at {output_path}"
        )
        return output_path

    def _merge_and_write(
        self,
        ddf: dask_cudf.DataFrame,
        ddf_id: dask_cudf.DataFrame,
        output_path: str,
        id_column: str,
    ) -> None:
        st = time.time()
        # Ensure 'id_columns' is a list
        ddf_id = ddf_id.set_index(id_column)
        for tag in ["x", "y"]:
            pair_id = f"{id_column}_{tag}"
            # Merge 'ddf' with 'ddf_id' to map ids to uids
            ddf = ddf.merge(
                ddf_id,
                left_on=pair_id,
                right_index=True,
                how="inner",
                broadcast=True,
            )
            ddf = ddf.drop(columns=pair_id)
            ddf = ddf.rename(columns={"uid": f"{self.id_column}_{tag}"})
        ddf = ddf[[self.left_id, self.right_id, "jaccard"]]
        ddf.to_parquet(output_path, write_index=False, overwrite=True)

        et = time.time()
        self._logger.info(
            f"Time taken for merge and write = {et - st}s and output written at {output_path}"
        )

    @staticmethod
    def _get_unique_ids_per_partition(df, id_columns):
        unique_df_ls = []
        for tag in ["x", "y"]:
            cols_to_drop = []
            for id_col in id_columns:
                cols_to_drop.append(f"{id_col}_{tag}")

            subset_df = df[cols_to_drop].drop_duplicates(ignore_index=True)
            subset_df = subset_df.rename(
                columns={f"{id_col}_{tag}": f"{id_col}" for id_col in id_columns}
            )
            unique_df_ls.append(subset_df)
        unique_df = cudf.concat(unique_df_ls, ignore_index=True)
        unique_df = unique_df.drop_duplicates(ignore_index=True)
        return unique_df
