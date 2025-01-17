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


import logging
import os
import shutil
import time
from typing import Optional, Union

import cudf
import cupy as cp
import dask.dataframe as dd
import numpy as np
from cuml.dask.cluster import KMeans

from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.utils.distributed_utils import performance_report_if_with_ts_suffix
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir
from nemo_curator.utils.semdedup_utils import assign_and_sort_clusters


### Clustering Module
def get_embedding_ar(df: "cudf.DataFrame", embedding_col: str) -> cp.ndarray:
    return df[embedding_col].list.leaves.values.reshape(len(df), -1)


def add_dist_to_cents(
    df: "cudf.DataFrame", embedding_col: str, centroids: cp.ndarray
) -> "cudf.DataFrame":
    embed_array = get_embedding_ar(df, embedding_col)
    centroids_ar = centroids[df["nearest_cent"].values]
    dist_to_cents = cp.sqrt(np.sum((embed_array - centroids_ar) ** 2, axis=1))
    df["dist_to_cent"] = dist_to_cents
    return df


class ClusteringModel:
    def __init__(
        self,
        id_column: str,
        max_iter: int,
        n_clusters: int,
        clustering_output_dir: str,
        embedding_col: str = "embeddings",
        sim_metric: str = "cosine",
        which_to_keep: str = "hard",
        sort_clusters: bool = True,
        kmeans_with_cos_dist: bool = False,
        partition_size: str = "2gb",
        logger: Union[logging.Logger, str] = "./",
        profile_dir: Optional[str] = None,
    ):
        """
        Initializes the ClusteringModel with the provided settings for semantic clustering to help semantic deduplication.

        Args:
            id_column (str): Column name used as the identifier in the dataset.
            max_iter (int): Maximum number of iterations for the clustering algorithm.
            n_clusters (int): The number of clusters to form.
            clustering_output_dir (str): Directory path where clustering results will be saved.
            embedding_col (str): Column name where the embeddings are stored.
            sim_metric (str): Similarity metric to use for clustering, default is "cosine".
            which_to_keep (str): Strategy to decide which duplicates to keep; default is "hard".
            sort_clusters (bool): Whether to sort clusters, default is True.
            kmeans_with_cos_dist (bool): Whether to use KMeans with cosine distance, default is False.
            partition_size (str): The size of data partition to run kmeans with, default is "2gb".
            logger (Union[logging.Logger, str]): Logger object or directory path to save logs; default is "./".
            profile_dir (str): If specified directory to write dask profile. Default is None.

        This constructor sets up the parameters required for clustering operations.
        """
        self.id_col = id_column
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.clustering_output_dir = clustering_output_dir
        self.embedding_col = embedding_col
        self.sim_metric = sim_metric
        self.keep_hard = which_to_keep == "hard"
        self.kmeans_with_cos_dist = kmeans_with_cos_dist
        self.partition_size = partition_size
        self.sort_clusters = sort_clusters
        self.logger = self._setup_logger(logger)
        self.profile_dir = profile_dir

        if not os.path.exists(self.clustering_output_dir):
            expand_outdir_and_mkdir(self.clustering_output_dir)
        else:
            self.logger.warning(
                f"Clustering output directory {self.clustering_output_dir} already exists and will be overwritten"
            )

    def _setup_logger(self, logger):
        if isinstance(logger, str):
            return create_logger(
                rank=0,
                name="SemanticClusterLevelDedup",
                log_file=os.path.join(logger, "SemanticClusterLevelDedup.log"),
                log_level=logging.INFO,
                stdout=True,
            )
        else:
            return logger

    def __call__(self, embeddings_dataset: DocumentDataset):
        embeddings_df = embeddings_dataset.df

        if self.embedding_col not in embeddings_df.columns:
            raise ValueError(
                f"Expected embedding column '{self.embedding_col}'"
                f" to be in dataset. Only found columns {embeddings_df.columns}"
            )

        with performance_report_if_with_ts_suffix(self.profile_dir, "clustering-model"):
            embeddings_df = embeddings_df[[self.id_col, self.embedding_col]]
            embeddings_df = embeddings_df.repartition(
                partition_size=self.partition_size
            )
            embeddings_df = embeddings_df.to_backend("pandas").persist()
            embeddings_df = embeddings_df.to_backend("cudf")

            cupy_darr = embeddings_df.map_partitions(
                get_embedding_ar, self.embedding_col, meta=cp.ndarray([1, 1])
            )
            cupy_darr.compute_chunk_sizes()
            t0 = time.time()
            kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter)
            self.logger.info("KMeans starting fit")
            kmeans.fit(cupy_darr)
            self.logger.info("KMeans fit complete")
            self.logger.info(f"Time taken for KMeans Fit: {time.time() - t0}")

            self.logger.info(
                "Computing nearest centroids + distance to centers using kmeans.predict"
            )
            t0 = time.time()
            nearest_cents = kmeans.predict(cupy_darr)
            self.logger.info(f"Time taken for KMeans Predict: {time.time() - t0}")

            t0 = time.time()
            embeddings_df["nearest_cent"] = nearest_cents.astype(np.int32)
            del nearest_cents
            meta_df = embeddings_df._meta.copy()
            meta_df["dist_to_cent"] = cp.zeros(1)
            embeddings_df = embeddings_df.map_partitions(
                add_dist_to_cents,
                embedding_col=self.embedding_col,
                centroids=kmeans.cluster_centers_,
                meta=meta_df,
            )
            embeddings_df = embeddings_df.reset_index(drop=True)
            centroids = kmeans.cluster_centers_
            kmeans_centroids_file = os.path.join(
                self.clustering_output_dir, "kmeans_centroids.npy"
            )
            np.save(kmeans_centroids_file, centroids)
            self.logger.info("Saving centroids complete")
            del kmeans, cupy_darr, centroids

            clustering_output_dir = os.path.join(
                self.clustering_output_dir, "embs_by_nearest_center"
            )
            if os.path.exists(clustering_output_dir):
                self.logger.warning(
                    f"Output directory {clustering_output_dir} already exists and will be overwritten"
                )
                shutil.rmtree(clustering_output_dir)

            embeddings_df.to_parquet(
                clustering_output_dir,
                index=False,
                partition_on="nearest_cent",
            )
            self.logger.info(
                f"Time taken for Assigning distance to each embedding : {time.time() - t0} "
                f"and output written at {clustering_output_dir}"
            )

            del embeddings_df

        if self.sort_clusters:
            assign_and_sort_clusters(
                id_col=self.id_col,
                kmeans_centroids_file=kmeans_centroids_file,
                nearest_cent_dir=clustering_output_dir,
                output_sorted_clusters_dir=os.path.join(
                    self.clustering_output_dir, "sorted"
                ),
                embedding_col=self.embedding_col,
                sim_metric=self.sim_metric,
                keep_hard=self.keep_hard,
                kmeans_with_cos_dist=self.kmeans_with_cos_dist,
                cluster_ids=range(self.n_clusters),
                logger=self.logger,
                profile_dir=self.profile_dir,
            )

        fps = [
            os.path.join(clustering_output_dir, file_name)
            for file_name in os.listdir(clustering_output_dir)
        ]
        embeddings_df = dd.from_map(cudf.read_parquet, fps)
        return DocumentDataset(embeddings_df)
