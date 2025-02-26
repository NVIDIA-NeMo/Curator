# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


# Clustering module
class ClusteringModel:
    def __init__(
        self,
        id_column: str = "id",
        max_iter: int = 100,
        n_clusters: int = 1000,
        clustering_output_dir: str = "./clustering_results",
        embedding_column: str = "embeddings",
        random_state: int = 1234,
        sim_metric: str = "cosine",
        which_to_keep: str = "hard",
        sort_clusters: bool = True,
        kmeans_with_cos_dist: bool = False,
        clustering_input_partition_size: str = "2gb",
        logger: Union[logging.Logger, str] = "./",
        profile_dir: Optional[str] = None,
    ):
        """
        Initializes the ClusteringModel with the provided settings for semantic clustering to help semantic deduplication.

        Args:
            id_column (str): Column name used as the identifier in the dataset.
                Default is "id".
            max_iter (int): Maximum iterations for clustering. Default is 100.
            n_clusters (int): Number of clusters. Default is 1000.
            clustering_output_dir (str): Location to save clustering results.
                Default is "./clustering_results".
            embedding_column (str): The column name that stores the embeddings.
                Default is "embeddings".
            random_state (int): KMeans random state used for reproducibility.
                Default is 1234.
            sim_metric (str): Similarity metric for deduplication.
                Default is "cosine".
            which_to_keep (str): Method to determine which duplicates to keep.
                Default is "hard".
            sort_clusters (bool): Whether to sort clusters. Default is True.
            kmeans_with_cos_dist (bool): Whether or not to use KMeans with cosine distance.
                Default is False.
            clustering_input_partition_size (str): The size of data partition with which to run KMeans.
                Default is "2gb".
            logger (Union[logging.Logger, str]): Existing logger to log to, or a path to a log directory.
                Default is "./".
            profile_dir (Optional[str]): If specified, directory to write Dask profile.
                Default is None.

        """
        self.id_col = id_column
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.clustering_output_dir = clustering_output_dir
        self.embedding_column = embedding_column
        self.random_state = random_state
        self.sim_metric = sim_metric
        self.keep_hard = which_to_keep == "hard"
        self.kmeans_with_cos_dist = kmeans_with_cos_dist
        self.clustering_input_partition_size = clustering_input_partition_size
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

        if self.embedding_column not in embeddings_df.columns:
            raise ValueError(
                f'Expected embedding column "{self.embedding_column}"'
                f" to be in dataset. Only found columns {embeddings_df.columns}"
            )

        with performance_report_if_with_ts_suffix(self.profile_dir, "clustering-model"):
            embeddings_df = embeddings_df[[self.id_col, self.embedding_column]]
            embeddings_df = embeddings_df.repartition(
                partition_size=self.clustering_input_partition_size
            )

            try:
                embeddings_df = embeddings_df.to_backend("pandas").persist()
                embeddings_length = embeddings_df.shape[0].compute()

                if embeddings_length < self.n_clusters:
                    raise ValueError(
                        "Number of clusters is greater than the number of documents in your dataset: "
                        f"dataset length is {embeddings_length} while n_clusters is set to {self.n_clusters}. "
                        f"Please reduce n_clusters to be less than or equal to {embeddings_length}."
                    )
            except IndexError as e:
                raise IndexError(
                    f'Original error message: "{e}". '
                    "This could be due to empty partitions in your DocumentDataset. "
                    "Please check your dataset for empty partitions and remove them if necessary."
                )

            embeddings_df = embeddings_df.to_backend("cudf")

            cupy_darr = embeddings_df.map_partitions(
                get_embedding_ar, self.embedding_column, meta=cp.ndarray([1, 1])
            )
            cupy_darr.compute_chunk_sizes()
            t0 = time.time()
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            self.logger.info("KMeans starting fit")
            kmeans.fit(cupy_darr)
            self.logger.info("KMeans fit complete")
            self.logger.info(f"Time taken for KMeans fit: {time.time() - t0}")

            self.logger.info(
                "Computing nearest centroids and distance to centers using kmeans.predict"
            )
            t0 = time.time()
            nearest_cents = kmeans.predict(cupy_darr)
            self.logger.info(f"Time taken for KMeans predict: {time.time() - t0}")

            t0 = time.time()
            embeddings_df["nearest_cent"] = nearest_cents.astype(np.int32)
            del nearest_cents
            meta_df = embeddings_df._meta.copy()
            meta_df["dist_to_cent"] = cp.zeros(1)
            embeddings_df = embeddings_df.map_partitions(
                add_dist_to_cents,
                embedding_col=self.embedding_column,
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
                clustering_output_dir, index=False, partition_on="nearest_cent"
            )
            self.logger.info(
                f"Time taken for assigning distance to each embedding: {time.time() - t0}s"
                f" and output written at {clustering_output_dir}"
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
                embedding_col=self.embedding_column,
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
