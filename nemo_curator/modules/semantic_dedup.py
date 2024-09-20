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
from dataclasses import dataclass
from typing import List, Optional, Union

import cudf
import cupy as cp
import dask.bag as db
import dask.dataframe as dd
import dask_cudf
import numpy as np
import torch
import torch.nn as nn
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
from cuml.dask.cluster import KMeans
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.modules.config import SemDedupConfig
from nemo_curator.utils.distributed_utils import write_to_disk
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir
from nemo_curator.utils.semdedup_utils import (
    assign_and_sort_clusters,
    extract_dedup_data,
    get_semantic_matches_per_cluster,
)


# Embedding Creation Module
@dataclass
class EmbeddingConfig:
    model_name_or_path: str
    max_mem_gb: int
    max_seq_length: int = None

    def __post_init__(self):
        self.max_seq_length = AutoTokenizer.from_pretrained(
            self.model_name_or_path
        ).model_max_length
        # Gaurd against the HF bug
        # which sets max_seq_length to max(int) for some models
        if self.max_seq_length > 1e5:
            self.max_seq_length = AutoConfig.from_pretrained(
                self.model_name_or_path
            ).max_position_embeddings


class EmbeddingPytorchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(
            config.model_name_or_path, config=self.config, force_download=False
        )

    def feature(self, input_ids, attention_mask):
        with torch.autocast(device_type=input_ids.device.type):
            embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return embeddings

    @torch.no_grad()
    def forward(self, batch):
        feature = self.feature(batch["input_ids"], batch["attention_mask"])
        return self._mean_pooling(feature, batch["attention_mask"])

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return F.normalize(sum_embeddings / sum_mask, dim=1)


class EmbeddingCrossFitModel(HFModel):
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        super().__init__(
            self.config.model_name_or_path, max_mem_gb=self.config.max_mem_gb
        )

    def load_model(self, device="cuda"):
        model = EmbeddingPytorchModel(self.config)
        model = model.to(device)
        model.eval()
        return model

    def max_seq_length(self):
        return self.config.max_seq_length

    def load_config(self):
        return AutoConfig.from_pretrained(self.config.model_name_or_path)

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.config.model_name_or_path)


class EmbeddingCreator:
    def __init__(
        self,
        embedding_model_name_or_path: str,
        embedding_max_mem_gb: str,
        embedding_batch_size: int,
        embedding_output_dir: str,
        input_column: str = "text",
        embedding_column: str = "embeddings",
        write_embeddings_to_disk: bool = True,
        write_to_filename: bool = False,
        logger: Union[logging.Logger, str] = "./",
    ):
        """
        Initializes an EmbeddingCreator for generating embeddings using the specified model configurations.

        Args:
            embedding_model_name_or_path (str): The path or identifier for the model used to generate embeddings.
            embedding_max_mem_gb (str): Maximum memory usage for the embedding process.
            embedding_batch_size (int): Number of samples to process in each batch.
            embedding_output_dir (str): Directory path where embeddings will be saved.
            input_column (str): Column name from the data to be used for embedding generation, defaults to "text".
            write_embeddings_to_disk (bool, optional): If True, saves the embeddings to disk, defaults to True.
                                We recommend setting this to False when you have a delayed pipeline.
                                Setting it to False can lead to more memory overhead.
            write_to_filename (bool): If True, saves the embeddings to the same filename as input files, defaults to False.
            logger (Union[logging.Logger, str]): Logger object or path to store logs, defaults to "./".

        Attributes:
            embeddings_config (EmbeddingConfig): Configuration for embeddings.
            batch_size (int): Batch size for embedding generation.
            logger (logging.Logger): Logger instance for the class.
            embedding_output_dir (str): Output directory for embeddings.
            input_column (str): Input column for data processing.
            model (EmbeddingCrossFitModel): Model instance for embedding generation.
            write_to_filename (bool): If True, saves the embeddings to the same filename as input files, defaults to False.
        """

        self.embeddings_config = EmbeddingConfig(
            model_name_or_path=embedding_model_name_or_path,
            max_mem_gb=embedding_max_mem_gb,
        )
        self.batch_size = embedding_batch_size
        self.logger = self._setup_logger(logger)
        self.embedding_output_dir = embedding_output_dir
        self.input_column = input_column
        self.embedding_column = embedding_column
        self.model = EmbeddingCrossFitModel(self.embeddings_config)
        self.write_embeddings_to_disk = write_embeddings_to_disk
        self.write_to_filename = write_to_filename

    def _setup_logger(self, logger):
        if isinstance(logger, str):
            return create_logger(
                rank=0,
                name="compute-embeddings",
                log_file=os.path.join(logger, "compute_embeddings.log"),
                log_level=logging.INFO,
                stdout=True,
            )
        else:
            return logger

    def create_embeddings(
        self, ddf: dask_cudf.DataFrame, input_column="text"
    ) -> dask_cudf.DataFrame:
        pipe = op.Sequential(
            op.Tokenizer(
                self.model,
                cols=[input_column],
                tokenizer_type="sentencepiece",
                max_length=self.embeddings_config.max_seq_length,
            ),
            op.Predictor(
                self.model,
                sorted_data_loader=True,
                batch_size=self.batch_size,
                pred_output_col=self.embedding_column,
            ),
            keep_cols=ddf.columns.tolist(),
        )
        return pipe(ddf)

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        embedding_ddf = self.create_embeddings(dataset.df, self.input_column)
        if self.write_embeddings_to_disk:
            write_to_disk(
                embedding_ddf,
                self.embedding_output_dir,
                write_to_filename=self.write_to_filename,
                output_type="parquet",
            )
            return DocumentDataset(
                dask_cudf.read_parquet(
                    self.embedding_output_dir, blocksize="2GB", aggregate_files=True
                )
            )
        else:
            return DocumentDataset(embedding_ddf)


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
        id_col: str,
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
    ):
        """
        Initializes the ClusteringModel with the provided settings for semantic clustering to help semantic deduplication.

        Args:
            id_col (str): Column name used as the identifier in the dataset.
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

        This constructor sets up the parameters required for clustering operations.
        """
        self.id_col = id_col
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

        embeddings_df = embeddings_df[[self.id_col, self.embedding_col]]

        embeddings_df = embeddings_df.to_backend("pandas").persist()
        embeddings_df = embeddings_df.repartition(partition_size=self.partition_size)
        embeddings_df = embeddings_df.to_backend("cudf")

        cupy_darr = embeddings_df.map_partitions(
            get_embedding_ar, self.embedding_col, meta=cp.ndarray([1, 1])
        )
        cupy_darr.compute_chunk_sizes()

        kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter)
        self.logger.info("KMeans starting fit")
        kmeans.fit(cupy_darr)
        self.logger.info("KMeans fit complete")

        self.logger.info(
            "Computing nearest centroids + distance to centers using kmeans.predict"
        )
        nearest_cents = kmeans.predict(cupy_darr)
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
        centroids = kmeans.cluster_centers_
        embeddings_df = embeddings_df.reset_index(drop=True)
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
            f"Saved embeddings by nearest center to {clustering_output_dir}"
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
            )

        fps = [
            os.path.join(clustering_output_dir, file_name)
            for file_name in os.listdir(clustering_output_dir)
        ]
        embeddings_df = dd.from_map(cudf.read_parquet, fps)
        return DocumentDataset(embeddings_df)


class SemanticClusterLevelDedup:
    def __init__(
        self,
        n_clusters: int,
        emb_by_clust_dir: str,
        sorted_clusters_dir: str,
        id_col: str,
        id_col_type: str,
        which_to_keep: str,
        output_dir: str,
        embedding_col: str = "embeddings",
        logger: Union[logging.Logger, str] = "./",
    ) -> None:
        """
        Initialize the SemanticClusterLevelDedup class.

        Args:
            n_clusters (int): Number of clusters.
            emb_by_clust_dir (str): Directory containing embeddings by cluster.
            sorted_clusters_dir (str): Directory containing sorted clusters.
            id_col (str): Column name for IDs.
            id_col_type (str): Data type of the ID column.
            which_to_keep (str): Strategy for which duplicate to keep.
            output_dir (str): Directory to save output files.
            embedding_col (str): Column where the embeddings are stored.
            logger (Union[logging.Logger, str]): Logger instance or path to the log file directory.
        """
        self.n_clusters = n_clusters
        self.emb_by_clust_dir = emb_by_clust_dir
        self.sorted_clusters_dir = sorted_clusters_dir
        self.id_col = id_col
        self.id_col_type = id_col_type
        self.which_to_keep = which_to_keep
        self.output_dir = output_dir
        self.semdedup_pruning_tables_dir = os.path.join(
            output_dir, "semdedup_pruning_tables"
        )
        self.computed_semantic_match_dfs = False
        self.embedding_col = embedding_col
        self.logger = self._setup_logger(logger)

    def _setup_logger(self, logger: Union[logging.Logger, str]) -> logging.Logger:
        """
        Set up the logger.

        Args:
            logger (Union[logging.Logger, str]): Logger instance or path to the log file directory.

        Returns:
            logging.Logger: Configured logger.
        """
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

    def compute_semantic_match_dfs(
        self, eps_list: Optional[List[float]] = None
    ) -> None:
        """
        Compute semantic match dataframes for clusters.

        Args:
            eps_list (Optional[List[float]]): List of epsilon values for clustering.
        """
        if eps_list is None:
            eps_list1 = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]
            eps_list2 = [0.1 + x * 0.005 for x in range(34)]
            eps_list = eps_list1 + eps_list2

        if os.path.exists(self.semdedup_pruning_tables_dir):
            self.logger.info(
                f"Removing existing directory {self.semdedup_pruning_tables_dir}"
            )
            shutil.rmtree(self.semdedup_pruning_tables_dir)
        expand_outdir_and_mkdir(self.semdedup_pruning_tables_dir)

        tasks = db.from_sequence(
            list(range(self.n_clusters)), npartitions=self.n_clusters
        ).map(
            lambda cluster_id: get_semantic_matches_per_cluster(
                cluster_id=cluster_id,
                emb_by_clust_dir=self.emb_by_clust_dir,
                sorted_clusters_dir=self.sorted_clusters_dir,
                id_col=self.id_col,
                id_col_type=self.id_col_type,
                eps_list=eps_list,
                output_dir=self.semdedup_pruning_tables_dir,
                embedding_col=self.embedding_col,
                which_to_keep=self.which_to_keep,
            )
        )
        tasks.compute()
        self.computed_semantic_match_dfs = True

    def extract_dedup_data(self, eps_to_extract: float) -> DocumentDataset:
        """
        Extract deduplicated data based on epsilon value.

        Args:
            eps_to_extract (float): Epsilon threshold for extracting deduplicated data.

        Returns:
            DocumentDataset: Dataset containing deduplicated documents.
        """
        if not self.computed_semantic_match_dfs:
            raise ValueError(
                "Run compute_semantic_match_dfs before calling extract_dedup_data"
            )

        output_summary_file = os.path.join(
            self.output_dir, f"dedup_summary_{eps_to_extract}.csv"
        )
        output_parquet_path = os.path.join(
            self.output_dir, f"unique_ids_{eps_to_extract}.parquet"
        )
        extract_dedup_data(
            eps=eps_to_extract,
            n_clusters=self.n_clusters,
            id_col=self.id_col,
            id_col_type=self.id_col_type,
            sorted_clusters_dir=self.sorted_clusters_dir,
            semdedup_pruning_tables_dir=self.semdedup_pruning_tables_dir,
            output_summary_file=output_summary_file,
            output_parquet_path=output_parquet_path,
            logger=self.logger,
        )

        fps = [
            os.path.join(output_parquet_path, file_name)
            for file_name in os.listdir(output_parquet_path)
        ]
        return DocumentDataset.read_parquet(fps, backend="cudf")


class SemDedup:
    def __init__(
        self,
        config: SemDedupConfig,
        logger: Union[logging.Logger, str] = "./",
    ) -> None:
        """
        Initialize the SemDedup class.

        Args:
            config (SemDedupConfig): Configuration for SemDedup.
            logger (Union[logging.Logger, str]): Logger instance or path to the log file directory.
        """
        self.config = config
        self.logger = logger
        cache_dir = config.cache_dir
        self.embedding_creator = EmbeddingCreator(
            embedding_model_name_or_path=config.embedding_model_name_or_path,
            embedding_max_mem_gb=config.embedding_max_mem_gb,
            embedding_batch_size=config.embedding_batch_size,
            input_column=config.input_column,
            embedding_output_dir=os.path.join(cache_dir, config.embeddings_save_loc),
            logger=logger,
        )
        self.clustering_model = ClusteringModel(
            id_col=config.id_col_name,
            max_iter=config.max_iter,
            n_clusters=config.n_clusters,
            clustering_output_dir=os.path.join(cache_dir, config.clustering_save_loc),
            logger=logger,
        )
        self.semantic_cluster_dedup = SemanticClusterLevelDedup(
            n_clusters=config.n_clusters,
            emb_by_clust_dir=os.path.join(
                cache_dir, config.clustering_save_loc, "embs_by_nearest_center"
            ),
            sorted_clusters_dir=os.path.join(
                cache_dir, config.clustering_save_loc, "sorted"
            ),
            id_col=config.id_col_name,
            id_col_type=config.id_col_type,
            which_to_keep=config.which_to_keep,
            output_dir=os.path.join(cache_dir, config.clustering_save_loc),
            logger=logger,
        )
        self.eps_thresholds = config.eps_thresholds
        self.eps_to_extract = config.eps_to_extract

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        """
        Execute the SemDedup process.

        Args:
            dataset (DocumentDataset): Input dataset for deduplication.

        Returns:
            DocumentDataset: Deduplicated dataset.
        """
        embeddings_dataset = self.embedding_creator(dataset)
        self.clustering_model(embeddings_dataset)
        self.semantic_cluster_dedup.compute_semantic_match_dfs(self.eps_thresholds)
        return self.semantic_cluster_dedup.extract_dedup_data(
            eps_to_extract=self.eps_to_extract
        )
