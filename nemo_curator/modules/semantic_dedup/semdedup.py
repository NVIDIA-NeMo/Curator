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
from typing import Union

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules.config import SemDedupConfig
from nemo_curator.modules.semantic_dedup.clusteringmodel import ClusteringModel
from nemo_curator.modules.semantic_dedup.embeddings import EmbeddingCreator
from nemo_curator.modules.semantic_dedup.semanticclusterleveldedup import (
    SemanticClusterLevelDedup,
)


class SemDedup:
    def __init__(
        self,
        config: SemDedupConfig,
        input_column: str = "text",
        id_column: str = "id",
        id_column_type: str = "int",
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
            embedding_batch_size=config.embedding_batch_size,
            input_column=input_column,
            embedding_output_dir=os.path.join(cache_dir, config.embeddings_save_loc),
            write_embeddings_to_disk=config.write_embeddings_to_disk,
            logger=logger,
            profile_dir=self.config.profile_dir,
        )
        self.clustering_model = ClusteringModel(
            id_column=id_column,
            max_iter=config.max_iter,
            n_clusters=config.n_clusters,
            clustering_output_dir=os.path.join(cache_dir, config.clustering_save_loc),
            logger=logger,
            profile_dir=self.config.profile_dir,
        )
        self.semantic_cluster_dedup = SemanticClusterLevelDedup(
            n_clusters=config.n_clusters,
            emb_by_clust_dir=os.path.join(
                cache_dir, config.clustering_save_loc, "embs_by_nearest_center"
            ),
            sorted_clusters_dir=os.path.join(
                cache_dir, config.clustering_save_loc, "sorted"
            ),
            id_column=id_column,
            id_column_type=id_column_type,
            which_to_keep=config.which_to_keep,
            output_dir=os.path.join(cache_dir, config.clustering_save_loc),
            logger=logger,
            profile_dir=self.config.profile_dir,
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
