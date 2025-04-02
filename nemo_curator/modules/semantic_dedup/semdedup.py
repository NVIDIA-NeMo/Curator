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
from typing import Union

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules.base import BaseDeduplicationModule
from nemo_curator.modules.config import SemDedupConfig
from nemo_curator.modules.semantic_dedup.clusteringmodel import ClusteringModel
from nemo_curator.modules.semantic_dedup.embeddings import EmbeddingCreator
from nemo_curator.modules.semantic_dedup.semanticclusterleveldedup import (
    SemanticClusterLevelDedup,
)
from nemo_curator.utils.duplicates_removal import remove_duplicates


class SemDedup(BaseDeduplicationModule):
    def __init__(
        self,
        config: SemDedupConfig,
        input_column: str = "text",
        id_column: str = "id",
        perform_removal: bool = False,
        logger: Union[logging.Logger, str] = "./",
    ) -> None:
        """
        Initialize the SemDedup class.

        Args:
            config (SemDedupConfig): Configuration for SemDedup.
            input_column (str): Column name from the data to be used for embedding generation.
                Default is "text".
            id_column (str): Column name used as the identifier in the dataset.
                Default is "id".
            perform_removal (bool): Whether to remove duplicates from the dataset.
                Default is False.
            logger (Union[logging.Logger, str]): Existing logger to log to, or a path to a log directory.
                Default is "./".
        """
        super().__init__(
            id_field=id_column,
            text_field=input_column,
            input_backend="cudf",
            logger=logger,
            perform_removal=perform_removal,
            profile_dir=config.profile_dir,
            cache_dir=config.cache_dir,
        )
        self.config = config
        embedding_output_dir = os.path.join(
            self.config.cache_dir, config.embeddings_save_loc
        )
        clustering_output_dir = os.path.join(
            self.config.cache_dir, config.clustering_save_loc
        )

        self.embedding_creator = EmbeddingCreator(
            embedding_model_name_or_path=config.embedding_model_name_or_path,
            embedding_batch_size=config.embedding_batch_size,
            embedding_output_dir=embedding_output_dir,
            embedding_max_mem_gb=config.embedding_max_mem_gb,
            embedding_pooling_strategy=config.embedding_pooling_strategy,
            input_column=input_column,
            embedding_column=config.embedding_column,
            write_embeddings_to_disk=config.write_embeddings_to_disk,
            write_to_filename=config.write_to_filename,
            logger=logger,
            profile_dir=self.config.profile_dir,
        )
        self.clustering_model = ClusteringModel(
            id_column=id_column,
            max_iter=config.max_iter,
            n_clusters=config.n_clusters,
            clustering_output_dir=clustering_output_dir,
            embedding_column=config.embedding_column,
            clustering_input_partition_size=config.clustering_input_partition_size,
            logger=logger,
            profile_dir=self.config.profile_dir,
        )
        self.semantic_cluster_dedup = SemanticClusterLevelDedup(
            n_clusters=config.n_clusters,
            emb_by_clust_dir=os.path.join(
                clustering_output_dir, "embs_by_nearest_center"
            ),
            id_column=id_column,
            which_to_keep=config.which_to_keep,
            sim_metric=config.sim_metric,
            batched_cosine_similarity=config.batched_cosine_similarity,
            output_dir=clustering_output_dir,
            embedding_column=config.embedding_column,
            logger=logger,
            profile_dir=self.config.profile_dir,
        )
        self.eps_to_extract = config.eps_to_extract

    def identify_duplicates(self, dataset: DocumentDataset) -> DocumentDataset:
        """
        Identify duplicates in the dataset. Returns a list of ids that are duplicates to each other.
        """
        embeddings_dataset = self.embedding_creator(dataset)
        self.clustering_model(embeddings_dataset)
        self.semantic_cluster_dedup.compute_semantic_match_dfs()
        return self.semantic_cluster_dedup.extract_dedup_data(
            eps_to_extract=self.eps_to_extract
        )

    def remove(
        self, dataset: DocumentDataset, duplicates_to_remove: DocumentDataset
    ) -> DocumentDataset:
        """
        Remove duplicates from the dataset.
        """
        result = remove_duplicates(
            dataset.df,
            duplicates_to_remove.df,
            self.id_field,
            group_field=None,
            perform_shuffle=False,
        )
        return DocumentDataset(result)
