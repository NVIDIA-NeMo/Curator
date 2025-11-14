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

"""
End-to-End Semantic Deduplication Pipeline for Ray Curator.

This module contains the complete semantic deduplication workflow:
1. K-means clustering on embedding data (always uses RayActorPoolExecutor)
2. Pairwise similarity computation within clusters + duplicate identification (configurable executor)
"""

import os
import time
from typing import Any, Literal

import numpy as np
from loguru import logger

# Ray Curator imports
from nemo_curator.backends.base import BaseExecutor
from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline

# Stage imports
from nemo_curator.stages.deduplication.semantic.identify_duplicates import IdentifyDuplicatesStage
from nemo_curator.stages.deduplication.semantic.kmeans import KMeansStage
from nemo_curator.stages.deduplication.semantic.pairwise import PairwiseStage
from nemo_curator.stages.deduplication.semantic.ranking import RankingStrategy
from nemo_curator.utils.file_utils import create_or_overwrite_dir


class SemanticDeduplicationWorkflow:
    """
    End-to-End Semantic Deduplication Workflow.
    It consists of the following stages:
    - KMeansStage
        Takes the input path (embeddings) and clusters the embeddings into n_clusters.
        Writes data partitioned by centroid to cache_path.
    - PairwiseStage
        Computes pairwise similarity between all embeddings in each cluster.
        Takes the output of KMeansStage and computes pairwise similarity between all embeddings in each cluster.
        This is written to cache_path.
    - IdentifyDuplicatesStage (optional)
        Identifies duplicates based on the pairwise similarity scores.
        Runs only if eps is provided.
        This is written to output_path.
    """

    def __init__(  # noqa: PLR0913
        self,
        # required args
        input_path: str | list[str],
        output_path: str,
        n_clusters: int,
        cache_path: str | None = None,
        # Core data configuration
        id_field: str = "id",
        embedding_field: str = "embeddings",
        embedding_dim: int | None = None,
        metadata_fields: list[str] | None = None,
        input_filetype: Literal["parquet", "jsonl"] = "parquet",
        input_file_extensions: list[str] | None = None,
        # K-means clustering parameters
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
        init: Literal["k-means||", "random"] | np.ndarray = "k-means||",
        n_init: int | Literal["auto"] = 1,
        oversampling_factor: float = 2.0,
        max_samples_per_batch: int = 1 << 15,
        # Pairwise similarity parameters
        distance_metric: Literal["cosine", "l2"] = "cosine",
        which_to_keep: Literal["hard", "easy", "random"] = "hard",
        ranking_strategy: RankingStrategy | None = None,
        pairwise_batch_size: int = 1024,
        # Duplicate identification parameters (optional)
        eps: float | None = None,
        _duplicates_num_row_groups_hint: int | None = None,
        # I/O and storage parameters
        read_kwargs: dict[str, Any] | None = None,
        cache_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        clear_output: bool = True,
        # Execution parameters
        verbose: bool = True,
    ):
        """
        Initialize the semantic deduplication workflow.

        Args:
            input_path: Directory or list of directories containing input files with embeddings
            output_path: Directory to write output files (i.e. ids to remove)
            n_clusters: Number of clusters for K-means
            cache_path: Directory to write cache files (i.e. kmeans and pairwise results)
                If None, will be set to output_path

            # Core data configuration
            id_field: Name of the ID field in the data
            embedding_field: Name of the embedding field in the data
            embedding_dim: Embedding dimension (for memory estimation)
            metadata_fields: List of metadata field names to preserve in output
            input_filetype: Type of input files ("parquet" or "jsonl")
            input_file_extensions: List of file extensions to process

            # K-means clustering parameters
            max_iter: Maximum number of K-means iterations
            tol: Tolerance for K-means convergence
            random_state: Random seed for K-means
            init: K-means initialization method
            n_init: Number of K-means initializations
            oversampling_factor: K-means++ oversampling factor
            max_samples_per_batch: Max samples per batch for K-means
            distance_metric: Distance metric for similarity ("cosine" or "l2")

            # Pairwise similarity parameters
            which_to_keep: Strategy for ranking within clusters ("hard", "easy", "random")
            ranking_strategy: Custom ranking strategy (overrides which_to_keep)
            pairwise_batch_size: Batch size for pairwise similarity computation

            # Duplicate identification parameters (optional)
            eps: Epsilon value for duplicate identification
            _duplicates_num_row_groups_hint: Number of row groups hint for duplicate removal

            # I/O and storage parameters
            read_kwargs: Keyword arguments for reading files (including storage_options)
            write_kwargs: Keyword arguments for writing files (including storage_options)
            clear_output: Clear output directory before running
            # Execution parameters
            verbose: Enable verbose output
        """
        # Core paths and configuration
        self.input_path = input_path
        self.output_path = output_path
        self.cache_path = cache_path or output_path

        self.kmeans_output_path = os.path.join(self.cache_path, "kmeans_results")
        self.pairwise_output_path = os.path.join(self.cache_path, "pairwise_results")
        self.duplicates_output_path = os.path.join(self.output_path, "duplicates")

        self.n_clusters = n_clusters

        # Data configuration
        self.id_field = id_field
        self.embedding_field = embedding_field
        self.embedding_dim = embedding_dim
        self.metadata_fields = metadata_fields
        self.input_filetype = input_filetype
        self.input_file_extensions = input_file_extensions

        # K-means parameters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.oversampling_factor = oversampling_factor
        self.max_samples_per_batch = max_samples_per_batch

        # Pairwise similarity parameters
        self.distance_metric = distance_metric
        self.which_to_keep = which_to_keep
        self.ranking_strategy = ranking_strategy
        self.pairwise_batch_size = pairwise_batch_size

        # Duplicate identification parameters
        self.eps = eps
        self._duplicates_num_row_groups_hint = _duplicates_num_row_groups_hint

        # I/O parameters
        self.read_kwargs = read_kwargs.copy() if read_kwargs else {}
        self.write_kwargs = write_kwargs.copy() if write_kwargs else {}
        self.cache_kwargs = cache_kwargs.copy() if cache_kwargs else self.write_kwargs.copy()
        self.clear_output = clear_output

        # Execution parameters
        self.verbose = verbose

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration."""
        # Note: Input path validation is handled by KMeansStage and FilePartitioningStage
        # Note: duplicates_path is now automatically created, no validation needed

        # Validate distance_metric
        if self.ranking_strategy is None:
            if self.distance_metric not in {"cosine", "l2"}:
                msg = f"Invalid distance_metric: {self.distance_metric}. Must be 'cosine' or 'l2'"
                raise ValueError(msg)

            # Validate which_to_keep
            if self.which_to_keep not in {"hard", "easy", "random"}:
                msg = f"Invalid which_to_keep: {self.which_to_keep}. Must be 'hard', 'easy', or 'random'"
                raise ValueError(msg)
        elif self.distance_metric or self.which_to_keep:
            msg = "distance_metric and which_to_keep are not used when ranking_strategy is provided"
            logger.warning(msg)
        else:
            cols_needed_for_ranking = self.ranking_strategy.metadata_cols
            missing_cols = set(cols_needed_for_ranking) - set(self.metadata_fields)
            if missing_cols:
                msg = f"Metadata fields {missing_cols} are required for ranking"
                raise ValueError(msg)

    def _setup_directories(self) -> None:
        """Setup output directories with fsspec compliance."""
        if self.clear_output:
            storage_options = self.write_kwargs.get("storage_options")
            create_or_overwrite_dir(self.output_path, storage_options=storage_options)
            create_or_overwrite_dir(self.kmeans_output_path, storage_options=storage_options)
            create_or_overwrite_dir(self.pairwise_output_path, storage_options=storage_options)
            if self.eps is not None:
                create_or_overwrite_dir(self.duplicates_output_path, storage_options=storage_options)

    def _run_kmeans_stage(self) -> list[Any]:
        """Run K-means clustering stage (always uses RayActorPoolExecutor)."""
        logger.info("Starting K-means clustering stage (RayActorPoolExecutor)...")

        pipeline = Pipeline(
            name="semantic_dedup_kmeans", description="K-means clustering stage of semantic deduplication"
        )

        kmeans_stage = KMeansStage(
            n_clusters=self.n_clusters,
            id_field=self.id_field,
            embedding_field=self.embedding_field,
            input_path=self.input_path,
            output_path=self.kmeans_output_path,
            metadata_fields=self.metadata_fields,
            embedding_dim=self.embedding_dim,
            input_filetype=self.input_filetype,
            input_file_extensions=self.input_file_extensions,
            verbose=self.verbose,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            init=self.init,
            n_init=self.n_init,
            oversampling_factor=self.oversampling_factor,
            max_samples_per_batch=self.max_samples_per_batch,
            read_kwargs=self.read_kwargs,
            write_kwargs=self.cache_kwargs,
        )
        pipeline.add_stage(kmeans_stage)

        # Always use RayActorPoolExecutor for K-means
        executor = RayActorPoolExecutor()
        return pipeline.run(executor)

    def _run_pairwise_stage(self, pairwise_executor: BaseExecutor | None = None) -> list[Any]:
        """Run pairwise similarity + duplicate identification stage."""
        logger.info(f"Starting pairwise similarity stage ({pairwise_executor})...")

        pipeline = Pipeline(
            name="semantic_dedup_pairwise",
            description="Pairwise similarity computation stage of semantic deduplication",
        )

        # Stage 1: Pairwise similarity computation
        pairwise_stage = PairwiseStage(
            id_field=self.id_field,
            embedding_field=self.embedding_field,
            input_path=self.kmeans_output_path,
            output_path=self.pairwise_output_path,
            ranking_strategy=self.ranking_strategy,
            embedding_dim=self.embedding_dim,
            pairwise_batch_size=self.pairwise_batch_size,
            verbose=self.verbose,
            which_to_keep=self.which_to_keep,
            sim_metric=self.distance_metric,
            random_seed=self.random_state,
            read_kwargs=self.cache_kwargs,
            write_kwargs=self.cache_kwargs,
        )
        pipeline.add_stage(pairwise_stage)

        # Stage 2: Optional duplicate identification stage
        if self.eps is not None:
            identify_duplicates_stage = IdentifyDuplicatesStage(
                output_path=self.duplicates_output_path,
                eps=self.eps,
                _num_row_groups_hint=self._duplicates_num_row_groups_hint,
                verbose=self.verbose,
                read_kwargs=self.cache_kwargs,
                write_kwargs=self.write_kwargs,
            )
            pipeline.add_stage(identify_duplicates_stage)

        return pipeline.run(pairwise_executor)

    def _log_configuration(self, pairwise_executor: BaseExecutor | None = None) -> None:
        """Log workflow configuration."""
        logger.info("=" * 60)
        logger.info("SEMANTIC DEDUPLICATION WORKFLOW CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Input path: {self.input_path}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"K-means output path: {self.kmeans_output_path}")
        logger.info(f"Pairwise output path: {self.pairwise_output_path}")
        if self.eps is not None:
            logger.info(f"Duplicates output path: {self.duplicates_output_path}")
            logger.info(f"Epsilon (similarity threshold): {self.eps}")
        logger.info(f"Number of clusters: {self.n_clusters}")
        logger.info("K-means executor: RayActorPoolExecutor (fixed)")
        logger.info(f"Pairwise executor: {pairwise_executor}")
        logger.info(f"Input file type: {self.input_filetype}")
        logger.info(f"ID field: {self.id_field}")
        logger.info(f"Embedding field: {self.embedding_field}")
        logger.info(f"Distance metric: {self.distance_metric}")
        logger.info(f"Which to keep: {self.which_to_keep}")
        logger.info(f"Ranking strategy: {self.ranking_strategy}")
        logger.info(f"Pairwise batch size: {self.pairwise_batch_size}")
        logger.info(f"Random state: {self.random_state}")
        logger.info("=" * 60)

    def run(self, pairwise_executor: BaseExecutor | None = None) -> dict[str, Any]:
        """
        Run the complete semantic deduplication pipeline.

        Args:
            pairwise_executor: Executor for pairwise stage. Defaults to XennaExecutor().

        Returns:
            Dictionary with results and timing information
        """
        total_start_time = time.time()
        pairwise_executor = pairwise_executor or XennaExecutor()

        try:
            # Setup
            self._setup_directories()
            self._log_configuration(pairwise_executor)

            # Stage 1: K-means clustering (always RayActorPoolExecutor)
            kmeans_start_time = time.time()
            kmeans_results = self._run_kmeans_stage()
            kmeans_end_time = time.time()
            kmeans_time = kmeans_end_time - kmeans_start_time

            logger.success(f"K-means clustering completed in {kmeans_time:.2f} seconds")

            # Stage 2: Pairwise similarity + duplicate identification
            pairwise_start_time = time.time()
            pairwise_results = self._run_pairwise_stage(pairwise_executor)
            pairwise_end_time = time.time()
            pairwise_time = pairwise_end_time - pairwise_start_time

            logger.success(f"Pairwise similarity stage completed in {pairwise_time:.2f} seconds")

            # Calculate total time
            total_end_time = time.time()
            total_time = total_end_time - total_start_time

            # Count duplicates if identified
            total_duplicates = 0
            if self.eps is not None and pairwise_results:
                for task in pairwise_results:
                    if hasattr(task, "_metadata") and "num_removed" in task._metadata:
                        total_duplicates += task._metadata["num_removed"]

            # Log final summary
            logger.success("=" * 60)
            logger.success("SEMANTIC DEDUPLICATION COMPLETED")
            logger.success("=" * 60)
            logger.success(f"Total execution time: {total_time:.2f} seconds")
            logger.info(f"K-means time: {kmeans_time:.2f} seconds")
            logger.info(f"Pairwise time: {pairwise_time:.2f} seconds")
            if total_duplicates > 0:
                logger.success(f"Total documents identified as duplicates: {total_duplicates}")
                logger.info(f"Similarity threshold used: {1.0 - self.eps:.3f} (eps={self.eps})")
            elif self.eps is not None:
                logger.info(
                    f"No duplicates identified with similarity threshold of {1.0 - self.eps:.3f} (eps={self.eps})"
                )
            logger.success("=" * 60)

        except Exception as e:
            logger.error(f"Semantic deduplication pipeline failed: {e}")
            raise
        else:
            return {
                "total_execution_time": total_time,
                "kmeans_execution_time": kmeans_time,
                "pairwise_execution_time": pairwise_time,
                "kmeans_results": kmeans_results,
                "pairwise_results": pairwise_results,
                **({"total_duplicates_identified": total_duplicates} if self.eps is not None else {}),
            }
