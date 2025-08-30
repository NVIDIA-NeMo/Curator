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
Monolithic Text Semantic Deduplication Workflow.

This module contains a complete end-to-end workflow for text semantic deduplication:
1. Embedding generation from text data
2. Semantic deduplication using clustering and pairwise similarity
3. Optional duplicate removal based on identified duplicates
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

# Nemo Curator imports
from nemo_curator.backends.base import BaseExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.id_generator import (
    CURATOR_DEDUP_ID_STR,
    create_id_generator_actor,
    kill_id_generator_actor,
    write_id_generator_to_disk,
)
from nemo_curator.stages.deduplication.semantic.ranking import RankingStrategy
from nemo_curator.stages.deduplication.semantic.workflow import SemanticDeduplicationWorkflow
from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow
from nemo_curator.stages.text.embedders import EmbeddingCreatorStage
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.utils.file_utils import create_or_overwrite_dir


@dataclass
class TextSemanticDeduplicationWorkflow:
    """
    Monolithic workflow for end-to-end text semantic deduplication.

    This workflow combines:
    1. Text embedding generation (configurable executor)
    2. Semantic deduplication (configurable executor for pairwise stage)
    3. Duplicate removal (configurable executor)

    Supports flexible executor configuration - can use a single executor for all stages
    or different executors for different phases.
    """

    # Input/Output configuration
    input_path: str | list[str]
    output_path: str
    cache_path: str | None = None
    perform_removal: bool = True
    # Embedding generation parameters
    text_field: str = "text"
    embedding_field: str = "embeddings"
    model_identifier: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_max_seq_length: int = 512
    embedding_max_chars: int | None = None
    embedding_padding_side: Literal["left", "right"] = "right"
    embedding_pooling: Literal["mean_pooling", "last_token"] = "mean_pooling"
    embedding_model_inference_batch_size: int = 256
    hf_token: str | None = None
    # Semantic deduplication parameters
    n_clusters: int = 100
    id_field: str = CURATOR_DEDUP_ID_STR
    embedding_dim: int | None = None
    metadata_fields: list[str] | None = None
    distance_metric: Literal["cosine", "l2"] = "cosine"
    which_to_keep: Literal["hard", "easy", "random"] = "hard"
    eps: float | None = 0.01
    # K-means clustering parameters
    kmeans_max_iter: int = 300
    kmeans_tol: float = 1e-4
    kmeans_random_state: int = 42
    kmeans_init: str = "k-means||"
    kmeans_n_init: int | Literal["auto"] = 1
    kmeans_oversampling_factor: float = 2.0
    kmeans_max_samples_per_batch: int = 1 << 15  # 32768
    # Pairwise similarity parameters
    ranking_strategy: RankingStrategy | None = None
    pairwise_batch_size: int = 1024
    _duplicates_num_row_groups_hint: int | None = None
    # ID generator parameters
    use_id_generator: bool = False
    id_generator_state_file: str | None = None
    # I/O parameters
    input_filetype: Literal["jsonl", "parquet"] = "parquet"
    input_file_extensions: list[str] | None = None
    input_files_per_partition: int | None = None
    input_blocksize: int | None = None
    output_filetype: Literal["jsonl", "parquet"] = "parquet"
    output_file_extension: str | None = None
    output_fields: list[str] | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    cache_kwargs: dict[str, Any] = field(default_factory=dict)
    write_kwargs: dict[str, Any] = field(default_factory=dict)
    # Execution parameters
    verbose: bool = True
    clear_output: bool = True
    """
    Initialize the text semantic deduplication workflow.

    Args:
        input_path: Path(s) to input files containing text data
        output_path: Directory to write deduplicated (or ids to remove) output
        cache_path: Directory to cache intermediate results (embeddings, kmeans, pairwise, etc.)
        perform_removal: Whether to perform duplicate removal (True) or just identify duplicates (False)

        # Embedding generation parameters
        text_field: Name of the text field in input data
        embedding_field: Name of the embedding field to create
        model_identifier: HuggingFace model identifier for embeddings
        embedding_max_seq_length: Maximum sequence length for tokenization
        embedding_max_chars: Maximum number of characters for tokenization
        embedding_padding_side: Padding side for tokenization
        embedding_pooling: Pooling strategy for embeddings
        embedding_model_inference_batch_size: Batch size for model inference
        hf_token: HuggingFace token for private models

        # Semantic deduplication parameters
        n_clusters: Number of clusters for K-means
        id_field: Name of the ID field in the data
        embedding_dim: Embedding dimension (for memory estimation)
        metadata_fields: List of metadata field names to preserve
        distance_metric: Distance metric for similarity ("cosine" or "l2")
        which_to_keep: Strategy for ranking within clusters ("hard", "easy", "random")
        eps: Epsilon value for duplicate identification (None to skip)
        kmeans_max_iter: Maximum number of iterations for K-means clustering
        kmeans_tol: Tolerance for K-means convergence
        kmeans_random_state: Random state for K-means (None for random)
        kmeans_init: Initialization method for K-means centroids
        kmeans_n_init: Number of K-means initialization runs
        kmeans_oversampling_factor: Oversampling factor for K-means
        kmeans_max_samples_per_batch: Maximum samples per batch for K-means
        ranking_strategy: Custom ranking strategy for documents within clusters (None uses which_to_keep/distance_metric)
        pairwise_batch_size: Batch size for pairwise similarity computation
        _duplicates_num_row_groups_hint: Hint for number of row groups in duplicates output

        # ID generator parameters
        use_id_generator: Whether to use ID generator for document IDs
        id_generator_state_file: Path to save/load ID generator state (auto-generated if None)

        # I/O parameters
        input_files_per_partition: Number of files per partition for reading
        input_blocksize: Blocksize for reading files
        input_filetype: Type of input files ("jsonl" or "parquet")
        input_file_extensions: List of file extensions to process
        output_filetype: Type of output files ("jsonl" or "parquet")
        output_file_extension: File extension for output files (None for default)
        output_fields: List of fields to include in final output (None for all fields)
        read_kwargs: Keyword arguments for reading files
        cache_kwargs: Keyword arguments for cache operations and storage
        write_kwargs: Keyword arguments for writing files

        # Execution parameters
        verbose: Enable verbose output
        clear_output: Clear output directory before running
    """

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""

        # Core paths
        self.cache_path = self.cache_path or self.output_path

        # Intermediate paths
        self.embeddings_path = os.path.join(self.cache_path, "embeddings")
        self.semantic_dedup_path = os.path.join(self.cache_path, "semantic_dedup")
        # Output paths
        self.duplicates_path = None if self.eps is None else os.path.join(self.output_path, "duplicates")
        self.deduplicated_output_path = (
            None if not self.perform_removal else os.path.join(self.output_path, "deduplicated")
        )
        self.id_generator_state_file = os.path.join(self.output_path, "semantic_id_generator.json")

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate workflow configuration."""
        if self.perform_removal and self.eps is None:
            msg = "perform_removal=True but eps=None. Without eps, duplicates can't be identified. "
            msg += "Either set eps or set perform_removal=False"
            raise ValueError(msg)

        if self.use_id_generator and self.id_field != CURATOR_DEDUP_ID_STR:
            msg = "use_id_generator=True but id_field is not CURATOR_DEDUP_ID_STR. "
            msg += "ID generator only works with CURATOR_DEDUP_ID_STR"
            raise ValueError(msg)
        elif not self.use_id_generator and self.id_field == CURATOR_DEDUP_ID_STR:
            msg = "use_id_generator=False but id_field is CURATOR_DEDUP_ID_STR."
            logger.warning(msg)

    def _setup_directories(self) -> None:
        """Setup output directories."""
        if self.clear_output:
            create_or_overwrite_dir(self.output_path, storage_options=self.write_kwargs.get("storage_options"))

        # Cache paths
        create_or_overwrite_dir(self.embeddings_path, storage_options=self.cache_kwargs.get("storage_options"))
        create_or_overwrite_dir(self.semantic_dedup_path, storage_options=self.cache_kwargs.get("storage_options"))

        # Output paths
        if self.duplicates_path is not None:
            create_or_overwrite_dir(self.duplicates_path, storage_options=self.write_kwargs.get("storage_options"))
        if self.deduplicated_output_path is not None:
            create_or_overwrite_dir(
                self.deduplicated_output_path, storage_options=self.write_kwargs.get("storage_options")
            )

    def _run_embedding_generation(self, executor: BaseExecutor) -> list[Any]:
        """Run embedding generation stage."""
        if self.verbose:
            logger.info("Starting embedding generation stage...")

        pipeline = Pipeline(
            name="text_semantic_dedup_embedding",
            description="Generate embeddings from text data for semantic deduplication",
        )

        # Reader stage
        if self.input_filetype == "jsonl":
            reader = JsonlReader(
                file_paths=self.input_path,
                files_per_partition=self.input_files_per_partition,
                blocksize=self.input_blocksize,
                fields=(
                    ([self.id_field] if not self.use_id_generator else [])
                    + [self.text_field]
                    + (self.metadata_fields or [])
                ),
                file_extensions=self.input_file_extensions,
                _generate_ids=self.use_id_generator,
                read_kwargs=self.read_kwargs,
            )
        elif self.input_filetype == "parquet":
            reader = ParquetReader(
                file_paths=self.input_path,
                files_per_partition=self.input_files_per_partition,
                blocksize=self.input_blocksize,
                fields=(
                    ([self.id_field] if not self.use_id_generator else [])
                    + [self.text_field]
                    + (self.metadata_fields or [])
                ),
                file_extensions=self.input_file_extensions,
                read_kwargs=self.read_kwargs,
                _generate_ids=self.use_id_generator,
            )
        else:
            msg = f"Input filetype {self.input_filetype} not supported yet"
            raise NotImplementedError(msg)

        pipeline.add_stage(reader)

        # Embedding generation stage
        embedding_stage = EmbeddingCreatorStage(
            model_identifier=self.model_identifier,
            text_field=self.text_field,
            embedding_field=self.embedding_field,
            max_chars=self.embedding_max_chars,
            max_seq_length=self.embedding_max_seq_length,
            padding_side=self.embedding_padding_side,
            embedding_pooling=self.embedding_pooling,
            model_inference_batch_size=self.embedding_model_inference_batch_size,
            hf_token=self.hf_token,
        )
        pipeline.add_stage(embedding_stage)

        # Writer stage
        if self.output_filetype == "parquet":
            writer = ParquetWriter(
                path=self.embeddings_path,
                fields=[self.id_field, self.embedding_field] + (self.metadata_fields or []),
                write_kwargs=self.cache_kwargs,
            )
        else:
            msg = f"Output filetype {self.output_filetype} not supported yet"
            raise NotImplementedError(msg)

        pipeline.add_stage(writer)

        return pipeline.run(executor)

    def _run_semantic_deduplication(self, executor: BaseExecutor) -> dict[str, Any]:
        """Run semantic deduplication stage."""
        if self.verbose:
            logger.debug("Starting semantic deduplication stage...")

        workflow = SemanticDeduplicationWorkflow(
            input_path=self.embeddings_path,
            cache_path=self.semantic_dedup_path,
            output_path=self.output_path,
            n_clusters=self.n_clusters,
            # Core data configuration
            id_field=self.id_field,
            embedding_field=self.embedding_field,
            embedding_dim=self.embedding_dim,
            metadata_fields=self.metadata_fields,
            # K-means clustering parameters
            max_iter=self.kmeans_max_iter,
            tol=self.kmeans_tol,
            random_state=self.kmeans_random_state,
            init=self.kmeans_init,
            n_init=self.kmeans_n_init,
            oversampling_factor=self.kmeans_oversampling_factor,
            max_samples_per_batch=self.kmeans_max_samples_per_batch,
            # Pairwise similarity parameters
            distance_metric=self.distance_metric,
            which_to_keep=self.which_to_keep,
            ranking_strategy=self.ranking_strategy,
            pairwise_batch_size=self.pairwise_batch_size,
            # Duplicate identification parameters (optional)
            eps=self.eps,
            _duplicates_num_row_groups_hint=self._duplicates_num_row_groups_hint,
            # I/O and storage parameters
            read_kwargs=self.cache_kwargs,
            write_kwargs=self.cache_kwargs,
            clear_output=False,  # since the init of the workflow clears the output path
            verbose=self.verbose,
        )

        return workflow.run(pairwise_executor=executor)

    def _run_duplicate_removal(self, executor: BaseExecutor) -> list[Any]:
        """Run duplicate removal stage."""
        if not self.perform_removal:
            if self.verbose:
                logger.info("Skipping duplicate removal (perform_removal=False)")
            return []

        if self.verbose:
            logger.debug("Starting duplicate removal stage...")

        # Find the duplicates file from semantic deduplication
        workflow = TextDuplicatesRemovalWorkflow(
            # Use the original dataset as input so final outputs have original columns
            input_path=self.input_path,
            ids_to_remove_path=self.duplicates_path,
            output_path=self.deduplicated_output_path,
            input_filetype=self.input_filetype,
            input_id_field=self.id_field,
            input_files_per_partition=self.input_files_per_partition,
            input_blocksize=self.input_blocksize,
            input_file_extensions=self.input_file_extensions,
            input_kwargs=self.read_kwargs,
            # Ids to remove args
            ids_to_remove_duplicate_id_field="id",
            ids_to_remove_read_kwargs=self.write_kwargs,
            # ID generator parameters
            id_generator_path=self.id_generator_state_file if self.use_id_generator else None,
            id_generator_storage_options=self.write_kwargs.get("storage_options"),
            # Output args
            output_filetype=self.output_filetype,
            output_file_extension=self.output_file_extension,
            output_kwargs=self.write_kwargs,
            output_fields=self.output_fields,
            output_mode="ignore",
        )

        return workflow.run(executor=executor)

    def _log_configuration(self) -> None:
        """Log workflow configuration."""
        logger.info("=" * 80)
        logger.info("TEXT SEMANTIC DEDUPLICATION WORKFLOW CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Input path: {self.input_path}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Perform removal: {self.perform_removal}")

        logger.info("Embedding generation:")
        logger.info(f"  - Model: {self.model_identifier}")
        logger.info(f"  - Text field: {self.text_field}")
        logger.info(f"  - Embedding field: {self.embedding_field}")
        logger.info(f"  - Max sequence length: {self.embedding_max_seq_length}")
        logger.info(f"  - Batch size: {self.embedding_model_inference_batch_size}")
        logger.info(f"  - Executor: {type(self.embedding_executor).__name__}")

        logger.info("Semantic deduplication:")
        logger.info(f"  - Number of clusters: {self.n_clusters}")
        logger.info(f"  - ID field: {self.id_field}")
        logger.info(f"  - Distance metric: {self.distance_metric}")
        logger.info(f"  - Which to keep: {self.which_to_keep}")
        logger.info(f"  - Epsilon (similarity threshold): {self.eps}")
        logger.info(f"  - Pairwise executor: {type(self.pairwise_executor).__name__}")

        if self.perform_removal:
            logger.info("Duplicate removal:")
            logger.info(f"  - Removal executor: {type(self.removal_executor).__name__}")

        logger.info(f"Use ID generator: {self.use_id_generator}")
        if self.use_id_generator:
            logger.info(f"  - ID generator state file: {self.id_generator_state_file}")

        logger.info("=" * 80)

    def run(  # noqa: C901, PLR0912, PLR0915
        self, executor: BaseExecutor | tuple[BaseExecutor, BaseExecutor, BaseExecutor] | None = None
    ) -> dict[str, Any]:
        """
        Run the complete text semantic deduplication workflow.

        Returns:
            Dictionary with results and timing information from all stages
        """

        if isinstance(executor, tuple):
            if len(executor) != 3:  # noqa: PLR2004
                msg = f"Expected 3 executors in tuple, got {len(executor)}"
                raise ValueError(msg)
            embedding_executor, pairwise_executor, removal_executor = executor
        else:
            # Use same executor for all stages
            if executor is None:
                from nemo_curator.backends.xenna import XennaExecutor

                executor = XennaExecutor()
            embedding_executor = pairwise_executor = removal_executor = executor

        # Expose executors as attributes for logging and downstream access
        self.embedding_executor = embedding_executor
        self.pairwise_executor = pairwise_executor
        self.removal_executor = removal_executor

        total_start_time = time.time()

        try:
            # Setup
            self._setup_directories()
            if self.verbose:
                self._log_configuration()

            # Setup ID generator if needed
            if self.use_id_generator:
                logger.debug(f"Setting up ID generator, state will be saved to: {self.id_generator_state_file}")
                try:
                    create_id_generator_actor()
                except ValueError as e:
                    if "already taken" in str(e):
                        logger.debug("ID generator actor already exists, using existing actor")
                    else:
                        raise

            # Stage 1: Embedding generation
            embedding_start_time = time.time()
            embedding_results = self._run_embedding_generation(embedding_executor)
            embedding_end_time = time.time()
            embedding_time = embedding_end_time - embedding_start_time
            logger.success(f"Embedding generation completed in {embedding_time:.2f} seconds")

            if self.use_id_generator:
                try:
                    write_id_generator_to_disk(self.id_generator_state_file)
                    if self.verbose:
                        logger.debug(f"ID generator state saved for removal stage to: {self.id_generator_state_file}")
                except Exception as save_error:
                    logger.error(f"Error saving ID generator state: {save_error}")
                    raise
                finally:
                    if self.verbose:
                        logger.debug("Killing ID generator actor...")
                    kill_id_generator_actor()

            # Stage 2: Semantic deduplication
            semantic_start_time = time.time()
            semantic_results = self._run_semantic_deduplication(pairwise_executor)
            semantic_end_time = time.time()
            semantic_time = semantic_end_time - semantic_start_time

            logger.success(f"Semantic deduplication completed in {semantic_time:.2f} seconds")

            # Stage 3: Duplicate removal (optional)
            removal_results = []
            removal_time = 0.0
            if self.perform_removal:
                removal_start_time = time.time()
                removal_results = self._run_duplicate_removal(removal_executor)
                removal_end_time = time.time()
                removal_time = removal_end_time - removal_start_time

                logger.success(f"Duplicate removal completed in {removal_time:.2f} seconds")

            # Calculate total time
            total_end_time = time.time()
            total_time = total_end_time - total_start_time

            # Log final summary
            if self.verbose:
                logger.success("=" * 80)
                logger.success("TEXT SEMANTIC DEDUPLICATION WORKFLOW COMPLETED")
                logger.success("=" * 80)
                logger.success(f"Total execution time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
                logger.info(f"Embedding generation time: {embedding_time:.2f} seconds")
                logger.info(f"Semantic deduplication time: {semantic_time:.2f} seconds")
                if self.perform_removal:
                    logger.info(f"Duplicate removal time: {removal_time:.2f} seconds (removed {removal_results} rows)")
                if semantic_results.get("total_duplicates_identified", 0) > 0:
                    logger.success(
                        f"Total documents identified as duplicates: {semantic_results['total_duplicates_identified']}"
                    )
            logger.success("=" * 80)

        except Exception as e:
            logger.error(f"Text semantic deduplication workflow failed: {e}")
            raise

        return {
            "total_execution_time": total_time,
            "embedding_execution_time": embedding_time,
            "semantic_execution_time": semantic_time,
            "removal_execution_time": removal_time,
            "embedding_results": embedding_results,
            "semantic_results": semantic_results,
            "removal_results": removal_results,
            "embeddings_path": self.embeddings_path,
            "semantic_dedup_path": self.semantic_dedup_path,
            "final_output_path": self.deduplicated_output_path if self.perform_removal else None,
        }
